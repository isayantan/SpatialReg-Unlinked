import torch
import torch.nn as nn
import numpy as np

def torch_log_det_jacobian(P, jitter=1e-5, return_intermediates=False):
    K = P.shape[0]
    assert P.shape == (K, K)

    logdet = 0.0

    # Initialize upper bounds and lower bounds
    ub_rows = torch.ones(K, K, dtype=P.dtype, device=P.device)
    ub_cols = torch.ones(K, K, dtype=P.dtype, device=P.device)
    lbs = torch.zeros(K, K, dtype=P.dtype, device=P.device)

    for i in range(K - 1):
        for j in range(K - 1):
            # Compute lower bound
            lbs_ij = ub_rows[i, j] - torch.sum(ub_cols[i, j + 1:])
            lbs = lbs.clone()
            lbs[i, j] = lbs_ij

            # Four cases
            if ub_rows[i, j] < ub_cols[i, j]:
                if lbs_ij > 0:
                    logdet -= torch.log(ub_rows[i, j] - lbs_ij)
                else:
                    if ub_rows[i, j] > 0:
                        logdet -= torch.log(ub_rows[i, j])
                    else:
                        logdet -= torch.log(torch.tensor(jitter, dtype=P.dtype, device=P.device))
            else:
                if lbs_ij > 0:
                    logdet -= torch.log(ub_cols[i, j] - lbs_ij)
                else:
                    if ub_cols[i, j] > 0:
                        logdet -= torch.log(ub_cols[i, j])
                    else:
                        logdet -= torch.log(torch.tensor(jitter, dtype=P.dtype, device=P.device))

            # Update upper bounds (avoid in-place ops)
            ub_rows = ub_rows.clone()
            ub_cols = ub_cols.clone()
            ub_rows[i, j + 1] = ub_rows[i, j] - P[i, j]
            ub_cols[i + 1, j] = ub_cols[i, j] - P[i, j]

        # Finish off the row
        ub_cols = ub_cols.clone()
        ub_cols[i + 1, -1] = ub_cols[i, -1] - P[i, -1]

    if return_intermediates:
        return logdet, ub_rows, ub_cols, lbs
    else:
        return logdet


def sample_doubly_stochastic_stable(Psi, tol=1e-4, verbose=False):
    N = Psi.shape[0] + 1
    Psi = torch.special.expit(Psi)
    assert Psi.shape == (N - 1, N - 1)
    
    P = torch.zeros(N, N)

    for i in range(N - 1):
        for j in range(N - 1):
            ub_row = 1.0 - torch.sum(P[i, :j])
            ub_col = 1.0 - torch.sum(P[:i, j])
            lb_rem = (1.0 - torch.sum(P[i, :j])) \
                     - (N - (j + 1)) + torch.sum(P[:i, j + 1:N - 1])
            
            ub = torch.minimum(ub_row, ub_col)
            lb = torch.maximum(torch.tensor(0.0), lb_rem)

            Psi_val = Psi[i, j]
            if (ub - lb) < tol:
                P[i, j] = tol * Psi_val
            else:
                P[i, j] = lb + (ub - lb) * Psi_val
        
        # Complete the row
        P[i, -1] = 1.0 - torch.sum(P[i, :-1])

    for j in range(N):
        P[-1, j] = 1.0 - torch.sum(P[:-1, j])
    
    return P


def check_doubly_stochastic_stable(P, tol=1e-4):
    N = P.shape[0]
    assert torch.allclose(P.sum(0), torch.ones(N), atol=N * tol)
    assert torch.allclose(P.sum(1), torch.ones(N), atol=N * tol)
    assert torch.min(P) >= -N * tol
    assert torch.max(P) <= 1 + N * tol
    print(f"P min: {P.min():.5f} max {P.max():.5f}")


class vi_piX(nn.Module):
    def __init__(self, n_locations):
        super(vi_piX, self).__init__()
        self.n_locations = n_locations
        # Initialize with a random doubly stochastic matrix
        self.MX = nn.Parameter(torch.rand(n_locations-1, n_locations-1), requires_grad=True)
        self.VX = nn.Parameter((-2)*torch.ones(n_locations-1, n_locations-1),requires_grad=True)
        self.current_M_X_star = (1/torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_X_star = torch.eye(n_locations, n_locations)

    def forward(self, Y, X, mu_lambda_beta,
                sigmasq_lambda_beta, M_S_star, mu_W,
                eta_X_sq, lambda_a2, lambda_b2, 
                tau_X=0.1, n_piX_sample=100, VX_ub=2, seed=100):
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # calculate ELBO
        elbo = 0.0
        current_M_X_star = torch.zeros(self.n_locations, self.n_locations)
        current_V_X_star = torch.zeros(self.n_locations, self.n_locations)
        for i in range(n_piX_sample):
            torch.manual_seed(i * seed)
            z = torch.randn((self.n_locations - 1)**2).reshape(self.n_locations - 1, self.n_locations - 1)
            Phi = self.MX + z * torch.sqrt(torch.special.expit(self.VX)*(VX_ub-0.01) + 0.01)
            Phi = Phi/tau_X
            P_Phi = sample_doubly_stochastic_stable(Phi, tol = 0.01)
            current_piX = P_Phi
            current_piX_sq = current_piX.T @ current_piX
            current_M_X_star += current_piX
            current_V_X_star += current_piX_sq

            # Computation for all regions
            term1 = 0.
            proj1 = mu_lambda_beta * (current_piX @ X.T).T
            resid2 = Y - (M_S_star @ mu_W.T).T
            varX = (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * torch.trace(X @ current_piX_sq @ X.T)
            term1 = (varX - 2* torch.trace(resid2.T @ proj1))

            coeff = -lambda_a2 / (2 * lambda_b2)
            total_term1 = coeff * term1

            # Second summation: over entries of pi_x
            x_mk_squared = current_piX.pow(2)
            x_mk_minus1_squared = (current_piX - 1).pow(2)

            exponent1 = -x_mk_squared / (2 * eta_X_sq)
            exponent2 = -x_mk_minus1_squared / (2 * eta_X_sq)

            # Stable log-sum-exp
            log_term = torch.logsumexp(torch.stack([exponent1, exponent2]), dim=0)
            total_log_term = (log_term).sum()

            # Final terms
            #jacobian
            log_det_term = - torch_log_det_jacobian(current_piX, return_intermediates= False)
            log_det_term -= torch.log(torch.tensor(tau_X)) * ((self.n_locations - 1)**2)
            log_det_term += ((torch.log(torch.special.expit(Phi)) + torch.log(torch.special.expit(-Phi))).sum())

            # Update ELBO
            elbo += total_term1 + total_log_term + log_det_term + 0.5* torch.log(torch.special.expit(self.VX)*(VX_ub-0.01) + 0.01).sum()
        
        elbo = elbo / n_piX_sample 
        self.current_M_X_star = current_M_X_star / n_piX_sample
        self.current_V_X_star = current_V_X_star / n_piX_sample
        return -elbo


class vi_piS(nn.Module):
    def __init__(self, n_locations, n_blocks):
        super(vi_piS, self).__init__()
        self.n_locations = n_locations
        self.n_blocks = n_blocks
        self.MS = nn.Parameter(torch.rand(n_locations-1, n_locations-1), requires_grad=True)
        self.VS = nn.Parameter((-2) * torch.ones(n_locations-1, n_locations-1, requires_grad=True))
        self.current_M_S_star = (1/torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_S_star = torch.eye(n_locations, n_locations)

    def forward(self, Y, X, mu_lambda_beta, M_X_star,
                lambda_a2, lambda_b2, mu_W, Sigma_W,
                eta_S_sq, tau_S=0.1, n_piS_sample=100, VS_ub = 2, seed=100):

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        elbo = 0.0
        current_M_S_star = torch.zeros(self.n_locations, self.n_locations)
        current_V_S_star = torch.zeros(self.n_locations, self.n_locations)

        for i in range(n_piS_sample):
            torch.manual_seed(i * seed)
            z = torch.randn((self.n_locations - 1)**2).reshape(self.n_locations - 1, self.n_locations - 1)
            Phi = self.MS + z * torch.sqrt(torch.special.expit(self.VS)*(VS_ub-0.01) + 0.01)
            Phi = Phi/tau_S
            P_Phi = sample_doubly_stochastic_stable(Phi, tol = 0.01)
            current_piS = P_Phi
            current_piS_sq = current_piS.T @ current_piS
            current_M_S_star += current_piS
            current_V_S_star += current_piS_sq

            coeff = -lambda_a2 / (2 * lambda_b2)
            resid1 = Y - mu_lambda_beta * (M_X_star @ X.T).T
            proj2 = (current_piS @ mu_W.T).T
            mu_W_flat = mu_W.flatten()
            V_W = Sigma_W + torch.outer(mu_W_flat, mu_W_flat)
            varS =  torch.trace(torch.block_diag(*[current_piS_sq] * self.n_blocks) @ V_W) 
            total_term1 = coeff * (varS - 2 * torch.trace(resid1.T @ proj2))

            # Prior term over pi_S entries
            s_mk_squared = current_piS.pow(2)
            s_mk_minus1_squared = (current_piS - 1).pow(2)

            exponent1 = -s_mk_squared / (2 * eta_S_sq)
            exponent2 = -s_mk_minus1_squared / (2 * eta_S_sq)

            log_term = torch.logsumexp(torch.stack([exponent1, exponent2]), dim=0)
            total_log_term = log_term.sum()

            #jacobian
            log_det_term = - torch_log_det_jacobian(current_piS, return_intermediates= False)
            log_det_term -= torch.log(torch.tensor(tau_S)) * ((self.n_locations - 1)**2)
            log_det_term += ((torch.log(torch.special.expit(Phi)) + torch.log(torch.special.expit(-Phi))).sum())

            # Update ELBO
            elbo += total_term1 + total_log_term + log_det_term + 0.5* torch.log(torch.special.expit(self.VS)*(VS_ub-0.01) + 0.01).sum()

        elbo = elbo / n_piS_sample
        self.current_M_S_star = current_M_S_star / n_piS_sample
        self.current_V_S_star = current_V_S_star / n_piS_sample
        return -elbo

