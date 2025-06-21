import torch
import torch.nn as nn
import numpy as np
from utils import round_to_perm, sinkhorn_logspace

class vi_piX(nn.Module):
    def __init__(self, n_locations):
        super(vi_piX, self).__init__()
        self.n_locations = n_locations
        # Initialize with a random doubly stochastic matrix
        random_matrix = torch.rand(n_locations, n_locations)
        log_doubly_stochastic = sinkhorn_logspace((random_matrix), niters=10)
        self.MX = nn.Parameter(log_doubly_stochastic, requires_grad=True)
        self.VX = nn.Parameter((-2)*torch.ones(n_locations, n_locations),requires_grad=True)
        self.current_M_X_star = (1/torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_X_star = torch.eye(n_locations, n_locations)
        #self.VX_unconstrained = nn.Parameter(torch.full((n_locations, n_locations), 0.2))
            
    def forward(self, Y, X, mu_lambda_beta,
                sigmasq_lambda_beta, M_S_star, mu_W,
                eta_X_sq, lambda_a2, lambda_b2, 
                tau_X=0.1, n_piX_sample=100, VX_ub=2, seed=100):
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)


        # calculate nearest doubly stochastic matrix to MX once
        log_MX = self.MX
        log_MX_tilde = sinkhorn_logspace(log_MX, niters=10)
        MX_tilde = torch.exp(log_MX_tilde)
    

        # Compute the ELBO
        B = Y.shape[0]
        elbo = 0.0
        current_M_X_star = torch.zeros(self.n_locations, self.n_locations)
        current_V_X_star = torch.zeros(self.n_locations, self.n_locations)
        for i in range(n_piX_sample):
            torch.manual_seed(i * seed)
            z = torch.randn(self.n_locations, self.n_locations)
            Phi = MX_tilde + torch.sqrt(torch.special.expit(self.VX)*(VX_ub-0.01) + 0.01) * z
            round_Phi = round_to_perm((Phi).detach().numpy())
            current_piX = tau_X * Phi + (1 - tau_X) * torch.tensor(round_Phi, dtype=Phi.dtype)  # Access the current sample of piS
            current_M_X_star += current_piX
            current_piX_sq = current_piX.T @ current_piX
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
            neg_log_tauX = self.n_locations ** 2 * torch.log(torch.tensor(tau_X))

            # Update ELBO
            #elbo += total_term1 + total_log_term + neg_log_tauX + 0.5 * self.VX.sum()
            elbo += total_term1 + total_log_term + neg_log_tauX + 0.5* torch.log(torch.special.expit(self.VX)*(VX_ub-0.01) + 0.01).sum()  # Ensure positive definiteness
            #elbo += total_term1 + neg_log_tauX + 2 * self.VX.sum()

        
        elbo = elbo / n_piX_sample 
        self.current_M_X_star = current_M_X_star / n_piX_sample
        self.current_V_X_star = current_V_X_star / n_piX_sample
        return -elbo


class vi_piS(nn.Module):
    def __init__(self, n_locations, n_blocks):
        super(vi_piS, self).__init__()
        self.n_locations = n_locations
        self.n_blocks = n_blocks
        self.MS = nn.Parameter(torch.log(1 / torch.tensor(n_locations)) * torch.ones(n_locations, n_locations, requires_grad=True))
        self.VS = nn.Parameter((-2) * torch.ones(n_locations, n_locations, requires_grad=True))
        self.current_M_S_star = (1/torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_S_star = torch.eye(n_locations, n_locations)

    def forward(self, Y, X, mu_lambda_beta, M_X_star,
                lambda_a2, lambda_b2, mu_W, Sigma_W,
                eta_S_sq, tau_S=0.1, n_piS_sample=100, VS_ub = 2, seed=100):

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
    
        # Sinkhorn to project to the doubly stochastic matrix
        log_MS = self.MS
        log_MS_tilde = sinkhorn_logspace(log_MS, niters=10)
        MS_tilde = torch.exp(log_MS_tilde)

        B = Y.shape[0]
        elbo = 0.0
        current_M_S_star = torch.zeros(self.n_locations, self.n_locations)
        current_V_S_star = torch.zeros(self.n_locations, self.n_locations)

        for i in range(n_piS_sample):
            torch.manual_seed(i * seed)
            z = torch.randn(self.n_locations, self.n_locations)
            Phi = MS_tilde + torch.sqrt(torch.special.expit(self.VS)*(VS_ub-0.01) + 0.01) * z
            round_Phi = round_to_perm((Phi).detach().numpy())
            current_piS = tau_S * Phi + (1 - tau_S) * torch.tensor(round_Phi, dtype=Phi.dtype)  # Access the current sample of piS

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

            # Final terms
            neg_log_tauS = (self.n_locations ** 2 ) * torch.log(torch.tensor(tau_S))

            # Update ELBO
            elbo += total_term1 + total_log_term + neg_log_tauS + 0.5* torch.log(torch.special.expit(self.VS)*(VS_ub-0.01) + 0.01).sum()

        elbo = elbo / n_piS_sample
        self.current_M_S_star = current_M_S_star / n_piS_sample
        self.current_V_S_star = current_V_S_star / n_piS_sample
        return -elbo

