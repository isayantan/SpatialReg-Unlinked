import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import round_to_perm, sinkhorn_logspace

class vi_piX(nn.Module):
    def __init__(self, n_locations, block_idx):
        super(vi_piX, self).__init__()
        self.n_locations = n_locations
        self.block_idx = block_idx
        # Initialize with a random doubly stochastic matrix
        self.MX = nn.Parameter(torch.log(1 / torch.tensor(n_locations)) * torch.ones(n_locations, n_locations, requires_grad=True))
        self.VX = nn.Parameter((-2) * torch.ones(n_locations, n_locations), requires_grad=True)
        self.current_M_X_star = (1 / torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_X_star = torch.eye(n_locations, n_locations)
        #self.VX_unconstrained = nn.Parameter(torch.full((n_locations, n_locations), 0.2))
            
    def forward(self, Y, X, mu_lambda_beta,
                sigmasq_lambda_beta, M_S_star, mu_W,
                eta_X_sq, lambda_a2, lambda_b2, 
                tau_X=0.1, n_piX_sample=100, VX_ub=2, seed=100):
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        elbo = 0
        # Generate permutation matrices for the specified block
        current_M_X_star = torch.zeros(self.n_locations, self.n_locations)
        current_V_X_star = torch.zeros(self.n_locations, self.n_locations)
        
        log_MX = self.MX
        log_MX_tilde = sinkhorn_logspace(log_MX, niters=10)
        MX_tilde = torch.exp(log_MX_tilde)
        
        for i in range(n_piX_sample):
            torch.manual_seed(i * seed)
            z = torch.randn(self.n_locations, self.n_locations)
            Phi = MX_tilde + torch.sqrt(torch.special.expit(self.VX) * (VX_ub - 0.01) + 0.01) * z
            round_Phi = round_to_perm((Phi - 0.95 * Phi.min()).detach().numpy())
            current_piX = tau_X * Phi + (1 - tau_X) * torch.tensor(round_Phi, dtype=Phi.dtype)
            current_M_X_star += current_piX
            current_piX_sq = current_piX.T @ current_piX
            current_V_X_star += current_piX_sq
            
            Y_b = Y[self.block_idx]
            X_b = X[self.block_idx]
            proj1_b = mu_lambda_beta * (current_piX @ X_b)
            resid2_b = Y_b - (M_S_star[self.block_idx] @ mu_W[self.block_idx])
            varX_b = (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * (X_b.T @ current_piX_sq @ X_b)
            term1_b = (varX_b - 2 * (resid2_b.T @ proj1_b))
            coeff_b = -lambda_a2 / (2 * lambda_b2)
            total_term1_b = coeff_b * term1_b

            # Second summation: over entries of pi_x
            x_mk_squared_b = current_piX.pow(2)
            x_mk_minus1_squared_b = (current_piX - 1).pow(2)

            exponent1_b = -x_mk_squared_b / (2 * eta_X_sq)
            exponent2_b = -x_mk_minus1_squared_b / (2 * eta_X_sq)

            # Stable log-sum-exp
            log_term_b = torch.logsumexp(torch.stack([exponent1_b, exponent2_b]), dim=0)
            total_log_term_b = log_term_b.sum()

            # Final terms
            neg_log_tauX_b = self.n_locations ** 2 * torch.log(torch.tensor(tau_X))

            # Update ELBO
            elbo += total_term1_b + total_log_term_b + neg_log_tauX_b + 0.5 * torch.log(torch.special.expit(self.VX) * (VX_ub - 0.01) + 0.01).sum()
        
        self.current_M_X_star = current_M_X_star / n_piX_sample
        self.current_V_X_star = current_V_X_star / n_piX_sample    
        elbo = elbo / n_piX_sample
        return -elbo


class vi_piS(nn.Module):
    def __init__(self, n_locations, block_idx):
        super(vi_piS, self).__init__()
        self.n_locations = n_locations
        self.block_idx = block_idx
        # Initialize block-specific parameters for piS
        self.MS = nn.Parameter(torch.log(1 / torch.tensor(n_locations)) * torch.ones(n_locations, n_locations, requires_grad=True))
        self.VS = nn.Parameter((-2) * torch.ones(n_locations, n_locations, requires_grad=True))
        self.current_M_S_star = (1 / torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_S_star = torch.eye(n_locations, n_locations)
        
    def forward(self, Y, X, mu_lambda_beta, M_X_star,
            lambda_a2, lambda_b2, mu_W, Sigma_W,
            eta_S_sq, tau_S=0.1, n_piS_sample=100, VS_ub=2, seed=100):

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize ELBO components for the specified block
        elbo = 0
        current_M_S_star = torch.zeros(self.n_locations, self.n_locations)
        current_V_S_star = torch.zeros(self.n_locations, self.n_locations)

        # Process the specified block
        log_MS = self.MS
        log_MS_tilde = sinkhorn_logspace(log_MS, niters=10)
        MS_tilde = torch.exp(log_MS_tilde)

        for i in range(n_piS_sample):
            torch.manual_seed(i * seed)
            z = torch.randn(self.n_locations, self.n_locations)
            Phi = MS_tilde + torch.sqrt(torch.special.expit(self.VS) * (VS_ub - 0.01) + 0.01) * z
            round_Phi = round_to_perm((Phi - 0.95 * Phi.min()).detach().numpy())
            current_piS = tau_S * Phi + (1 - tau_S) * torch.tensor(round_Phi, dtype=Phi.dtype)

            current_piS_sq = current_piS.T @ current_piS
            current_M_S_star += current_piS
            current_V_S_star += current_piS_sq

            coeff = -lambda_a2 / (2 * lambda_b2)
            resid1 = Y[self.block_idx] - mu_lambda_beta * (M_X_star[self.block_idx] @ X[self.block_idx])
            proj2 = current_piS @ mu_W[self.block_idx]
            mu_W_flat = mu_W[self.block_idx].flatten()
            V_W = Sigma_W[self.block_idx * self.n_locations:(self.block_idx + 1) * self.n_locations, 
                  self.block_idx * self.n_locations:(self.block_idx + 1) * self.n_locations] + \
              torch.outer(mu_W_flat, mu_W_flat)
            varS = torch.trace(current_piS_sq @ V_W)
            total_term1 = coeff * (varS - 2 * (resid1.T @ proj2))

            # Prior term over pi_S entries
            s_mk_squared = current_piS.pow(2)
            s_mk_minus1_squared = (current_piS - 1).pow(2)

            exponent1 = -s_mk_squared / (2 * eta_S_sq)
            exponent2 = -s_mk_minus1_squared / (2 * eta_S_sq)

            log_term = torch.logsumexp(torch.stack([exponent1, exponent2]), dim=0)
            total_log_term = log_term.sum()

            # Final terms
            neg_log_tauS = self.n_locations ** 2 * torch.log(torch.tensor(tau_S))

            # Update ELBO
            elbo += total_term1 + total_log_term + neg_log_tauS + 0.5 * torch.log(torch.special.expit(self.VS) * (VS_ub - 0.01) + 0.01).sum()

        self.current_M_S_star = current_M_S_star / n_piS_sample
        self.current_V_S_star = current_V_S_star / n_piS_sample
        
        elbo = elbo / n_piS_sample
        return -elbo
