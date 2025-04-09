import torch
import torch.nn as nn
import math
from hungarian import hungarian
from torch import Tensor

def sinkhorn_logspace(logP, niters=10):
    for _ in range(niters):
        # Normalize columns and take the log again
        logP = logP - torch.logsumexp(logP, dim=0, keepdim=True)
        # Normalize rows and take the log again
        logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)
    return logP

class vi_piX(nn.Module):
    def __init__(self, n_locations):
        super(vi_piX, self).__init__()
        self.n_locations = n_locations
        self.MX = nn.Parameter(torch.ones(n_locations, n_locations, requires_grad=True))
        self.VX = nn.Parameter(0.2 * torch.ones(n_locations, n_locations, requires_grad=True))
            
    def forward(self, Y, X, mu_lambda_beta,
                sigmasq_lambda_beta, M_S_star, mu_W,
                eta_X, lambda_a2, lambda_b2, 
                tau_X = 0.1, n_sample = 100):
        
        # sample piX
        torch.manual_seed(42)  # Set seed for reproducibility
        sampled_piX = torch.zeros(n_sample, self.n_locations, self.n_locations)
        
        for i in range(n_sample):
            log_MX = torch.log_softmax(self.MX, dim=1)
            log_MX_tilde = sinkhorn_logspace(log_MX, niters=10)
            MX_tilde = torch.exp(log_MX_tilde)
            Phi = MX_tilde + self.VX * torch.randn(self.n_locations, self.n_locations)
            sampled_piX[i] = tau_X * Phi + (1 - tau_X) * hungarian(-Phi)  # Exponentiate to get piX
        
        # Compute the ELBO
        B = Y.shape[0]
        elbo = 0.0
        for i in range(n_sample):
            current_piX = sampled_piX[i]  # Access the current sample of piX

            # Vectorized computation for all regions
            temp = torch.matmul(current_piX, X.transpose(1, 2))  # Shape: (B, n_features)
            part1 = 2 * mu_lambda_beta * torch.sum(Y * temp, dim=1)  # Shape: (B,)
            part2 = (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * torch.sum(temp ** 2, dim=1)  # Shape: (B,)
            part3 = 2 * mu_lambda_beta * torch.sum(temp * (M_S_star @ mu_W), dim=1)  # Shape: (B,)

            term1 = part1 + part2 + part3  # Shape: (B,)
            total_term1 = -lambda_a2 / (2 * lambda_b2) * term1.sum()  # Scalar

            # Vectorized computation for the second summation
            exponent1 = -current_piX ** 2 / (2 * eta_X ** 2)
            exponent2 = -(current_piX - 1) ** 2 / (2 * eta_X ** 2)
            log_term = torch.log(torch.exp(exponent1) + torch.exp(exponent2))  # Shape: (n_locations, n_locations)

            const = -0.5 * math.log(2 * math.pi * eta_X ** 2)
            total_log_term = (const + log_term).sum()  # Scalar

            # Compute the negative log term for tau_X
            neg_log_tauX = -self.n_locations ** 2 * torch.log(torch.tensor(tau_X))

            # Update ELBO
            elbo += total_term1 + total_log_term + neg_log_tauX - self.VX.sum()
        elbo = elbo / n_sample 
        
        # B, d = X.shape

        # # First summation term
        # term1 = 0.
        # for i in range(B):
        #     Y_i = Y[i]                    # (d,)
        #     X_i = X[i]                    # (d,)
        #     mu_Wi = mu_W[i]              # (d,)

        #     # (1) 2 * Y_i^T * pi_x * X_i * mu
        #     part1 = 2 * mu_lambda_beta * torch.dot(Y_i, pi_x @ X_i)

        #     # (2) (mu^2 + sigma^2) * X_i^T * pi_x^T * pi_x * X_i
        #     temp = pi_x @ X_i            # (d,)
        #     part2 = (mu_lambda_beta ** 2 + sigma2_lambda_beta) * temp.dot(temp)

        #     # (3) 2 * mu * X_i^T * pi_x^T * M_star_S * mu_Wi
        #     part3 = 2 * mu_lambda_beta * (X_i @ pi_x.T @ M_star_S @ mu_Wi)

        #     term1 += part1 + part2 + part3

        # coeff = -lambda_a2 / (2 * lambda_b2)
        # total_term1 = coeff * term1

        # # Second summation: over entries of pi_x
        # x_mk = pi_x  # (d, d)
        # x_mk_squared = x_mk ** 2
        # x_mk_minus1_squared = (x_mk - 1) ** 2

        # # log[exp(-x^2/2η^2) + exp(-(x-1)^2/2η^2)]
        # exponent1 = -x_mk_squared / (2 * eta_x**2)
        # exponent2 = -x_mk_minus1_squared / (2 * eta_x**2)
        # log_term = torch.log(torch.exp(exponent1) + torch.exp(exponent2))

        # const = -0.5 * math.log(2 * math.pi * eta_x ** 2)
        # total_log_term = (const + log_term).sum()  # scalar

        # # Final terms
        # n = d
        # neg_log_tauX = -n ** 2 * torch.log(torch.tensor(tau_X))

        # final_obj = total_term1 + total_log_term + neg_log_tauX - V_X_entropy
        
        
        return -elbo
    
    

