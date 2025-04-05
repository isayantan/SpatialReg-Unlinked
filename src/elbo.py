import torch
import torch.nn as nn
import math

class vi_piX(nn.Module):
    def __init__(self, n_locations):
        super(vi_piX, self).__init__()
        self.n_locations = n_locations
        self.MX = nn.Parameter(torch.ones(n_locations, n_locations, requires_grad=True))
        self.VX = nn.Parameter(0.2 * torch.ones(n_locations, n_locations, requires_grad=True))
    
    def generate_Z(self, n_sample):
        return torch.randn(n_sample, self.n_locations, self.n_locations)
        
    def forward(self, Y, X, mu_lambda_beta,
                sigmasq_lambda_beta, M_S_star, mu_W,
                eta_X, lambda_a2, lambda_b2, 
                tau_X = 0.1, n_sample = 100):
        
        # sample piX
        torch.manual_seed(42)  # Set seed for reproducibility
        
        Z = torch.randn(n_sample, self.n_locations, self.n_locations)
        # MX_tilde =
        # Phi = MX_tilde + self.VX * Z
        
        sampled_piX = torch.tensor(n_sample, self.n_locations, self.n_locations) 
        
        B = Y.shape[0]
        # for i in range(n_sample):
        #     pi_X = Pi_X[i]
        #     for i in range(B):
                
        #         # (1) 2 * Y_i^T * pi_x * X_i * mu
        #         temp = pi_X @ X[i] 
        #         part1 = 2 * mu_lambda_beta * torch.dot(Y[i], temp)
                
        #         # (2) (mu^2 + sigma^2) * X_i^T * pi_x^T * pi_x * X_i
        #         part2 = (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * temp.dot(temp)
                
        #         # (3) 2 * mu * X_i^T * pi_x^T * M_star_S * mu_Wi
        #         part3 = 2 * mu_lambda_beta * (temp.T @ M_S_star @ mu_W[i])
                
        #         term1 += part1 + part2 + part3
                
        #     coeff = -lambda_a2 / (2 * lambda_b2)
        #     total_term1 = coeff * term1
            
        #     # Second summation: over entries of pi_x

        #     # log[exp(-x^2/2η^2) + exp(-(x-1)^2/2η^2)]
        #     exponent1 = - pi_X**2 / (2 * eta_X**2)
        #     exponent2 = - (pi_X - 1)**2 / (2 * eta_X**2)
        #     log_term = torch.log(torch.exp(exponent1) + torch.exp(exponent2))

        #     const = -0.5 * math.log(2 * math.pi * eta_X ** 2)
        #     total_log_term = (const + log_term).sum()  # scalar
        #     neg_log_tauX = - self.n_locations ** 2 * torch.log(torch.tensor(tau_X))
            
        #     elbo += total_term1 + total_log_term + neg_log_tauX - self.VX.sum()
        
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
        return -elbo
    
    

