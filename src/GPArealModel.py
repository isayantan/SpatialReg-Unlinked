import torch
import torch.nn as nn
from utils import compute_regionwise_covariance


# PyTorch Module for optimizing block-level covariance
class GPArealModel(nn.Module):
    def __init__(self, input_dim = 1, device='cpu'):
        super(GPArealModel, self).__init__()
        self.device = device

        # Learnable parameters
        self.nu = nn.Parameter(torch.tensor(0.5, device=device))
        self.phi = nn.Parameter(torch.tensor(1.0, device=device))
        self.tausq = nn.Parameter(torch.tensor(1.0, device=device))
        self.sigmasq = nn.Parameter(torch.tensor(2.0, device=device))
        self.beta = nn.Parameter(torch.zeros(input_dim, device=device))
    
    def GP_log_likelihood(self, y, s, x, beta, K):
        n = y.shape[0]
        K_NN = K  # Use precomputed covariance
        K_NN_noise = K_NN
        log_det = torch.logdet(K_NN_noise)
        
        residual = y - x @ beta
        quadratic_term = residual.T @ torch.linalg.solve(K_NN_noise, residual)
        log_likelihood = -0.5 * (log_det + quadratic_term)
        return log_likelihood   
        

    def forward(self, s, region_assignments, x, y):
        # Compute covariance matrix using Matern kernel
        K = compute_regionwise_covariance(s, region_assignments, self.sigmasq, self.phi, self.nu, self.tausq)
        return -self.GP_log_likelihood(y, s, x, self.beta, K)