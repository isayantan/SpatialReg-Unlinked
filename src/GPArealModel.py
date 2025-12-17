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
        self.logphi = nn.Parameter(torch.tensor(0.05, device=device))
        self.logtausq = nn.Parameter(torch.tensor(0.05, device=device))
        self.logsigmasq = nn.Parameter(torch.tensor(0.05, device=device))
        self.beta = nn.Parameter(torch.zeros(input_dim, device=device))
    
    def GP_log_likelihood(self, y, x, beta, K):
        K_NN = K  # Use precomputed covariance
        K_NN_noise = K_NN
        log_det = torch.logdet(K_NN_noise)
        
        residual = y - x @ beta
        quadratic_term = residual.T @ torch.linalg.solve(K_NN_noise, residual)
        log_likelihood = -0.5 * (log_det + quadratic_term)
        return log_likelihood   
        

    def forward(self, s, region_assignments, x, y):
        # Compute covariance matrix using Matern kernel
        unique_regions = torch.unique(region_assignments)
        B = len(unique_regions)
    
        # Compute full covariance matrix for all locations using Matern kernel
        #scaled_s = s / length_scale
        dists = torch.cdist(s, s, p=2)  # Compute pairwise Euclidean distances
        dists = (dists + dists.T) / 2  # Ensure symmetry
        sigmasq = torch.exp(self.logsigmasq)
        phi = torch.exp(self.logphi)
        K = sigmasq * torch.exp(- ((1/phi)* dists))
        tausq = torch.exp(self.logtausq)
        K += tausq * torch.eye(s.shape[0], device=s.device)  # Add noise term
    
        # Initialize variance-covariance matrix
        cov_matrix = torch.zeros((B, B))
    
        # Create a binary indicator matrix of shape (num_samples, num_regions)
        region_mask = torch.stack([(region_assignments == region).float() for region in unique_regions], dim=1)

        # Compute region sizes (n_i for each region)
        region_sizes = region_mask.sum(dim=0, keepdim=True)  # Shape: (1, num_regions)

        # Compute the covariance matrix efficiently
        # Equivalent to looping over i, j but using matrix multiplication
        region_sums = region_mask.T @ K @ region_mask  # Shape: (num_regions, num_regions)
        cov_matrix = region_sums / (region_sizes.T @ region_sizes)  # Element-wise division


        return -self.GP_log_likelihood(y, x, self.beta, cov_matrix)