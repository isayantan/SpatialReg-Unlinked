import torch
import torch.nn as nn
from utils import exponential_kernel, nearest_pd_torch

class GPModel(nn.Module):
    def __init__(self, input_dim = 1, device='cpu'):
        super(GPModel, self).__init__()
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
        

    def forward(self, s, x, y):
        n = y.shape[0]
        # Compute covariance matrix using Matern kernel
        dists = torch.cdist(s, s, p=2)  # Compute pairwise Euclidean distances
        dists = (dists + dists.T) / 2  # Ensure symmetry
        sigmasq=torch.exp(self.logsigmasq)
        phi = torch.exp(self.logphi)
        tausq = torch.exp(self.logtausq)
        K = sigmasq * torch.exp(- ((1/phi)* dists)) + tausq * torch.eye(n, device=self.device)
        #K = self.sigmasq * exponential_kernel(s, s, phi=self.phi) + self.tausq * torch.eye(n, device=self.device)
        return -self.GP_log_likelihood(y, x, self.beta, K)