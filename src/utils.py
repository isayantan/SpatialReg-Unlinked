import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def sinkhorn_logspace(logP, niters=10):
    for _ in range(niters):
        # Normalize columns and take the log again
        logP = logP - torch.logsumexp(logP, dim=0, keepdim=True)
        # Normalize rows and take the log again
        logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)
    return logP

def exponential_kernel(X1, X2, phi=1.0, sigma=1.0):
    """
    Computes the Matérn covariance function with ν = 1/2 (exponential covariance).

    Args:
        X1 (torch.Tensor): Input tensor of shape (n, d).
        X2 (torch.Tensor): Input tensor of shape (m, d).
        rho (float): Characteristic lengthscale parameter.
        sigma (float): Scaling factor (variance term).

    Returns:
        torch.Tensor: Covariance matrix of shape (n, m).
    """
    dists = torch.cdist(X1, X2, p=2)  # Compute pairwise Euclidean distances
    dists = (dists + dists.T) / 2  # Ensure symmetry
    return sigma * torch.exp(- (phi * dists))


def compute_regionwise_covariance(s, region_assignments, sigmasq, phi, nu=0.5, tausq=1):
    """
    Computes the variance-covariance matrix for the averaged spatial GP process w_ibar.

    Parameters:
    s: (N, 2) tensor of spatial locations
    region_assignments: (N,) tensor assigning each location to a region
    sigma_sq: scalar, variance of the GP
    length_scale: scalar, length scale of the Matern kernel
    nu: scalar, smoothness parameter of the Matern kernel
    tausq: scalar, variance of the noise term

    Returns:
    cov_matrix: (B, B) tensor, variance-covariance matrix of region-averaged w
    """
    unique_regions = torch.unique(region_assignments)
    B = len(unique_regions)
    
    # Compute full covariance matrix for all locations using Matern kernel
    #scaled_s = s / length_scale
    matern_kernel = exponential_kernel(s, s, phi = phi)
    K = sigmasq * matern_kernel  # Scale by variance
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

    # # Ensure numerical stability by avoiding division by zero
    # cov_matrix[region_sizes.T @ region_sizes == 0] = 0

    # # Add diagonal tausq / n_i term
    # diag_tau_term = torch.diag(tausq / region_sizes)* torch.eye(B)  # Shape: (num_regions, num_regions)
    # cov_matrix += diag_tau_term
        
    return cov_matrix


def round_to_perm(P):
    N = P.shape[0]
    assert P.shape == (N, N)
    row, col = linear_sum_assignment(-P)
    P = np.zeros((N, N))
    P[row, col] = 1.0
    return P

def nearest_pd_torch(A, epsilon=1e-6):
    """
    Find the nearest positive definite matrix to A using eigenvalue clipping.

    Args:
        A (torch.Tensor): A symmetric matrix (n x n).
        epsilon (float): Minimum eigenvalue threshold to ensure positive definiteness.

    Returns:
        torch.Tensor: A positive definite matrix close to A.
    """
    # Ensure symmetry
    A_sym = (A + A.T) / 2

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(A_sym)

    # Clip eigenvalues to be at least epsilon
    eigvals_clipped = torch.clamp(eigvals, min=epsilon)

    # Reconstruct matrix: V Λ V^T
    A_pd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

    # Ensure symmetry again (numerical stability)
    A_pd = (A_pd + A_pd.T) / 2

    return A_pd

def compute_q_phi(phi, Dist, mu_W, Sigma_W, lambda_a1, lambda_b1, eps = 1e-6):
    """
    Compute q(phi) as defined by the given expression.

    Args:
    - phi (scalar): The tensor for which q(phi) needs to be calculated (B x d).
    - Dist (torch.Tensor): The distance matrix (nB X nB).
    - Sigma_W (torch.Tensor): The matrix Sigma_W (nB x nB).
    - mu_W (torch.Tensor): The vector mu_W (d,).
    - lambda_a1 (float): The scalar value for lambda_a1.
    - lambda_b1 (float): The scalar value for lambda_b1.
    
    Returns:
    - q_phi (torch.Tensor): The calculated q(phi).
    """
    
    # Compute R(phi)
    R_phi = torch.exp(-phi * Dist)

    # Optionally: Ensure R_phi is positive-definite
    #R_phi = nearest_pd_torch(R_phi, epsilon=eps)

    # Compute log(det(R_phi)) safely
    sign, logdet = torch.linalg.slogdet(R_phi)
    if sign <= 0:
        # Handle log of non-positive determinant safely
        logdet = torch.tensor(float('-inf'), device=R_phi.device)
        R_phi += eps * torch.eye(R_phi.shape[0], device=R_phi.device)  # Regularization

    # Solve R_phi x = mean
    mu_W_flat = mu_W.flatten()
    V_W = Sigma_W + torch.outer(mu_W_flat, mu_W_flat)

    
    R_phi_inv_mean = torch.linalg.solve(R_phi, V_W)
    #R_phi_inv_mean = torch.linalg.pinv(R_phi) @ V_W


    # Compute exponent
    exponent = -0.5 * logdet - (lambda_a1 / (2 * lambda_b1)) * (torch.trace(R_phi_inv_mean))

    # Return log of q_phi (numerically stable)
    return exponent  # this is log(q_phi), better for log-domain work
 
    