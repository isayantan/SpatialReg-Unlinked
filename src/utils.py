import torch

def exponential_kernel(X1, X2, length_scale=1.0, sigma=1.0):
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
    return sigma * torch.exp(-dists / length_scale)


def compute_regionwise_covariance(s, region_assignments, sigmasq, length_scale, nu=0.5, tausq=1):
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
    matern_kernel = exponential_kernel(s, s, length_scale=length_scale)
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