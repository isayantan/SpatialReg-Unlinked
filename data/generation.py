import numpy as np
from scipy.special import gamma, kv

def matern_kernel(X, Y, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """
    Compute the Matern kernel between points X and Y with smoothness parameter nu.
    
    Args:
    X : numpy.ndarray
        Locations (n_samples, n_features).
    Y : numpy.ndarray
        Locations (n_samples, n_features).
    length_scale : float, optional
        Length scale parameter, default is 1.0.
    sigma_f : float, optional
        Variance of the kernel, default is 1.0.
    nu : float, optional
        Smoothness parameter (nu > 0), default is 1.5 (for nu=3/2).
        
    Returns:
    numpy.ndarray
        The kernel matrix (n_samples_X, n_samples_Y).
    """
    # Compute pairwise distances
    dists = np.linalg.norm(X[:, np.newaxis] - Y, axis=2)
    
    # Compute the scaling factor for the kernel
    scale = np.sqrt(2 * nu) * dists / length_scale
    
    # Matern kernel formula using the Bessel function (modified)
    # Compute the constant factor: (2^(1-nu)) / Gamma(nu)
    const_factor = (2 ** (1 - nu)) / gamma(nu)
    
    # Compute the kernel matrix with the Bessel function
    kernel_matrix = sigma_f**2 * const_factor * (scale ** nu) * kv(nu, scale)
    
    return kernel_matrix
    
# Data generation process
def generate_data(B, n_i, length_scale=1.0, sigma_f=1.0, nu=1.5, beta_true=2.0, tau_true=1.0):
    '''
    B: Number of regions
    n_i: Number of locations per region
    length_scale: Length scale for the Matern kernel
    sigma_f: Signal variance for the Matern kernel
    beta_true: True value of the regression coefficient
    tau_true: True value of the nugget variance
    
    Returns:
    y: Observed values (n_i x 1) for each region i
    x: Covariates (n_i x 1) for each region i
    w: Spatial residuals (n_i x 1) for each region i
    e: Independent errors (n_i x 1) for each region i
    s: Spatial locations (n_i x 2) for each region i
    
    Note: The spatial residuals w(s_ij) are generated using the Matern kernel.
    '''
    y = []
    x = []
    w = []
    e = []
    s = []
    
    for i in range(B):
        # Generate random spatial locations for area i
        s_i = np.random.rand(n_i, 2)  # 2D coordinates for simplicity
        s.append(s_i)
        
        # Generate covariates x(s_ij)
        x_i = np.random.rand(n_i, 1)  # Random covariates
        x.append(x_i)
        
        # Generate spatially correlated residuals w(s_ij) using Matern kernel
        K = matern_kernel(s_i, s_i, length_scale, sigma_f, nu) + np.eye(n_i) * 1
        w_i = np.random.multivariate_normal(np.zeros(n_i), K)  # Generate residuals
        w.append(w_i)
        
        # Generate independent errors e_ij
        e_i = np.random.normal(0, tau_true, n_i)
        e.append(e_i)
        
        # Generate observed y(s_ij)
        y_i = x_i * beta_true + w_i + e_i.reshape(-1, 1)
        y.append(y_i)
    
    return np.array(y), np.array(x), np.array(w), np.array(e), np.array(s)

# Example: Generate synthetic data for 3 regions, each with 10 locations
B = 3  # Number of regions
n_i = 100  # Number of locations per region

# Generate data
y, x, w, e, s = generate_data(B, n_i)

# Output some of the generated data
print("Generated y (observed values):", y)
print("Generated x (covariates):", x)
print("Generated w (spatial residuals):", w)
print("Generated e (errors):", e)
