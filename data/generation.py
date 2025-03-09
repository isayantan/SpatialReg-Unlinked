import numpy as np
from scipy.special import gamma, kv

def get_region_points(s, region_assignments, region_id):
    """
    Extract the points from the spatial locations (s) that belong to the region with id `region_id`.
    
    Parameters:
    - s: A 2D array of spatial locations (shape: [total_points, 2]).
    - region_assignments: A 1D array of region assignments for each point (shape: [total_points,]).
    - region_id: The region ID for which points are to be extracted.
    
    Returns:
    - region_points: A 2D array of points corresponding to the specified region_id.
    """
    # Get boolean mask for points that belong to the region `region_id`
    region_mask = (region_assignments == region_id)
    
    # Use the mask to extract the points for the specific region
    region_points = s[region_mask]
    
    return region_points

def generate_ordered_uniform_random_locations(n_i, x_min, x_max, y_min, y_max):
    """
    Generate n_i ordered uniform random locations within the bounding box defined by
    (x_min, x_max) and (y_min, y_max).
    
    Parameters:
    - n_i: Number of points to generate.
    - x_min, x_max: The bounding x-axis values.
    - y_min, y_max: The bounding y-axis values.

    Returns:
    - ordered_locations: A 2D array of n_i ordered points within the region.
    """
    # Generate n_i uniform random points in the given bounds
    x_coords = np.random.uniform(x_min, x_max, n_i)
    y_coords = np.random.uniform(y_min, y_max, n_i)
    
    # Order the points, for example, by sorting first on x, then on y
    ordered_indices = np.argsort(x_coords)  # Sort by x-coordinate
    ordered_x = x_coords[ordered_indices]
    ordered_y = y_coords[ordered_indices]
    
    # Combine x and y into ordered locations
    ordered_locations = np.vstack([ordered_x, ordered_y]).T
    
    return ordered_locations

def partition_domain_into_regions(B, n_i):
    """
    Partition the unit square (0,1) x (0,1) into B regions, then generate n_i uniformly distributed
    points for each region and assign each point to the corresponding region, while ensuring points are ordered.

    Parameters:
    - B: Number of regions to divide the unit square into.
    - n_i: Number of points to generate per region.

    Returns:
    - s: A (n_i * B, 2) array of spatial locations uniformly sampled in the unit square, partitioned into regions.
    - region_assignments: A list of region assignments for each point, where each point is assigned to a region.
    """
    # Calculate the number of divisions along the x and y axis
    grid_size = int(np.ceil(np.sqrt(B)))  # Grid dimensions, assuming roughly a square grid
    
    # Generate the grid lines for dividing the unit square
    x_divisions = np.linspace(0, 1, grid_size + 1)  # Divisions along x-axis
    y_divisions = np.linspace(0, 1, grid_size + 1)  # Divisions along y-axis

    # Initialize the list to store region assignments
    s = []
    region_assignments = []
    
    # For each region, generate n_i random points and order them
    region_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if region_id >= B:
                break  # Stop if we've created B regions
            
            # Generate ordered n_i points in this region
            x_min, x_max = x_divisions[i], x_divisions[i + 1]
            y_min, y_max = y_divisions[j], y_divisions[j + 1]
            
            # Generate and order the points within the current region
            ordered_region_points = generate_ordered_uniform_random_locations(n_i, x_min, x_max, y_min, y_max)
            s.append(ordered_region_points)
            
            # Assign all points in this region to the current region id
            region_assignments.extend([region_id] * n_i)
            
            region_id += 1
    
    # Flatten the list of regions and their points into a single array
    s = np.vstack(s)  # All points in the domain
    region_assignments = np.array(region_assignments)  # Region assignments for each point
    
    return s, region_assignments


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

    s, region_assignments = partition_domain_into_regions(B, n_i)

    
    
    for i in range(B):
        # Generate random spatial locations for area i
        s_i= get_region_points(s, region_assignments, i)
        #s_i = np.random.rand(n_i, 2)  # 2D coordinates for simplicity
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
    # return np.array(s)

# # Example: Generate synthetic data for 9 regions, each with 100 locations
B = 9
n_i = 100
 # Generate data
y, x, w, e, s = generate_data(B, n_i)

# Output some of the generated data
print("Generated y (observed values):", y)
print("Generated x (covariates):", x)
print("Generated w (spatial residuals):", w)
print("Generated e (errors):", e)
