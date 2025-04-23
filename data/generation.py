import numpy as np
import random
from scipy.special import gamma, kv
from sklearn.gaussian_process.kernels import Matern as skMatern
import torch
import torch.special  # Contains Bessel function of the second kind (kv)



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
 
 

def generate_data(B, n_i, sigmasq, length_scale, nu, beta_true, tausq_true,seed=42, spatial=True):
    """
    Generates spatial data for a full region first, then assigns regions.

    Parameters:
    -----------
    B : int
        Number of regions.
    n_i : int
        Number of points per region.
    sigmasq : float
        Variance parameter for the Matérn kernel.
    length_scale : float
        Spatial correlation length parameter.
    nu : float
        Smoothness parameter for Matérn covariance.
    beta_true : float
        Coefficient for the covariate.
    tau_true : float
        Standard deviation of independent noise.

    Returns:
    --------
    y, x, w, e, s, region_assignments : np.ndarrays
        Outcome, covariates, spatial residuals, noise, locations, and region assignments.
    """
    np.random.seed(seed)  # Set seed for reproducibility
    
    N = B * n_i  # Total number of spatial locations
    
    # Generate spatial locations s
    s, region_assignments = partition_domain_into_regions(B, n_i)

    # Generate covariates x(s)
    x = np.random.rand(N, 1)  # Covariates for all locations

    # Compute spatially correlated residuals w(s) using Matérn kernel
    if spatial == True:
        matern_kernel = sigmasq * skMatern(nu=nu, length_scale=length_scale)
        K = matern_kernel(s)  # Full covariance matrix
        w = np.random.multivariate_normal(np.zeros(N), K)  # Residuals
    else:
        w = np.zeros(N)

    # Generate independent errors e(s)
    e = np.random.normal(0, np.sqrt(tausq_true), N)  # Independent noise

    # Generate outcomes y(s)
    y = beta_true * x.flatten() + w + e  # Outcome variable

    return y, x, w, e, s, region_assignments

# # # # Example: Generate synthetic data for 9 regions, each with 100 locations
# B = 9
# n_i = 20
# #  # Generate data
# y, x, w, e, s = generate_data(B, n_i)

# # # Output some of the generated data
# print("Generated y (observed values):", y)
# print("Generated x (covariates):", x)
# print("Generated w (spatial residuals):", w)
# print("Generated e (errors):", e)
# print("Generated s (spatial locations):", s)


# # Print the shape of s and the first few points
# print(f"Shape of s: {s.shape}")
# print(f"First few points of s: {s[:5]}")

# # Plot the grid and points
# grid_size = int(np.ceil(np.sqrt(B)))

# # Create the plot
# plt.figure(figsize=(8, 8))

# # Plot the grid
# x_divisions = np.linspace(0, 1, grid_size + 1)
# y_divisions = np.linspace(0, 1, grid_size + 1)
# for x in x_divisions:
#     plt.axvline(x=x, color='k', linestyle='--', alpha=0.5)
# for y in y_divisions:
#     plt.axhline(y=y, color='k', linestyle='--', alpha=0.5)

# # Plot the points
# plt.scatter(s[:, 0], s[:, 1], c=region_assignments, cmap='tab10', marker='o', edgecolor='k', s=30)

# # Title and labels
# plt.title("Partitioned Grid with Points")
# plt.xlabel("X-coordinate")
# plt.ylabel("Y-coordinate")
# plt.colorbar(label='Region ID')

# plt.show()
