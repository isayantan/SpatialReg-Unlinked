# main.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from generation import generate_data  # Import the generate_data function from generation.py
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Assume the generate_data and other necessary methods like matern_kernel, nngp_kernel are defined already.

# Step 1: Order the locations initially (e.g., based on centroid)
def order_locations(s):
    """
    Orders the locations based on some criterion, e.g., by spatial coordinates.
    This function returns the ordered locations and the corresponding indices.
    """
    # Assuming s is a matrix where each row corresponds to a location's spatial coordinate.
    order = np.argsort(s[:, 0])  # Sorting based on x-coordinate (you can change this to y or any other rule)
    return s[order], order

# Step 2: Generate residuals w_hat (y - x*beta)
def compute_residuals(x, y, beta_hat):
    """
    Compute the residuals w_hat as the difference between observed and predicted y values.
    """
    y_pred = np.dot(x, beta_hat)  # Predicted y values
    w_hat = y - y_pred  # Residuals
    return w_hat

# Step 3: Generate pseudo-spatial errors w_tilde using GP with kernel (K_theta + tau^2)
def generate_gp_residuals(s_ordered, tau_hat, theta_hat, matern_kernel):
    """
    Generate pseudo-residuals w_tilde using a Gaussian Process with the given kernel and parameters.
    """
    K_theta = matern_kernel(s_ordered, s_ordered, length_scale=theta_hat[0], sigma_f=theta_hat[1], nu=theta_hat[2])
    noise_matrix = np.eye(len(s_ordered)) * tau_hat**2  # Adding tau^2 to the diagonal
    K_total = K_theta + noise_matrix  # Covariance matrix
    w_tilde = np.random.multivariate_normal(mean=np.zeros(len(s_ordered)), cov=K_total)
    return w_tilde

# Step 4: Hungarian algorithm to match w_hat to w_tilde
def hungarian_matching(w_hat, w_tilde):
    """
    Use the Hungarian algorithm to match w_hat to w_tilde.
    Returns the matched indices.
    """
    cost_matrix = cdist(w_hat.reshape(-1, 1), w_tilde.reshape(-1, 1), metric='euclidean')  # Cost matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Solve assignment problem
    return row_ind, col_ind


# Matern kernel with Bessel function
def matern_kernel(X, Y, length_scale=1.0, sigma_f=1.0, nu=1.5):
    dists = np.linalg.norm(X[:, np.newaxis] - Y, axis=2)
    scale = np.sqrt(2 * nu) * dists / length_scale
    const_factor = (2 ** (1 - nu)) / gamma(nu)
    kernel_matrix = sigma_f**2 * const_factor * (scale ** nu) * kv(nu, scale)
    return kernel_matrix

# Nearest Neighbor Gaussian Process kernel approximation
def nngp_kernel(X, Y, nu=1.5, length_scale=1.0, sigma_f=1.0, num_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(Y)
    kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dist = distances[j]
            kernel_matrix[i, j] = np.sum(np.exp(-dist / length_scale))
    return kernel_matrix

# Likelihood function to estimate parameters
def log_likelihood(params, X, Y, kernel_matrix):
    beta = params[0]
    tau2 = params[1]
    theta = params[2:]
    cov_matrix = kernel_matrix * tau2 + np.eye(len(X)) * tau2
    residuals = Y - X @ beta
    log_det = np.linalg.slogdet(cov_matrix)[1]
    inv_cov = np.linalg.inv(cov_matrix)
    likelihood = -0.5 * (residuals.T @ inv_cov @ residuals + log_det + len(X) * np.log(2 * np.pi))
    return -likelihood

# Generate data using the generation.py script
B = 9  # Number of regions
n = 100  # Number of locations per region
y, x, w, e, s, region_assignments = generate_data(B, n)

# Calculate regionwise averages and centroids
y_average = np.mean(y, axis=1)
x_average = np.mean(x, axis=1)
s_average = np.mean(x, axis=1)  # Centroid of locations

# Compute NNGP kernel matrix using the Matern kernel
K_nngp = nngp_kernel(s_average, s_average, nu=1.5, length_scale=1.0, sigma_f=1.0, num_neighbors=5)

# Optimize parameters using the log-likelihood function
initial_params = np.ones(3)  # Initial guess for [beta, tau2, theta]
result = minimize(log_likelihood, initial_params, args=(x_average, y_average, K_nngp))

# Extract the estimated parameters
beta_hat = result.x[0]
tau2_hat = result.x[1]
theta_hat = result.x[2:]

# Output estimated parameters
print(f"Estimated beta: {beta_hat}")
print(f"Estimated tau^2: {tau2_hat}")
print(f"Estimated theta: {theta_hat}")


# Iterate over the regions and perform matching
for i in range(B):
    # Extract region data
    x_region = x[i]
    y_region = y[i]
    s_region = s[i]

    # Step 3: Use the matched indices to reorder y and x
    y_matched = y_region[col_ind]
    x_matched = x_region[col_ind]
    
    # Step 4: Compute residuals for the matched pairs
    w_hat = compute_residuals(x_matched, y_matched, beta_hat)
    
    
    # Step 1: Compute residuals w_hat (y - x*beta)
    w_hat = compute_residuals(x_region, y_region, beta_hat)
    
    # Step 2: Generate pseudo-spatial errors w_tilde using GP
    s_ordered_region, order_indices_region = order_locations(s_region)
    w_tilde = generate_gp_residuals(s_ordered_region, tau_hat, theta_hat, matern_kernel)
    
    # Step 3: Match residuals w_hat to pseudo-residuals w_tilde using Hungarian algorithm
    row_ind, col_ind = hungarian_matching(w_hat, w_tilde)
    
    # Store matched locations
    matched_locations = order_indices_region[col_ind]  # Matched location indices for this region
    print(f"Region {i+1}: Matched locations (true y to predicted y) - {matched_locations}")

