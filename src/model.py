# main.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from generation import generate_data  # Import the generate_data function from generation.py

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
B = 5  # Number of regions
n = 10  # Number of locations per region
y, x, s = generate_data(B, n)

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
