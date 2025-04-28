# Import functions from preprocessing.py
import sys
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from GPModel import GPModel
from GPArealModel import GPArealModel
from VIGP_Unlinked import VIGP_Unlinked


# Add the path to the src directory
sys.path.append(os.path.abspath(os.path.join('..', 'data')))

# Get input from command line arguments
if len(sys.argv) < 4:
    raise ValueError("Please provide values for B, n_i, and seed as command line arguments.")

B = int(sys.argv[1])
n_i = int(sys.argv[2])
seed = int(sys.argv[3])

result = {}
input_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

niter_GP=3000
niter_GPAreal=3000
niter_VI= 100

# Load data from the specified path
data_path = os.path.join('..', 'data', f'B_{B}_n_{n_i}', f'data_seed_{seed}.pt')
data = torch.load(data_path)

# Extract variables from the data dictionary
y = data['y']
region_assignments = data['region_assignments']
x = data['x']
w = data['w']
e = data['e']
s = data['s']
x_jumbled_within_regions = data['x_jumbled_within_regions']
s_jumbled_within_regions = data['s_jumbled_within_regions']
perm_matrix_x = data['perm_matrix_x']
perm_matrix_s = data['perm_matrix_s']
sigmasq_true = data['sigmasq_true']
phi_true = data['phi_true']
beta_true = data['beta_true']
nu_true = data['nu_true']
tausq_true = data['tausq_true']

# Train the GPmodel (oracle)
# Oracle GP model if the locations and links are known
# Initialize and optimize the model
model = GPModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

for i in tqdm(range(niter_GP)):
    optimizer.zero_grad()
    loss = model(s, x, y)
    loss.backward()
    optimizer.step()
    
    # Constrain sigmasq, length_scale, and tausq to be positive
    with torch.no_grad():
        model.sigmasq.clamp_(min=1e-6)
        model.length_scale.clamp_(min=1e-6)
        model.tausq.clamp_(min=1e-6)
        
# Save the model parameters
model_params = {
    'nu': model.nu.item(),
    'phi': 1/model.length_scale.item(),
    'sigmasq': model.sigmasq.item(),
    'tausq': model.tausq.item(),
    'beta': model.beta.detach().cpu().numpy()
}

# Save the model parameters to a file
result['GPmodel'] = model_params



# Train the model GPareal
# Compute region-wise averages directly
unique_regions = torch.unique(region_assignments)
B = len(unique_regions)

ybar = torch.zeros(B, device=y.device)
xbar = torch.zeros(B, input_dim, device=x.device)

for i, region in enumerate(unique_regions):
    # Get indices for the current region
    indices = torch.where(region_assignments == region)[0]
    
    # Compute region-wise averages for y and x
    ybar[i] = torch.mean(y[indices])
    xbar[i] = torch.mean(x_jumbled_within_regions[indices], dim=0)


# Initialize and optimize the model
model = GPArealModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

for i in tqdm(range(niter_GPAreal)):
    optimizer.zero_grad()
    loss = model(s_jumbled_within_regions, region_assignments, xbar, ybar)
    loss.backward()
    optimizer.step()


# Save the model parameters
model_params = {
    'nu': model.nu.item(),
    'phi': 1/model.length_scale.item(),
    'sigmasq': model.sigmasq.item(),
    'tausq': model.tausq.item(),
    'beta': model.beta.detach().cpu().numpy()
}
# Save the model parameters to a file
result['GPArealModel'] = model_params


# Train the model VIGP_unlinked
n_blocks = B
n_locations = n_i

# Random toy data for X and Y
X = torch.tensor(x_jumbled_within_regions, dtype=torch.float32).reshape(n_blocks, n_locations)
Y = torch.tensor(y, dtype=torch.float32).reshape(n_blocks, n_locations)

# Generate (n_blocks * n_locations) 2D coordinates
total_points = n_blocks * n_locations
locations = torch.tensor(s_jumbled_within_regions,dtype=torch.float32)

# Compute distance matrix from locations
Dist = torch.cdist(locations, locations, p=2)  # Pairwise distances
Dist = (Dist + Dist.T) / 2  # Make it symmetric manually to avoid numerical asymmetry

# Set optional args
n_steps = 50
n_phi_samples = 100
n_piX_sample = 50
n_piS_sample = 50

for tau in [0.05, 0.1, 0.2, 0.3,0.4, 0.5, 0.55]:
    tau_X = tau
    tau_S = tau

    results_VI = VIGP_Unlinked(
        n_iter=niter_VI,
        n_blocks=n_blocks,
        n_locations=n_locations,
        phi_prior_ub=10,
        X=X,
        Y=Y,
        Dist=Dist,
        n_steps=n_steps,
        n_phi_samples=n_phi_samples,
        n_piX_sample=n_piX_sample,
        tau_X=tau_X, tau_S=tau_S,
        n_piS_sample=n_piS_sample,
        seed=521, 
        fix_piX= False, 
        fix_piS= False,
        fix_mu_lambda_beta=False,
        fix_sigmasq_lambda_beta=False,
        fix_lambda_b1=False,
        lambda_b1_fixed=((B*n_i)*0.5 + 0.1) * 5,
        fix_lambda_b2=False,
        M_X_star_fixed=perm_matrix_x.T,
        M_S_star_fixed=perm_matrix_s.T,
        V_X_star_fixed=torch.eye(n_locations, n_locations, device=device),
        V_S_star_fixed=torch.eye(n_locations, n_locations, device=device), 
        phi_init = 0.5,
        mean_Rphi_inv_fixed= torch.linalg.inv(torch.exp(-4 * Dist)),
        fix_mean_Rphi_inv=False, 
        pi_X_true = perm_matrix_x.T,
        pi_S_true = perm_matrix_s.T,
        VX_ub = 0.5,
        VS_ub=0.5,
        lr_piS=0.01,
        lr_piX=0.01
    )
   
    # Save the model parameters to the result dictionary
    result[f'VIGP_unlinked_tau_{tau}'] = results_VI
   
# Save the result dictionary to a file
result_path = os.path.join('..', 'data', 'results' , f'B_{B}_n_{n_i}', f'results_seed_{seed}.pt')
os.makedirs(os.path.dirname(result_path), exist_ok=True)
torch.save(result, result_path)