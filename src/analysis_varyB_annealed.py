# analysis.py — tailored to your vary_B2 (no SNR) data layout
import sys
import os
import re
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np

from GPModel import GPModel
from GPArealModel import GPArealModel
from revised_VIGP_Unlinked import VIGP_Unlinked

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 1

# --------- CLI ---------
# Usage:
#   python analysis.py B n_i seed
#   python analysis.py B n_i seed phi
if len(sys.argv) < 4:
    raise ValueError("Usage: python analysis.py B n_i seed [phi]")

B_arg   = int(sys.argv[1])
n_i_arg = int(sys.argv[2])
seed    = int(sys.argv[3])

phi_cli = None
if len(sys.argv) >= 5:
    phi_cli = float(sys.argv[4])

# --------- Resolve data path ---------
base_dir = os.path.join('..', 'data', 'vary_B2', f'B_{B_arg}_n_{n_i_arg}')

def autodetect_phi(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    candidates = [
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and d.startswith('phi_')
    ]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No phi subfolders found in {folder}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple phi folders found in {folder}: {candidates}. "
            f"Re-run with explicit phi."
        )
    d = candidates[0]  # e.g., 'phi_2.0'
    m = re.match(r'^phi_([^/]+)$', d)
    if not m:
        raise ValueError(f"Cannot parse phi from folder name: {d}")
    return float(m.group(1)), d

if phi_cli is None:
    phi_resolved, phi_dir = autodetect_phi(base_dir)
else:
    phi_resolved = phi_cli
    phi_dir = f"phi_{phi_resolved}"

data_path = os.path.join(base_dir, phi_dir, f"data_seed_{seed}.pt")

# --------- Load data ---------
data = torch.load(data_path, map_location=device)

y  = data['y'].to(device).float()
x  = data['x'].to(device).float()
w  = data['w'].to(device).float()
e  = data['e'].to(device).float()
s  = data['s'].to(device).float()
region_assignments = data['region_assignments'].to(device).long()

x_jumbled_within_regions = data['x_jumbled_within_regions'].to(device).float()
s_jumbled_within_regions = data['s_jumbled_within_regions'].to(device).float()

perm_matrix_x = data['perm_matrix_x'].to(device).float()
perm_matrix_s = data['perm_matrix_s'].to(device).float()

sigmasq_true = float(data['sigmasq_true'])
phi_true     = float(data['phi_true'])
beta_true    = float(data['beta_true'])
nu_true      = float(data['nu_true'])
tausq_true   = float(data['tausq_true'])

# Sanity: unique region count
unique_regions = torch.unique(region_assignments)
B_in_data = len(unique_regions)
if B_in_data != B_arg:
    print(f"[WARN] B in data ({B_in_data}) != B from CLI ({B_arg}). Using B_in_data for shapes.")
n_blocks = B_in_data
n_locations = n_i_arg  # expected design; will assert below

N = y.numel()
if N % n_blocks != 0:
    raise ValueError(f"N={N} not divisible by B={n_blocks}")
if n_locations != (N // n_blocks):
    print(f"[WARN] n_i from CLI ({n_i_arg}) != inferred ({N // n_blocks}). Using inferred.")
    n_locations = N // n_blocks

# --------- Training iters ---------
niter_GP = 3000
niter_GPAreal = 3000
niter_VI = 100

result = {}
torch.manual_seed(521)

# ===================== 1) Oracle GP (locations & links known) =====================
gp = GPModel().to(device)
opt = optim.AdamW(gp.parameters(), lr=0.01, weight_decay=0.01)

for _ in tqdm(range(niter_GP), desc="Train GPModel (oracle)"):
    opt.zero_grad()
    loss = gp(s, x, y)
    loss.backward()
    opt.step()
    with torch.no_grad():
        gp.sigmasq.clamp_(min=1e-6)
        gp.phi.clamp_(min=1e-6)
        gp.tausq.clamp_(min=1e-6)

result['GPmodel'] = {
    'nu': float(gp.nu.item()),
    'phi': float(gp.phi.item()),
    'sigmasq': float(gp.sigmasq.item()),
    'tausq': float(gp.tausq.item()),
    'beta': gp.beta.detach().float().cpu().numpy(),
    'true_params': {
        'nu_true': nu_true, 'phi_true': phi_true, 'sigmasq_true': sigmasq_true,
        'tausq_true': tausq_true, 'beta_true': beta_true
    }
}

# ===================== 2) Areal GP (region-averaged) =====================
ybar = torch.zeros(n_blocks, device=device)
xbar = torch.zeros(n_blocks, input_dim, device=device)
for i, region in enumerate(unique_regions):
    idx = torch.where(region_assignments == region)[0]
    ybar[i] = torch.mean(y[idx])
    xbar[i] = torch.mean(x_jumbled_within_regions[idx], dim=0)

gpa = GPArealModel().to(device)
opt = optim.AdamW(gpa.parameters(), lr=0.01, weight_decay=0.01)
for _ in tqdm(range(niter_GPAreal), desc="Train GPArealModel"):
    opt.zero_grad()
    loss = gpa(s_jumbled_within_regions, region_assignments, xbar, ybar)
    loss.backward()
    opt.step()
    with torch.no_grad():
        gpa.sigmasq.clamp_(min=1e-6)
        gpa.phi.clamp_(min=1e-6)
        gpa.tausq.clamp_(min=1e-6)

result['GPArealModel'] = {
    'nu': float(gpa.nu.item()),
    'phi': float(gpa.phi.item()),
    'sigmasq': float(gpa.sigmasq.item()),
    'tausq': float(gpa.tausq.item()),
    'beta': gpa.beta.detach().float().cpu().numpy()
}

# ===================== 3) VI for Unlinked GP =====================
# Reshape to (B, n_i)
X = x_jumbled_within_regions.reshape(n_blocks, n_locations).contiguous()
Y = y.reshape(n_blocks, n_locations).contiguous()

locations = s_jumbled_within_regions  # (N, d)
Dist = torch.cdist(locations, locations, p=2)
Dist = (Dist + Dist.T) / 2

n_steps = 50
n_phi_samples = 120
n_piX_sample = 50
n_piS_sample = 50

prior_parameters = {
    "a1": 0.1, "b1": 0.1,
    "a2": 0.1, "b2": 0.1,
    "eta_X_sq": 0.1, "eta_S_sq": 0.1,
    "mu_beta": 0.0, "sigmasq_beta": 100.0,
    "phi_prior_lb": (1 / torch.max(Dist)),  # units: 1/distance
    "phi_prior_ub": 10.0
}

for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    results_VI = VIGP_Unlinked(
        n_iter=niter_VI,
        n_blocks=n_blocks,
        n_locations=n_locations,
        X=X, Y=Y,
        Dist=Dist,
        n_steps=n_steps,
        n_phi_samples=n_phi_samples,
        n_piX_sample=n_piX_sample,
        tau_X=tau, tau_S=tau,
        n_piS_sample=n_piS_sample,
        seed=521,
        fix_piX=False, fix_piS=False,
        fix_mu_lambda_beta=False,
        fix_sigmasq_lambda_beta=False,
        fix_lambda_b1=False,
        lambda_b1_fixed=((n_blocks * n_locations) * 0.5 + 0.1) * 5,
        fix_lambda_b2=False,
        M_X_star_fixed=perm_matrix_x.T,
        M_S_star_fixed=perm_matrix_s.T,
        V_X_star_fixed=torch.eye(n_locations, n_locations, device=device),
        V_S_star_fixed=torch.eye(n_locations, n_locations, device=device),
        phi_init=0.1,
        mean_Rphi_inv_fixed=torch.linalg.inv(torch.exp(-4 * Dist)),
        fix_mean_Rphi_inv=False,
        pi_X_true=perm_matrix_x.T,
        pi_S_true=perm_matrix_s.T,
        VX_ub=0.5, VS_ub=0.5,
        lr_piS=0.01, lr_piX=0.01,
        prior_parameters=prior_parameters,
        anneal_every=20, use_global_tau_anneal= True, elbo_W= 5
    )
    result[f'VIGP_unlinked_tau_{tau}'] = results_VI

# --------- Save results ---------
results_dir = os.path.join('..', 'data', 'results', 'vary_B2_annealed',
                           f'B_{B_arg}_n_{n_i_arg}', phi_dir)
os.makedirs(results_dir, exist_ok=True)
result_path = os.path.join(results_dir, f'results_seed_{seed}.pt')
torch.save(result, result_path)
print(f"[OK] Loaded data from {data_path}")
print(f"[OK] Saved results to {result_path}")