"""
helper_functions.py
===================
High-level wrappers for REPAIR (REgression with Permutation Alignment
via variational InfeRence).

This module exposes a clean, paper-ready API on top of the core
implementation files (revised_VIGP_Unlinked.py, GPModel.py,
GPArealModel.py, utils.py).  Import it from any script or notebook
that lives in, or adds, the ``src/`` directory to ``sys.path``.

Functions
---------
generate_synthetic_data   Generate synthetic unlinked spatial-regression data.
train_oracle_gp           Fit the FullGP (known links) oracle baseline.
train_areal_gp            Fit the ArealGP (block-averaged) baseline.
run_vigp_unlinked         Run the main REPAIR variational method.
permutation_accuracy      Evaluate how well a permutation was recovered.
summarise_results         Build a comparison table (True / FullGP / ArealGP / REPAIR).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data generation
# ─────────────────────────────────────────────────────────────────────────────

def _best_grid(B: int) -> tuple[int, int]:
    """
    Return (n_rows, n_cols) such that n_rows * n_cols == B and the layout
    is as close to square as possible (n_rows <= n_cols, n_rows maximised).

    Works for any positive integer B, including non-square values like 6, 10,
    12, 15, 20, etc.  For a perfect square B = k^2 the result is (k, k).
    """
    best_r, best_c = 1, B
    for r in range(1, int(B**0.5) + 1):
        if B % r == 0:
            best_r, best_c = r, B // r   # always r <= B//r
    return best_r, best_c


def generate_synthetic_data(
    B: int = 9,
    n_i: int = 5,
    beta_true: float = 1.5,
    phi_true: float = 2.0,
    sigmasq_true: float = 1.0,
    tausq_true: float = 0.2,
    seed: int = 42,
    n_rows: int | None = None,
    n_cols: int | None = None,
) -> dict:
    """
    Generate synthetic data for the unlinked spatial-regression problem.

    The domain [0, 1]^2 is divided into an ``n_rows`` × ``n_cols`` rectangular
    grid of blocks (B = n_rows * n_cols).  Each block receives ``n_i`` uniformly
    random locations.  The *same* unknown permutation pi_X shuffles covariates
    within every block, and a second permutation pi_S shuffles the spatial
    locations within every block, creating the doubly-unlinked structure that
    REPAIR aims to recover.

    Any positive integer B is supported — B does not have to be a perfect
    square.  If ``n_rows`` / ``n_cols`` are not supplied, the function chooses
    the most square-like factorisation automatically (e.g. B=6 → 2×3,
    B=12 → 3×4, B=9 → 3×3).

    Model
    -----
    Y_i = pi_X @ X_i * beta + pi_S @ W_i + eps_i,    i = 1, ..., B
        W ~ GP(0, sigma^2 * C(.; phi)),   C(h; phi) = exp(-phi * h)
        eps_i ~ N(0, tau^2 * I_K)

    where K = n_i is the block size and pi_X, pi_S are within-block
    permutation matrices shared across all B blocks.

    Parameters
    ----------
    B : int
        Total number of spatial blocks.  Any positive integer is accepted.
    n_i : int
        Number of observations per block (balanced design).
    beta_true : float
        True regression coefficient.
    phi_true : float
        True spatial range parameter of the exponential kernel
        C(h) = sigma^2 * exp(-phi * h).
    sigmasq_true : float
        True partial sill (GP marginal variance).
    tausq_true : float
        True nugget (measurement-error variance).
    seed : int
        Random seed for full reproducibility.
    n_rows : int or None
        Number of block rows in the spatial grid.  If None, inferred from B.
    n_cols : int or None
        Number of block columns in the spatial grid.  If None, inferred from B.
        Must satisfy n_rows * n_cols == B when both are supplied.

    Returns
    -------
    dict with the following keys:

    Inputs to the three models
        Y            : (B, n_i) tensor  – reshaped outcomes
        X            : (B, n_i) tensor  – permuted covariates, same shape
        Dist         : (N, N) tensor    – pairwise distances of *permuted* locations
        s_perm       : (N, 2) tensor    – permuted spatial locations
        x_perm_flat  : (N, 1) tensor    – permuted covariates (flat)
        region_assignments : (N,) tensor – block index for each observation

    Ground-truth quantities (for oracle baseline and evaluation)
        y            : (N,) tensor      – flat outcomes
        x            : (N, 1) tensor    – true (unpermuted) covariates
        s            : (N, 2) tensor    – true spatial locations
        Dist_true    : (N, N) tensor    – pairwise distances of true locations
        perm_matrix_x: (n_i, n_i) tensor – true covariate-permutation matrix
        perm_matrix_s: (n_i, n_i) tensor – true location-permutation matrix

    Scalars / layout
        beta_true, phi_true, sigmasq_true, tausq_true, B, n_i, N,
        n_rows, n_cols
    """
    # ── resolve grid dimensions ───────────────────────────────────────────
    if n_rows is None and n_cols is None:
        n_rows, n_cols = _best_grid(B)
    elif n_rows is None:
        n_rows = B // n_cols
    elif n_cols is None:
        n_cols = B // n_rows
    if n_rows * n_cols != B:
        raise ValueError(f"n_rows ({n_rows}) * n_cols ({n_cols}) = "
                         f"{n_rows * n_cols} ≠ B ({B})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    N = B * n_i

    # ── 1. True spatial locations ─────────────────────────────────────────
    s_blocks = []
    for b in range(B):
        row, col = b // n_cols, b % n_cols
        offset = torch.tensor([col / n_cols, row / n_rows], dtype=torch.float32)
        cell_size = torch.tensor([1.0 / n_cols, 1.0 / n_rows], dtype=torch.float32)
        s_blocks.append(torch.rand(n_i, 2) * cell_size + offset)
    s = torch.cat(s_blocks, dim=0)                          # (N, 2)
    region_assignments = torch.repeat_interleave(torch.arange(B), n_i)  # (N,)

    # ── 2. Covariates ─────────────────────────────────────────────────────
    x = torch.randn(N, 1)

    # ── 3. Gaussian process sample ────────────────────────────────────────
    Dist_true = torch.cdist(s, s, p=2)
    Dist_true = (Dist_true + Dist_true.T) / 2
    K = sigmasq_true * torch.exp(-phi_true * Dist_true) \
        + (tausq_true + 1e-6) * torch.eye(N)
    L = torch.linalg.cholesky(K)
    w = (L @ torch.randn(N)).detach()

    # ── 4. Outcome ────────────────────────────────────────────────────────
    y = (beta_true * x.squeeze() + w).detach()

    # ── 5. Apply the SAME random permutation to all blocks ────────────────
    perm_x = torch.randperm(n_i)
    perm_s = torch.randperm(n_i)

    # Permutation matrices (P[i, j] = 1  iff position i maps to position j)
    P_x = torch.zeros(n_i, n_i)
    P_x[torch.arange(n_i), perm_x] = 1.0
    P_s = torch.zeros(n_i, n_i)
    P_s[torch.arange(n_i), perm_s] = 1.0

    x_perm = x.clone()
    s_perm = s.clone()
    for b in range(B):
        sl = slice(b * n_i, (b + 1) * n_i)
        x_perm[sl] = x[sl][perm_x]
        s_perm[sl] = s[sl][perm_s]

    Dist_perm = torch.cdist(s_perm, s_perm, p=2)
    Dist_perm = (Dist_perm + Dist_perm.T) / 2

    return dict(
        # ── model inputs ──
        Y=y.reshape(B, n_i),
        X=x_perm.reshape(B, n_i),
        Dist=Dist_perm,
        s_perm=s_perm,
        x_perm_flat=x_perm,
        region_assignments=region_assignments,
        # ── ground truth ──
        y=y,
        x=x,
        s=s,
        Dist_true=Dist_true,
        perm_matrix_x=P_x,
        perm_matrix_s=P_s,
        # ── scalars / layout ──
        beta_true=beta_true,
        phi_true=phi_true,
        sigmasq_true=sigmasq_true,
        tausq_true=tausq_true,
        B=B,
        n_i=n_i,
        N=N,
        n_rows=n_rows,
        n_cols=n_cols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FullGP (oracle baseline)
# ─────────────────────────────────────────────────────────────────────────────

def train_oracle_gp(  # FullGP baseline
    s,
    x,
    y,
    max_iter: int = 5000,
    lr: float = 0.01,
    W: int = 200,
    tol: float = 1e-4,
    device: str = "cpu",
) -> dict:
    """
    Fit the FullGP model (known spatial locations and covariate links).

    This oracle model has access to the *true* (unpermuted) spatial locations
    and covariates and therefore serves as the gold-standard performance
    ceiling.  It maximises the Matern (nu=1/2) log-likelihood using AdamW
    with plateau-based early stopping.

    Stopping rule: halt when the relative range of the loss over the last
    ``W`` iterations falls below ``tol``, i.e.
    (max - min) / (|last| + 1e-12) < tol.

    Parameters
    ----------
    s        : (N, 2) tensor – true spatial locations
    x        : (N, 1) tensor – true (unpermuted) covariates
    y        : (N,)   tensor – outcomes
    max_iter : int   – maximum number of gradient steps
    lr       : float – AdamW learning rate
    W        : int   – plateau-detection window (number of recent iterations)
    tol      : float – relative-range threshold for plateau detection
    device   : str   – "cpu" or "cuda"

    Returns
    -------
    dict with keys:
        phi, sigmasq, tausq  : float – estimated GP parameters
        beta                 : float – estimated regression coefficient
        loss_history         : list[float] – negative log-likelihood per step
    """
    from GPModel import GPModel

    gp = GPModel().to(device)
    opt = optim.AdamW(gp.parameters(), lr=lr, weight_decay=0.01)
    s, x, y = s.to(device), x.to(device), y.to(device)

    history = []
    for it in tqdm(range(max_iter), desc="FullGP", leave=False):
        opt.zero_grad(set_to_none=True)
        loss = gp(s, x, y)
        loss.backward()
        opt.step()

        cur = float(loss.detach().item())
        history.append(cur)

        if it > W:
            recent = history[-W:]
            rel_range = (max(recent) - min(recent)) / (abs(recent[-1]) + 1e-12)
            if rel_range < tol:
                print(f"[Stop FullGP] plateau over last {W} iters at it={it}, loss={cur:.6f}")
                break

    return dict(
        phi=float(np.exp(gp.logphi.item())),
        sigmasq=float(np.exp(gp.logsigmasq.item())),
        tausq=float(np.exp(gp.logtausq.item())),
        beta=float(gp.beta.detach().cpu().squeeze().numpy()),
        loss_history=history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ArealGP baseline
# ─────────────────────────────────────────────────────────────────────────────

def train_areal_gp(
    s_perm,
    region_assignments,
    x_perm_flat,
    y,
    max_iter: int = 5000,
    lr: float = 0.01,
    W: int = 200,
    tol: float = 1e-4,
    device: str = "cpu",
) -> dict:
    """
    Fit the ArealGP model (block-averaged outcomes and covariates).

    This model ignores within-block linkages entirely.  It aggregates
    observations to block-level means and fits a GP on block centroids.
    Block-averaged outcomes are permutation-invariant, so Y_bar is
    computed from the observed (permuted) data without bias.  X_bar
    is also computed from the permuted covariates (as in the real-data
    scenario where the true X ordering is unknown).

    Stopping rule: halt when the relative range of the loss over the last
    ``W`` iterations falls below ``tol``, i.e.
    (max - min) / (|last| + 1e-12) < tol.

    Parameters
    ----------
    s_perm            : (N, 2) tensor – permuted spatial locations
    region_assignments: (N,)   tensor – block index (0 to B-1)
    x_perm_flat       : (N, 1) tensor – permuted covariates
    y                 : (N,)   tensor – outcomes
    max_iter : int   – maximum number of gradient steps
    lr       : float – AdamW learning rate
    W        : int   – plateau-detection window (number of recent iterations)
    tol      : float – relative-range threshold for plateau detection
    device   : str   – "cpu" or "cuda"

    Returns
    -------
    dict with keys:
        phi, sigmasq, tausq  : float – estimated GP parameters
        beta                 : float – estimated regression coefficient
        loss_history         : list[float] – negative log-likelihood per step
    """
    from GPArealModel import GPArealModel

    unique_regions = torch.unique(region_assignments)
    B = len(unique_regions)
    input_dim = x_perm_flat.shape[1]

    ybar = torch.zeros(B, device=device)
    xbar = torch.zeros(B, input_dim, device=device)
    for i, r in enumerate(unique_regions):
        idx = region_assignments == r
        ybar[i] = y[idx].mean()
        xbar[i] = x_perm_flat[idx].mean(dim=0)

    gpa = GPArealModel().to(device)
    opt = optim.AdamW(gpa.parameters(), lr=lr, weight_decay=0.01)
    s_perm = s_perm.to(device)
    region_assignments = region_assignments.to(device)

    history = []
    for it in tqdm(range(max_iter), desc="ArealGP", leave=False):
        opt.zero_grad(set_to_none=True)
        loss = gpa(s_perm, region_assignments, xbar, ybar)
        loss.backward()
        opt.step()

        cur = float(loss.detach().item())
        history.append(cur)

        if it > W:
            recent = history[-W:]
            rel_range = (max(recent) - min(recent)) / (abs(recent[-1]) + 1e-12)
            if rel_range < tol:
                print(f"[Stop ArealGP] plateau over last {W} iters at it={it}, loss={cur:.6f}")
                break

    return dict(
        phi=float(np.exp(gpa.logphi.item())),
        sigmasq=float(np.exp(gpa.logsigmasq.item())),
        tausq=float(np.exp(gpa.logtausq.item())),
        beta=float(gpa.beta.detach().cpu().squeeze().numpy()),
        loss_history=history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REPAIR
# ─────────────────────────────────────────────────────────────────────────────

def run_vigp_unlinked(
    data: dict,
    n_iter: int = 50,
    n_steps: int = 50,
    n_phi_samples: int = 100,
    n_piX_sample: int = 50,
    n_piS_sample: int = 50,
    tau: float = 0.9,
    lr_piX: float = 0.01,
    lr_piS: float = 0.01,
    anneal_every: int = 20,
    elbo_W: int = 5,
    tol: float = 0.1,
    seed: int = 521,
) -> dict:
    """
    Run REPAIR: variational inference for the doubly-unlinked spatial GP model.

    The algorithm alternates between closed-form coordinate-ascent VI (CAVI)
    updates for GP parameters and gradient-based (AdamW) updates for the
    soft-permutation variational distributions q(pi_X) and q(pi_S).
    Temperature annealing gradually concentrates the Gumbel-softmax relaxation
    toward hard permutation matrices over the course of optimisation.

    Variational family
    ------------------
    q(beta) = N(mu_beta, sigma^2_beta)
    q(W)    = N(mu_W, Sigma_W)
    q(tau^2)= Gamma(lambda_a1, lambda_b1)
    q(sigma^2)=Gamma(lambda_a2, lambda_b2)
    q(phi)  importance-weighted mixture over a uniform prior grid
    q(pi_X) soft-permutation with mean M_X_star and variance V_X_star
    q(pi_S) soft-permutation with mean M_S_star and variance V_S_star

    Parameters
    ----------
    data          : dict returned by ``generate_synthetic_data``
    n_iter        : int   – maximum outer VI iterations
    n_steps       : int   – AdamW gradient steps for pi_X / pi_S per outer iter
    n_phi_samples : int   – Monte Carlo samples for the phi marginal
    n_piX_sample  : int   – permutation samples for the covariate-link ELBO
    n_piS_sample  : int   – permutation samples for the location-link ELBO
    tau           : float – initial Gumbel-softmax temperature (in [0, 1])
    lr_piX        : float – AdamW learning rate for q(pi_X)
    lr_piS        : float – AdamW learning rate for q(pi_S)
    anneal_every  : int   – decay temperature every this many gradient steps
    elbo_W        : int   – sliding window size for ELBO plateau detection
    tol           : float – minimum % ELBO improvement to avoid early stopping
    seed          : int   – random seed

    Returns
    -------
    dict containing all variational parameters from REPAIR (``VIGP_Unlinked``) plus:

    est_perm_x   : (n_i, n_i) tensor – hard permutation estimate for X
    est_perm_s   : (n_i, n_i) tensor – hard permutation estimate for S
    phi_est      : float – posterior-mean spatial range
    beta_est     : float – posterior-mean regression coefficient
    sigmasq_est  : float – posterior-mean GP variance  [E = lambda_b1/(lambda_a1-1)]
    tausq_est    : float – posterior-mean nugget        [E = lambda_b2/(lambda_a2-1)]
    """
    from revised_VIGP_Unlinked import VIGP_Unlinked
    from utils import round_to_perm

    B, n_i = data["B"], data["n_i"]
    Dist = data["Dist"]

    prior_parameters = dict(
        a1=0.1, b1=0.1,
        a2=0.1, b2=0.1,
        eta_X_sq=0.1, eta_S_sq=0.1,
        mu_beta=0.0, sigmasq_beta=100.0,
        phi_prior_lb=float(1.0 / torch.max(Dist)),
        phi_prior_ub=10.0,
    )

    results = VIGP_Unlinked(
        n_iter=n_iter,
        n_blocks=B,
        n_locations=n_i,
        X=data["X"],
        Y=data["Y"],
        Dist=Dist,
        n_steps=n_steps,
        n_phi_samples=n_phi_samples,
        n_piX_sample=n_piX_sample,
        n_piS_sample=n_piS_sample,
        tau_X=tau,
        tau_S=tau,
        seed=seed,
        lr_piX=lr_piX,
        lr_piS=lr_piS,
        VX_ub=0.5,
        VS_ub=0.5,
        prior_parameters=prior_parameters,
        use_global_tau_anneal=True,
        anneal_every=anneal_every,
        elbo_W=elbo_W,
        tol=tol,
    )

    # ── Hard permutation estimates (Hungarian algorithm) ──────────────────
    est_perm_x = round_to_perm(results["M_X_star"].detach().numpy())
    est_perm_s = round_to_perm(results["M_S_star"].detach().numpy())

    # ── GP parameter point estimates ──────────────────────────────────────
    la1 = float(results["lambda_a1"])
    lb1 = float(results["lambda_b1"])
    la2 = float(results["lambda_a2"])
    lb2 = float(results["lambda_b2"])

    results.update(
        est_perm_x=torch.tensor(est_perm_x),
        est_perm_s=torch.tensor(est_perm_s),
        phi_est=float(results["mean_phi"]),
        beta_est=float(results["mu_lambda_beta"]),
        sigmasq_est=lb1 / max(la1 - 1.0, 1e-8),   # E[sigma^2] under Gamma
        tausq_est=lb2 / max(la2 - 1.0, 1e-8),      # E[tau^2]   under Gamma
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def permutation_accuracy(est_perm, true_perm) -> float:
    """
    Proportion of within-block observations correctly matched by a permutation.

    Computed as  trace(P_est^T P_true) / n_i,  which equals the fraction of
    indices i such that P_est assigns i to the same position as P_true.

    Parameters
    ----------
    est_perm  : (n_i, n_i) array or tensor – estimated hard permutation matrix
    true_perm : (n_i, n_i) array or tensor – ground-truth permutation matrix

    Returns
    -------
    float in [0, 1]
    """
    if isinstance(est_perm, torch.Tensor):
        est_perm = est_perm.numpy()
    if isinstance(true_perm, torch.Tensor):
        true_perm = true_perm.numpy()
    n_i = est_perm.shape[0]
    return float(np.sum(est_perm * true_perm)) / n_i


def summarise_results(
    oracle: dict,
    areal: dict,
    vigp: dict,
    data: dict,
) -> "pd.DataFrame":
    """
    Build a parameter-comparison table across all three methods.

    Parameters
    ----------
    oracle : dict from ``train_oracle_gp``  (FullGP)
    areal  : dict from ``train_areal_gp``   (ArealGP)
    vigp   : dict from ``run_vigp_unlinked`` (REPAIR)
    data   : dict from ``generate_synthetic_data``

    Returns
    -------
    pandas.DataFrame  with rows = parameters, columns = [True, FullGP,
                       ArealGP, REPAIR]
    """
    import pandas as pd

    rows = {
        "True":    [data["beta_true"], data["phi_true"],
                    data["sigmasq_true"], data["tausq_true"]],
        "FullGP":  [oracle["beta"], 1/oracle["phi"],
                    oracle["sigmasq"], oracle["tausq"]],
        "ArealGP": [areal["beta"], 1/areal["phi"],
                    areal["sigmasq"], areal["tausq"]],
        "REPAIR":  [vigp["beta_est"],vigp["phi_est"],
                    vigp["sigmasq_est"], vigp["tausq_est"]],
    }
    return pd.DataFrame(rows, index=["β", "φ", "σ²", "τ²"]).round(3)
