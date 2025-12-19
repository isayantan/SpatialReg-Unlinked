import math
import torch
import torch.optim as optim
from tqdm import tqdm

from elbo import vi_piX, vi_piS
from utils import nearest_pd_torch, compute_q_phi, round_to_perm


# -----------------------------
# ELBO helper functions
# -----------------------------
def _safe_slogdet_spd(A: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Stable log|A| for (near) SPD matrices."""
    A = (A + A.T) / 2
    A = A + jitter * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    sign, ld = torch.linalg.slogdet(A)
    # if sign <= 0, ld may be nan/inf; jitter usually avoids this
    return ld


def _invgamma_E_log_p(a: torch.Tensor, b: torch.Tensor, E_log_x: torch.Tensor, E_inv_x: torch.Tensor) -> torch.Tensor:
    """
    For Inv-Gamma(a,b) with density: b^a / Gamma(a) * x^{-a-1} exp(-b/x).
    """
    return a * torch.log(b) - torch.lgamma(a) - (a + 1) * E_log_x - b * E_inv_x


def _invgamma_entropy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Entropy of Inv-Gamma(a,b) with the same parameterization:
    H = a + log b + lgamma(a) - (1+a) psi(a)
    """
    return a + torch.log(b) + torch.lgamma(a) - (1 + a) * torch.digamma(a)


def _normal_E_log_p(mu_q, var_q, mu_p, var_p) -> torch.Tensor:
    """E_q[log N(mu_p, var_p)] when q=N(mu_q, var_q) (scalar)."""
    quad = (mu_q - mu_p) ** 2 + var_q
    return -0.5 * (math.log(2 * math.pi) + torch.log(var_p) + quad / var_p)


def _normal_entropy(var_q) -> torch.Tensor:
    """Entropy of scalar Normal with variance var_q."""
    return 0.5 * (math.log(2 * math.pi * math.e) + torch.log(var_q))


def compute_global_elbo_monitor(
    *,
    # dims
    n_blocks: int,
    n_locations: int,
    # current "expected SSE" term from your lambda_b2 update
    E_SSE_term: torch.Tensor,
    # current VB params
    mu_lambda_beta: torch.Tensor,
    sigmasq_lambda_beta: torch.Tensor,
    mu_W: torch.Tensor,
    Sigma_W: torch.Tensor,
    lambda_a1: torch.Tensor,
    lambda_b1: torch.Tensor,
    lambda_a2: torch.Tensor,
    lambda_b2: torch.Tensor,
    # priors
    a1: torch.Tensor, b1: torch.Tensor,
    a2: torch.Tensor, b2: torch.Tensor,
    mu_beta: torch.Tensor, sigmasq_beta: torch.Tensor,
    # phi IS objects (optional)
    include_gp_prior: bool,
    Dist: torch.Tensor = None,
    phi_samples_kept: torch.Tensor = None,
    iw_kept: torch.Tensor = None,
    Rinv_kept: list = None,
    # permutation ELBO pieces
    elbo_piX_hat: float = 0.0,
    elbo_piS_hat: float = 0.0,
    # phi uniform prior support (optional constant term)
    phi_prior_lb: torch.Tensor = None,
    phi_prior_ub: torch.Tensor = None,
) -> torch.Tensor:
    """
    Returns a *monitoring* global ELBO scalar. Includes:
      - E log-likelihood (Gaussian)
      - beta prior + entropy
      - IG priors + entropies for sigma^2, tau^2
      - entropy for q(W)
      - optional GP prior term using phi importance sampling (logdet + quadratic)
      - + piX/piS ELBO pieces from your vi modules
      - + constant E log p(phi) for Uniform(lb,ub) if provided
    """

    N = n_blocks * n_locations  # total spatial locations stacked

    # -------- expectations under q(tau^2) and q(sigma^2) ----------
    E_tau_inv = lambda_a2 / lambda_b2
    E_log_tau2 = torch.log(lambda_b2) - torch.digamma(lambda_a2)

    E_sig_inv = lambda_a1 / lambda_b1
    E_log_sig2 = torch.log(lambda_b1) - torch.digamma(lambda_a1)

    # -------- expected log-likelihood ----------
    # Y | ... ~ N( mean, tau^2 I )
    # E log p(Y|.) = -N/2 log(2π) - N/2 E log tau^2 - 1/2 E(1/tau^2) * E[SSE]
    elbo_like = (
        -0.5 * N * math.log(2 * math.pi)
        -0.5 * N * E_log_tau2
        -0.5 * E_tau_inv * E_SSE_term
    )

    # -------- beta prior + entropy ----------
    # prior: beta ~ N(mu_beta, sigmasq_beta)
    # q: beta ~ N(mu_lambda_beta, sigmasq_lambda_beta)
    elbo_beta = _normal_E_log_p(mu_lambda_beta, sigmasq_lambda_beta, mu_beta, sigmasq_beta) \
                + _normal_entropy(sigmasq_lambda_beta)

    # -------- Inv-Gamma priors + entropies ----------
    # prior sigma^2 ~ IG(a1,b1), q sigma^2 ~ IG(lambda_a1, lambda_b1)
    elbo_sig2 = _invgamma_E_log_p(a1, b1, E_log_sig2, E_sig_inv) + _invgamma_entropy(lambda_a1, lambda_b1)

    # prior tau^2 ~ IG(a2,b2), q tau^2 ~ IG(lambda_a2, lambda_b2)
    elbo_tau2 = _invgamma_E_log_p(a2, b2, E_log_tau2, E_tau_inv) + _invgamma_entropy(lambda_a2, lambda_b2)

    # -------- entropy for q(W) ----------
    # q(W) = N(mu_W, Sigma_W)
    # H = 0.5 [ N log(2πe) + log|Sigma_W| ]
    elbo_W_entropy = 0.5 * (N * math.log(2 * math.pi * math.e) + _safe_slogdet_spd(Sigma_W))

    # -------- optional GP prior term: E log p(W | sigma^2, phi) ----------
    elbo_gp = torch.tensor(0.0, device=mu_W.device, dtype=mu_W.dtype)

    if include_gp_prior:
        if (Dist is None) or (phi_samples_kept is None) or (iw_kept is None) or (Rinv_kept is None) or (len(Rinv_kept) == 0):
            # if not enough info, skip GP prior term safely
            elbo_gp = torch.tensor(0.0, device=mu_W.device, dtype=mu_W.dtype)
        else:
            mu_W_flat = mu_W.flatten()
            V_W = Sigma_W + torch.outer(mu_W_flat, mu_W_flat)

            # E_logdet_R and E_tr(Rinv V_W) w.r.t q(phi) approximated by IS
            E_logdet_R = torch.tensor(0.0, device=mu_W.device, dtype=mu_W.dtype)
            E_tr = torch.tensor(0.0, device=mu_W.device, dtype=mu_W.dtype)

            for w, phi, Rinv in zip(iw_kept, phi_samples_kept, Rinv_kept):
                # IMPORTANT: this matches your current R(phi)=exp(-(1/phi)*Dist)
                R = torch.exp(-(1.0 / phi) * Dist)
                E_logdet_R = E_logdet_R + w * _safe_slogdet_spd(R)
                E_tr = E_tr + w * torch.trace(Rinv @ V_W)

            # log p(W|sigma^2,phi): -N/2 log(2π) - 1/2[N log sigma^2 + log|R|] - 1/(2 sigma^2) W^T R^{-1} W
            # Replace W^T R^{-1} W with tr(R^{-1} V_W) under q(W)
            elbo_gp = (
                -0.5 * N * math.log(2 * math.pi)
                -0.5 * (N * E_log_sig2 + E_logdet_R)
                -0.5 * E_sig_inv * E_tr
            )

    # -------- phi prior term: Uniform(lb,ub) constant if provided ----------
    elbo_phi_prior = torch.tensor(0.0, device=mu_W.device, dtype=mu_W.dtype)
    if (phi_prior_lb is not None) and (phi_prior_ub is not None):
        elbo_phi_prior = -torch.log(phi_prior_ub - phi_prior_lb)

    # -------- add permutation ELBO pieces (from your modules) ----------
    elbo_pi = torch.tensor(elbo_piX_hat + elbo_piS_hat, device=mu_W.device, dtype=mu_W.dtype)

    return elbo_like + elbo_beta + elbo_sig2 + elbo_tau2 + elbo_W_entropy + elbo_gp + elbo_phi_prior + elbo_pi


# -----------------------------
# Main VB routine (DROP-IN)
# -----------------------------
def VIGP_Unlinked(
    n_iter,
    n_blocks,
    n_locations,
    X, Y, Dist,
    n_steps=10,
    phi_init=3.0,
    n_phi_samples=100,
    n_piX_sample=10,
    n_piS_sample=10,
    tau_X=0.1, tau_S=0.1,
    seed=100,
    fix_mu_lambda_beta=False,
    fix_sigmasq_lambda_beta=False,
    fix_lambda_b1=False,
    fix_lambda_b2=False,
    fix_piX=False,
    fix_piS=False,
    fix_mu_W=False,
    fix_Sigma_W=False,
    fix_mean_Rphi_inv=False,
    sigmasq_lambda_beta_fixed=0.1,
    mu_lambda_beta_fixed=0.1,
    lambda_b1_fixed=0.1,
    lambda_b2_fixed=0.1,
    M_X_star_fixed=None,
    V_X_star_fixed=None,
    M_S_star_fixed=None,
    V_S_star_fixed=None,
    mu_W_fixed=None,
    Sigma_W_fixed=None,
    mean_Rphi_inv_fixed=None,
    pi_X_true=None,
    pi_S_true=None,
    VX_ub=0.5,
    VS_ub=0.5,
    lr_piX=0.1,
    lr_piS=0.1,
    prior_parameters={},

    # ---- NEW: ELBO convergence controls ----
    use_elbo_convergence=True,
    include_gp_prior_in_elbo=True,  # optional GP prior via phi-IS
    elbo_delta=1e-5,
    elbo_eps=1e-10,
    elbo_patience=5,
    elbo_smooth_alpha=0.1,
    elbo_check_every=1,
    iw_min=1e-5,                   # threshold for keeping IS weights for GP ELBO
):
    # Prior hyperparameters (as tensors for math stability)
    a1 = torch.as_tensor(prior_parameters["a1"], dtype=Y.dtype, device=Y.device)
    b1 = torch.as_tensor(prior_parameters["b1"], dtype=Y.dtype, device=Y.device)
    a2 = torch.as_tensor(prior_parameters["a2"], dtype=Y.dtype, device=Y.device)
    b2 = torch.as_tensor(prior_parameters["b2"], dtype=Y.dtype, device=Y.device)
    eta_X_sq = torch.as_tensor(prior_parameters["eta_X_sq"], dtype=Y.dtype, device=Y.device)
    eta_S_sq = torch.as_tensor(prior_parameters["eta_S_sq"], dtype=Y.dtype, device=Y.device)

    mu_beta = torch.as_tensor(prior_parameters["mu_beta"], dtype=Y.dtype, device=Y.device)
    sigmasq_beta = torch.as_tensor(prior_parameters["sigmasq_beta"], dtype=Y.dtype, device=Y.device)

    phi_prior_lb = torch.as_tensor(prior_parameters["phi_prior_lb"], dtype=Y.dtype, device=Y.device)
    phi_prior_ub = torch.as_tensor(prior_parameters["phi_prior_ub"], dtype=Y.dtype, device=Y.device)

    # Initialize VB parameters
    mu_lambda_beta = torch.as_tensor(0.1, dtype=Y.dtype, device=Y.device)
    sigmasq_lambda_beta = torch.as_tensor(0.1, dtype=Y.dtype, device=Y.device)

    mu_W = torch.zeros(n_blocks, n_locations, dtype=Y.dtype, device=Y.device)
    Sigma_W = torch.eye(n_blocks * n_locations, dtype=Y.dtype, device=Y.device)

    # R(phi) parameterization consistent with your code: exp(-(1/phi)*Dist)
    Rphi = torch.exp(-(1.0 / phi_init) * Dist)
    mean_Rphi_inv = torch.linalg.inv(nearest_pd_torch(Rphi, epsilon=0.01))

    lambda_a1 = torch.as_tensor(n_blocks * n_locations * 0.5, dtype=Y.dtype, device=Y.device) + a1
    lambda_b1 = torch.as_tensor(0.5, dtype=Y.dtype, device=Y.device)
    lambda_a2 = torch.as_tensor(n_blocks * n_locations * 0.5, dtype=Y.dtype, device=Y.device) + a2
    lambda_b2 = torch.as_tensor(0.5, dtype=Y.dtype, device=Y.device)

    # Initialize permutation variational models
    model_piX = vi_piX(n_locations=n_locations).to(Y.device)
    model_piS = vi_piS(n_locations=n_locations, n_blocks=n_blocks).to(Y.device)

    M_X_star = model_piX.current_M_X_star.clone().data.to(Y.device)
    V_X_star = model_piX.current_V_X_star.clone().data.to(Y.device)
    M_S_star = model_piS.current_M_S_star.clone().data.to(Y.device)
    V_S_star = model_piS.current_V_S_star.clone().data.to(Y.device)

    optimizer_piX = optim.AdamW(model_piX.parameters(), lr=lr_piX, weight_decay=1e-2)
    optimizer_piS = optim.AdamW(model_piS.parameters(), lr=lr_piS, weight_decay=1e-2)

    # Tracking
    loss_vector = torch.zeros(n_iter, dtype=Y.dtype, device=Y.device)

    # ELBO convergence state
    L_prev = None
    Lbar = None
    good_count = 0
    elbo_global_trace = []
    elbo_pi_trace = []

    mean_phi = torch.as_tensor(phi_init, dtype=Y.dtype, device=Y.device)

    for it in tqdm(range(n_iter)):
        # ----------------------
        # Update beta variance
        # ----------------------
        if not fix_sigmasq_lambda_beta:
            X_V_X_star_X = torch.einsum('bi,ij,bj->b', X, V_X_star, X).sum()
            sigmasq_lambda_beta = 1.0 / (((lambda_a2 * X_V_X_star_X) / lambda_b2) + (1.0 / sigmasq_beta))
        else:
            sigmasq_lambda_beta = torch.as_tensor(sigmasq_lambda_beta_fixed, dtype=Y.dtype, device=Y.device)

        # ----------------------
        # Update beta mean
        # ----------------------
        if not fix_mu_lambda_beta:
            residual = Y - (M_S_star @ mu_W.T).T
            X_M_X_star_residual = torch.einsum('bi,ij,bj->b', X, M_X_star.T, residual).sum()
            mu_lambda_beta = sigmasq_lambda_beta * ((lambda_a2 / lambda_b2) * X_M_X_star_residual + mu_beta / sigmasq_beta)
        else:
            mu_lambda_beta = torch.as_tensor(mu_lambda_beta_fixed, dtype=Y.dtype, device=Y.device)

        # ----------------------
        # Update lambda_b1
        # ----------------------
        if not fix_lambda_b1:
            mu_W_flat = mu_W.flatten()
            V_W = Sigma_W + torch.outer(mu_W_flat, mu_W_flat)
            lambda_b1 = 0.5 * torch.trace(mean_Rphi_inv @ V_W) + b1
        else:
            lambda_b1 = torch.as_tensor(lambda_b1_fixed, dtype=Y.dtype, device=Y.device)

        # ----------------------
        # Update lambda_b2 (and keep E[SSE] term for ELBO)
        # ----------------------
        if not fix_lambda_b2:
            res = Y - mu_lambda_beta * (M_X_star @ X.T).T - (M_S_star @ mu_W.T).T
            term = torch.trace(res.T @ res)
            term = term + mu_lambda_beta ** 2 * torch.trace(X @ (V_X_star - M_X_star.T @ M_X_star) @ X.T)
            term = term + sigmasq_lambda_beta * torch.trace(X @ V_X_star @ X.T)
            term = term + torch.trace(mu_W @ (V_S_star - M_S_star.T @ M_S_star) @ mu_W.T)
            term = term + torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)
            lambda_b2 = 0.5 * term + b2
            E_SSE_term = term.detach()
        else:
            lambda_b2 = torch.as_tensor(lambda_b2_fixed, dtype=Y.dtype, device=Y.device)
            # best effort if fixed: use current residual-based term
            res = Y - mu_lambda_beta * (M_X_star @ X.T).T - (M_S_star @ mu_W.T).T
            E_SSE_term = torch.trace(res.T @ res).detach()

        # ----------------------
        # Update Sigma_W
        # ----------------------
        if not fix_Sigma_W:
            term1 = (lambda_a2 / lambda_b2) * torch.block_diag(*[V_S_star] * n_blocks)
            term2 = (lambda_a1 / lambda_b1) * mean_Rphi_inv
            Sigma_W_inv = term1 + term2
            Sigma_W_inv = (Sigma_W_inv + Sigma_W_inv.T) / 2
            Sigma_W = torch.linalg.pinv(Sigma_W_inv, hermitian=True, rtol=1e-4)
            Sigma_W = (Sigma_W + Sigma_W.T) / 2
        else:
            Sigma_W = Sigma_W_fixed

        # ----------------------
        # Update mu_W
        # ----------------------
        if not fix_mu_W:
            mu_W = (lambda_a2 / lambda_b2) * (
                Sigma_W @ (M_S_star.T @ Y.T - mu_lambda_beta * M_S_star.T @ M_X_star @ X.T).T.flatten()
            ).reshape(n_blocks, n_locations)
        else:
            mu_W = mu_W_fixed

        # ----------------------
        # Importance sampling for phi and mean_Rphi_inv
        # Also store kept samples/weights/Rinv for GP ELBO
        # ----------------------
        phi_samples_kept = None
        iw_kept = None
        Rinv_kept = None

        if not fix_mean_Rphi_inv:
            phi_samples = torch.rand(n_phi_samples, device=Y.device, dtype=Y.dtype) * (phi_prior_ub - phi_prior_lb) + phi_prior_lb
            q_phi_values = [compute_q_phi(phi, Dist, mu_W, Sigma_W, lambda_a1, lambda_b1) for phi in phi_samples]

            logw = torch.tensor([q_phi_values[i][0] for i in range(n_phi_samples)], device=Y.device, dtype=Y.dtype)
            iw = torch.nn.functional.softmax(logw, dim=0)

            Rphi_inv_sum = torch.zeros_like(Dist)
            phi_sum = torch.tensor(0.0, device=Y.device, dtype=Y.dtype)
            w_sum = torch.tensor(0.0, device=Y.device, dtype=Y.dtype)

            # Keep only non-negligible weights (for stability and speed)
            kept_phis = []
            kept_ws = []
            kept_Rinv = []

            for i in range(n_phi_samples):
                if iw[i] > iw_min:
                    Rinv_i = q_phi_values[i][1]
                    Rphi_inv_sum += iw[i] * Rinv_i
                    phi_sum += iw[i] * phi_samples[i]
                    w_sum += iw[i]

                    if include_gp_prior_in_elbo:
                        kept_phis.append(phi_samples[i])
                        kept_ws.append(iw[i])
                        kept_Rinv.append(Rinv_i)

            mean_Rphi_inv = Rphi_inv_sum / w_sum
            mean_phi = phi_sum / w_sum

            if include_gp_prior_in_elbo and len(kept_ws) > 0:
                iw_kept = torch.stack(kept_ws)
                iw_kept = iw_kept / iw_kept.sum()  # renormalize after truncation
                phi_samples_kept = torch.stack(kept_phis)
                Rinv_kept = kept_Rinv
        else:
            mean_Rphi_inv = mean_Rphi_inv_fixed
            mean_phi = torch.as_tensor(phi_init, dtype=Y.dtype, device=Y.device)

        # ----------------------
        # Optimize piX and get elbo_piX_hat
        # ----------------------
        if not fix_piX:
            prev_lossX = None
            for step in range(n_steps):
                optimizer_piX.zero_grad()
                lossX = model_piX(
                    Y, X, mu_lambda_beta, sigmasq_lambda_beta, M_S_star, mu_W,
                    eta_X_sq, lambda_a2, lambda_b2, tau_X,
                    n_piX_sample, VX_ub=VX_ub, seed=seed
                )
                lossX.backward()
                optimizer_piX.step()

                if prev_lossX is not None and abs(prev_lossX - lossX.item()) < 1e-4:
                    break
                prev_lossX = lossX.item()

            M_X_star = model_piX.current_M_X_star.clone().data
            V_X_star = model_piX.current_V_X_star.clone().data
        else:
            M_X_star = M_X_star_fixed
            V_X_star = V_X_star_fixed

        # Always evaluate current piX ELBO piece consistently (even if fixed)
        with torch.no_grad():
            lossX_eval = model_piX(
                Y, X, mu_lambda_beta, sigmasq_lambda_beta, M_S_star, mu_W,
                eta_X_sq, lambda_a2, lambda_b2, tau_X,
                n_piX_sample, VX_ub=VX_ub, seed=seed
            )
        elbo_piX_hat = -float(lossX_eval.item())

        # ----------------------
        # Optimize piS and get elbo_piS_hat
        # ----------------------
        if not fix_piS:
            prev_lossS = None
            for step in range(n_steps):
                optimizer_piS.zero_grad()
                lossS = model_piS(
                    Y, X, mu_lambda_beta, M_X_star,
                    lambda_a2, lambda_b2, mu_W, Sigma_W,
                    eta_S_sq, tau_S, n_piS_sample, VS_ub=VS_ub, seed=seed
                )
                lossS.backward()
                optimizer_piS.step()

                if prev_lossS is not None and abs(prev_lossS - lossS.item()) < 1e-4:
                    break
                prev_lossS = lossS.item()

            M_S_star = model_piS.current_M_S_star.clone().data
            V_S_star = model_piS.current_V_S_star.clone().data
        else:
            M_S_star = M_S_star_fixed
            V_S_star = V_S_star_fixed

        with torch.no_grad():
            lossS_eval = model_piS(
                Y, X, mu_lambda_beta, M_X_star,
                lambda_a2, lambda_b2, mu_W, Sigma_W,
                eta_S_sq, tau_S, n_piS_sample, VS_ub=VS_ub, seed=seed
            )
        elbo_piS_hat = -float(lossS_eval.item())

        # ----------------------
        # GLOBAL ELBO MONITOR (fuller)
        # ----------------------
        L_global = compute_global_elbo_monitor(
            n_blocks=n_blocks,
            n_locations=n_locations,
            E_SSE_term=E_SSE_term,
            mu_lambda_beta=mu_lambda_beta,
            sigmasq_lambda_beta=sigmasq_lambda_beta,
            mu_W=mu_W,
            Sigma_W=Sigma_W,
            lambda_a1=lambda_a1, lambda_b1=lambda_b1,
            lambda_a2=lambda_a2, lambda_b2=lambda_b2,
            a1=a1, b1=b1, a2=a2, b2=b2,
            mu_beta=mu_beta, sigmasq_beta=sigmasq_beta,
            include_gp_prior=include_gp_prior_in_elbo,
            Dist=Dist,
            phi_samples_kept=phi_samples_kept,
            iw_kept=iw_kept,
            Rinv_kept=Rinv_kept,
            elbo_piX_hat=elbo_piX_hat,
            elbo_piS_hat=elbo_piS_hat,
            phi_prior_lb=phi_prior_lb,
            phi_prior_ub=phi_prior_ub,
        )

        L_pi = elbo_piX_hat + elbo_piS_hat
        elbo_global_trace.append(float(L_global.detach().item()))
        elbo_pi_trace.append(float(L_pi))

        # Smooth ELBO for robust stopping
        L_hat = float(L_global.detach().item())
        if Lbar is None:
            Lbar = L_hat
        else:
            Lbar = (1 - elbo_smooth_alpha) * Lbar + elbo_smooth_alpha * L_hat

        # Convergence check
        if use_elbo_convergence and (it % elbo_check_every == 0):
            if L_prev is not None:
                relchg = abs(Lbar - L_prev) / (abs(L_prev) + elbo_eps)
                if relchg < elbo_delta:
                    good_count += 1
                else:
                    good_count = 0

                if good_count >= elbo_patience:
                    print(f"\n[GLOBAL ELBO CONVERGED] iter={it+1} relchg={relchg:.2e} "
                          f"ELBO_smooth={Lbar:.6f} ELBO_raw={L_hat:.6f}")
                    loss_vector = loss_vector[:it+1]
                    break
            L_prev = Lbar

        # ----------------------
        # Your existing diagnostics
        # ----------------------
        print(f"Iter {it+1}/{n_iter} | mu_lambda_beta: {float(mu_lambda_beta):.4f} | "
              f"sigmasq_lambda_beta: {float(sigmasq_lambda_beta):.4f} | "
              f"lambda_a1: {float(lambda_a1):.4f} | lambda_b1: {float(lambda_b1):.4f} | "
              f"lambda_a2: {float(lambda_a2):.4f} | lambda_b2: {float(lambda_b2):.4f}")
        print(f"‣  E[ϕ]: {float(mean_phi):.4f} | "
              f"‣ E[sigmasq]*E[ϕ]: {float((lambda_b1/(lambda_a1-1))*mean_phi):.4f} | "
              f"‣ ||mu_W||: {float(torch.norm(mu_W)):.4f} | "
              f"‣ ELBO_global_raw: {L_hat:.6f} | ELBO_global_smooth: {Lbar:.6f} | "
              f"‣ ELBO_pi: {L_pi:.6f}")

        if pi_X_true is not None:
            est_perm_piX = round_to_perm(M_X_star.detach().cpu().numpy())
            correct_permutations = torch.sum(torch.tensor(est_perm_piX, device=Y.device, dtype=Y.dtype) * pi_X_true)
            print(f"Number of correct permutations recognized for piX: {int(correct_permutations.item())}")

        if pi_S_true is not None:
            est_perm_piS = round_to_perm(M_S_star.detach().cpu().numpy())
            correct_permutations = torch.sum(torch.tensor(est_perm_piS, device=Y.device, dtype=Y.dtype) * pi_S_true)
            print(f"Number of correct permutations recognized for piS: {int(correct_permutations.item())}")

        # Your RMSE-ish loss
        resid = Y - mu_lambda_beta * (M_X_star @ X.T).T - (M_S_star @ mu_W.T).T
        total_loss = torch.trace(resid.T @ resid)
        total_loss += mu_lambda_beta ** 2 * torch.trace(X @ (V_X_star - M_X_star.T @ M_X_star) @ X.T)
        total_loss += sigmasq_lambda_beta * torch.trace(X @ V_X_star @ X.T)
        total_loss += torch.trace(mu_W @ (V_S_star - M_S_star.T @ M_S_star) @ mu_W.T)
        total_loss += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)
        rmseish = torch.sqrt(total_loss / (n_locations * n_blocks))
        print(f"Total Loss (RMSE-ish): {float(rmseish):.4f}")

        loss_vector[it] = total_loss / (n_locations * n_blocks)

    # Return parameters
    parameters = {
        "M_X_star": M_X_star,
        "mean_Rphi_inv": mean_Rphi_inv,
        "V_X_star": V_X_star,
        "M_S_star": M_S_star,
        "V_S_star": V_S_star,
        "mu_lambda_beta": mu_lambda_beta,
        "sigmasq_lambda_beta": sigmasq_lambda_beta,
        "lambda_a1": lambda_a1,
        "lambda_b1": lambda_b1,
        "lambda_a2": lambda_a2,
        "lambda_b2": lambda_b2,
        "mean_phi": mean_phi,
        "loss_vector": loss_vector.detach().cpu(),
        "elbo_global_trace": torch.tensor(elbo_global_trace),
        "elbo_pi_trace": torch.tensor(elbo_pi_trace),
    }
    return parameters
