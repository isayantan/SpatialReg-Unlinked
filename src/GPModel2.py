# GPModel2.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPModel(nn.Module):
    """
    Natural-parameter GP with smooth positivity via softplus() in forward.
    nu is FIXED (not learned). Uses Cholesky-based NLL with adaptive jitter.
    """

    def __init__(
        self,
        input_dim: int = 1,
        device: str = "cpu",
        phi_init: float = 1.0,
        sigmasq_init: float = 1.0,
        tausq_init: float = 0.1,
        nu_fixed: float = 0.5,     # <-- fixed nu
        eps: float = 1e-6,
        jitter: float = 1e-6,
        max_chol_tries: int = 6,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        self.jitter = jitter
        self.max_chol_tries = max_chol_tries

        # nu is fixed (buffer, not a Parameter)
        self.register_buffer("nu_fixed", torch.tensor(float(nu_fixed), device=device))

        # Unconstrained raw params (NOT log). Positivity enforced via softplus in forward.
        self.phi_raw = nn.Parameter(torch.tensor(phi_init, device=device))
        self.sigmasq_raw = nn.Parameter(torch.tensor(sigmasq_init, device=device))
        self.tausq_raw = nn.Parameter(torch.tensor(tausq_init, device=device))

        self.beta = nn.Parameter(torch.zeros(input_dim, device=device))

    # ---- Convenience getters (positive params) ----
    @property
    def nu(self):
        # constant (no gradient)
        return self.nu_fixed

    @property
    def phi(self):
        return F.softplus(self.phi_raw) + self.eps

    @property
    def sigmasq(self):
        return F.softplus(self.sigmasq_raw) + self.eps

    @property
    def tausq(self):
        return F.softplus(self.tausq_raw) + self.eps

    def GP_nll(self, y: torch.Tensor, x: torch.Tensor, beta: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood for y ~ N(x beta, K) using Cholesky (stable).
        Adds adaptive jitter to ensure PD.
        """
        n = y.shape[0]
        r = y - x @ beta

        I = torch.eye(n, device=K.device, dtype=K.dtype)
        jitter_now = self.jitter

        L = None
        for _ in range(self.max_chol_tries):
            try:
                L = torch.linalg.cholesky(K + jitter_now * I)
                break
            except RuntimeError:
                jitter_now *= 10.0

        if L is None:
            # Keep graph but penalize hard if still failing
            return (K * 0.0).sum() + 1e10

        alpha = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
        quad = (r * alpha).sum()

        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * (logdet + quad + n * math.log(2.0 * math.pi))

    def forward(self, s: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = y.shape[0]
        dists = torch.cdist(s, s, p=2)
        dists = (dists + dists.T) / 2

        sigmasq = self.sigmasq
        phi = self.phi
        tausq = self.tausq

        # Exponential kernel: exp(-(1/phi) * d)
        K = sigmasq * torch.exp(-(1.0 / phi) * dists)
        K = K + tausq * torch.eye(n, device=self.device, dtype=K.dtype)

        return self.GP_nll(y, x, self.beta, K)