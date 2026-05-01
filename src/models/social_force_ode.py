"""
Phase 1: Social Force ODE Layer

SocialForceFunc  - ODE dynamics function for torchdiffeq
SocialForceODELayer - wraps odeint for trajectory integration

State tensor y: (..., N, 4)  where last dim = [x, y, vx, vy]

Social force on agent i from agent j:
    f_ij = w_ij * exp(-r_ij / sigma_ij) * (xi - xj) / r_ij   [repulsion]

Phase 1: w, sigma are scalar learnable parameters (shared across all edges).
Phase 2: override via set_edge_coefficients() with per-edge (N,N) tensors from GNN.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint


class SocialForceFunc(nn.Module):
    """
    Computes dy/dt = f(t, y) using Social Force dynamics.

    Learnable parameters (Phase 1):
        log_w     : log of repulsion strength  w = exp(log_w)
        log_sigma : log of influence range     sigma = exp(log_sigma)

    In Phase 2 these are replaced by per-edge tensors from the GNN
    via set_edge_coefficients().
    """

    def __init__(self, n_agents: int, r_thresh: float = 10.0) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.r_thresh = r_thresh

        # Scalar params in log-space → positivity guaranteed
        self.log_w = nn.Parameter(torch.tensor(0.0))      # w     = 1.0
        self.log_sigma = nn.Parameter(torch.tensor(0.0))  # sigma = 1.0

        self._w_ext: Optional[torch.Tensor] = None
        self._sigma_ext: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Phase 2 interface
    # ------------------------------------------------------------------

    def set_edge_coefficients(
        self, w: torch.Tensor, sigma: torch.Tensor
    ) -> None:
        """
        Inject per-edge coefficients from GNN.
        w, sigma: (..., N, N)  must be called before odeint().
        """
        self._w_ext = w
        self._sigma_ext = sigma

    def reset_edge_coefficients(self) -> None:
        self._w_ext = None
        self._sigma_ext = None

    # ------------------------------------------------------------------
    # ODE dynamics
    # ------------------------------------------------------------------

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar time (unused explicitly; required by torchdiffeq API)
            y: (..., N, 4)  [x, y, vx, vy] per agent

        Returns:
            dy/dt: (..., N, 4)  [vx, vy, ax, ay]
        """
        pos = y[..., :2]   # (..., N, 2)
        vel = y[..., 2:]   # (..., N, 2)

        w, sigma = self._resolve_coefficients()
        force = self._social_force(pos, w, sigma)  # (..., N, 2)

        # Kinematics: d(pos)/dt = vel,  d(vel)/dt = F  (mass = 1)
        return torch.cat([vel, force], dim=-1)

    # ------------------------------------------------------------------

    def _resolve_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._w_ext is not None:
            return self._w_ext, self._sigma_ext
        return torch.exp(self.log_w), torch.exp(self.log_sigma)

    def _social_force(
        self,
        pos: torch.Tensor,    # (..., N, 2)
        w: torch.Tensor,      # scalar or (N, N)
        sigma: torch.Tensor,  # scalar or (N, N)
    ) -> torch.Tensor:        # (..., N, 2)
        """
        Computes repulsive Social Force for all agents.

        Broadcasting design:
            unsqueeze(-2): (..., N, 1, 2)  — "i" dimension
            unsqueeze(-3): (..., 1, N, 2)  — "j" dimension
        Works for both non-batched (N,2) and batched (B,N,2) inputs.
        """
        # diff[..., i, j, :] = pos[i] - pos[j]  (repulsion direction: j → i)
        diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)   # (..., N, N, 2)
        dist = diff.norm(dim=-1)                        # (..., N, N)

        N = pos.shape[-2]
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        in_range = (dist < self.r_thresh) & ~eye        # (..., N, N)

        safe_dist = dist.clamp(min=1e-6)
        unit_vec = diff / safe_dist.unsqueeze(-1)       # (..., N, N, 2)

        # Repulsive force magnitude: f_ij = w * exp(-r_ij / sigma)
        force_mag = w * torch.exp(-safe_dist / sigma)   # (..., N, N)
        force_mag = force_mag * in_range.float()

        # F_i = Σ_j  f_ij * (xi - xj) / r_ij
        force = (force_mag.unsqueeze(-1) * unit_vec).sum(dim=-2)  # (..., N, 2)
        return force


# ---------------------------------------------------------------------------


class SocialForceODELayer(nn.Module):
    """
    Integrates the Social Force ODE over a trajectory window.

    Wraps torchdiffeq.odeint so that the entire layer is differentiable
    end-to-end, allowing gradients to flow back into SocialForceFunc
    (and, in Phase 2, into the GNN that produces w and sigma).
    """

    def __init__(
        self,
        n_agents: int,
        r_thresh: float = 10.0,
        method: str = "euler",
    ) -> None:
        super().__init__()
        self.ode_func = SocialForceFunc(n_agents, r_thresh)
        self.method = method

    def forward(
        self,
        y0: torch.Tensor,                        # (..., N, 4)  initial state
        t: torch.Tensor,                         # (K+1,)       integration time points
        w: Optional[torch.Tensor] = None,        # (N, N) or None
        sigma: Optional[torch.Tensor] = None,    # (N, N) or None
    ) -> torch.Tensor:                           # (K+1, ..., N, 4)
        """
        Args:
            y0   : Initial state [x, y, vx, vy] for each agent.
            t    : 1-D tensor of time points starting at 0.
                   e.g., torch.linspace(0, 1.0, 26) for K=25 steps.
            w    : Per-edge repulsion strength (Phase 2). Uses learnable
                   scalar if None.
            sigma: Per-edge influence range (Phase 2). Uses learnable
                   scalar if None.

        Returns:
            Integrated states at each time point in t.
            Shape: (len(t), ..., N, 4)
        """
        if w is not None and sigma is not None:
            self.ode_func.set_edge_coefficients(w, sigma)

        y_traj = odeint(self.ode_func, y0, t, method=self.method)

        if w is not None:
            self.ode_func.reset_edge_coefficients()

        return y_traj
