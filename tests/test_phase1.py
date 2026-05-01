"""
Phase 1 unit tests.

1. test_forward_shape          : output shape matches input
2. test_autograd_through_odeint: gradients flow back through torchdiffeq
3. test_repulsion_physics      : agents move apart under repulsive force
4. test_external_coefficients  : set_edge_coefficients() overrides scalars
5. test_no_self_interaction    : single agent produces zero force
6. test_beyond_threshold       : agents outside r_thresh produce no force

Run with: pytest tests/test_phase1.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.models.social_force_ode import SocialForceFunc, SocialForceODELayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_AGENTS = 4
DT = 0.04          # 25 Hz
N_STEPS = 10
R_THRESH = 5.0


@pytest.fixture
def func():
    return SocialForceFunc(n_agents=N_AGENTS, r_thresh=R_THRESH)


@pytest.fixture
def layer():
    return SocialForceODELayer(n_agents=N_AGENTS, r_thresh=R_THRESH, method="euler")


@pytest.fixture
def t():
    return torch.linspace(0.0, DT * N_STEPS, N_STEPS + 1)


@pytest.fixture
def y0():
    torch.manual_seed(0)
    return torch.randn(N_AGENTS, 4) * 2.0


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

def test_forward_shape_single(func, y0):
    """SocialForceFunc preserves state shape."""
    t_scalar = torch.tensor(0.0)
    dydt = func(t_scalar, y0)
    assert dydt.shape == y0.shape, f"Expected {y0.shape}, got {dydt.shape}"


def test_odeint_output_shape(layer, y0, t):
    """SocialForceODELayer output has shape (len(t), N, 4)."""
    y_traj = layer(y0, t)
    assert y_traj.shape == (len(t), N_AGENTS, 4), \
        f"Expected ({len(t)}, {N_AGENTS}, 4), got {y_traj.shape}"


def test_odeint_output_shape_batched(layer, t):
    """ODE layer handles batched initial states (B, N, 4)."""
    B = 8
    y0_batch = torch.randn(B, N_AGENTS, 4)
    y_traj = layer(y0_batch, t)
    assert y_traj.shape == (len(t), B, N_AGENTS, 4), \
        f"Expected ({len(t)}, {B}, {N_AGENTS}, 4), got {y_traj.shape}"


# ---------------------------------------------------------------------------
# 2. Autograd through odeint
# ---------------------------------------------------------------------------

def test_autograd_log_w(layer, y0, t):
    """Gradients flow back to log_w through the ODE solver."""
    y_traj = layer(y0, t)
    # Dummy loss: MSE of final positions
    loss = y_traj[-1, :, :2].pow(2).mean()
    loss.backward()
    assert layer.ode_func.log_w.grad is not None, "No gradient for log_w"
    assert not torch.isnan(layer.ode_func.log_w.grad), "NaN gradient for log_w"


def test_autograd_log_sigma(layer, y0, t):
    """Gradients flow back to log_sigma through the ODE solver."""
    y_traj = layer(y0, t)
    loss = y_traj[-1, :, :2].pow(2).mean()
    loss.backward()
    assert layer.ode_func.log_sigma.grad is not None, "No gradient for log_sigma"
    assert not torch.isnan(layer.ode_func.log_sigma.grad), "NaN gradient for log_sigma"


def test_autograd_external_coefficients(layer, y0, t):
    """Gradients flow back to externally supplied w and sigma tensors."""
    w = torch.ones(N_AGENTS, N_AGENTS, requires_grad=True)
    sigma = torch.ones(N_AGENTS, N_AGENTS, requires_grad=True)

    y_traj = layer(y0, t, w=w, sigma=sigma)
    loss = y_traj[-1, :, :2].pow(2).mean()
    loss.backward()

    assert w.grad is not None, "No gradient for external w"
    assert sigma.grad is not None, "No gradient for external sigma"


# ---------------------------------------------------------------------------
# 3. Physics: repulsion pushes agents apart
# ---------------------------------------------------------------------------

def test_repulsion_increases_distance():
    """
    Two agents starting 1 m apart should move further apart under repulsion.
    Uses a large w to make the effect clearly visible within a few steps.
    """
    func = SocialForceFunc(n_agents=2, r_thresh=10.0)
    # Large w for strong repulsion, small sigma so force decays slowly
    func.log_w.data = torch.log(torch.tensor(5.0))
    func.log_sigma.data = torch.log(torch.tensor(2.0))

    # Agents at x = -0.5 and x = +0.5, stationary
    y0 = torch.tensor([
        [-0.5, 0.0, 0.0, 0.0],   # agent 0: pos (-0.5, 0), vel (0, 0)
        [ 0.5, 0.0, 0.0, 0.0],   # agent 1: pos ( 0.5, 0), vel (0, 0)
    ])

    layer = SocialForceODELayer(n_agents=2, r_thresh=10.0, method="euler")
    layer.ode_func = func

    t = torch.linspace(0.0, 0.5, 26)   # 0.5 s at 50 steps
    with torch.no_grad():
        y_traj = layer(y0, t)

    dist_init = (y0[0, :2] - y0[1, :2]).norm().item()
    dist_final = (y_traj[-1, 0, :2] - y_traj[-1, 1, :2]).norm().item()

    assert dist_final > dist_init, \
        f"Agents should repel: init dist={dist_init:.4f}, final dist={dist_final:.4f}"


def test_repulsion_direction():
    """
    Agent 0 at origin, Agent 1 at (2, 0).
    Repulsive force on agent 0 should point in the -x direction.
    """
    func = SocialForceFunc(n_agents=2, r_thresh=10.0)
    func.log_w.data = torch.log(torch.tensor(1.0))
    func.log_sigma.data = torch.log(torch.tensor(1.0))

    y = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],   # agent 0 at origin
        [2.0, 0.0, 0.0, 0.0],   # agent 1 at (2, 0)
    ])
    t_scalar = torch.tensor(0.0)

    with torch.no_grad():
        dydt = func(t_scalar, y)

    # Force on agent 0 from agent 1 is in direction (0-2, 0-0) = (-2, 0) → -x
    ax0 = dydt[0, 2].item()   # x-acceleration of agent 0
    assert ax0 < 0.0, f"Agent 0 should accelerate in -x, got ax0={ax0:.6f}"


# ---------------------------------------------------------------------------
# 4. External coefficients override scalars
# ---------------------------------------------------------------------------

def test_external_coefficients_override(func, y0):
    """set_edge_coefficients() changes the force computation."""
    t_scalar = torch.tensor(0.0)

    with torch.no_grad():
        dydt_default = func(t_scalar, y0.clone())

    # External: zero w → zero force → dydt[vel_part] = 0
    w_zero = torch.zeros(N_AGENTS, N_AGENTS)
    sigma_one = torch.ones(N_AGENTS, N_AGENTS)
    func.set_edge_coefficients(w_zero, sigma_one)

    with torch.no_grad():
        dydt_zero_w = func(t_scalar, y0.clone())

    func.reset_edge_coefficients()

    force_with_zero_w = dydt_zero_w[:, 2:]   # acceleration part
    assert force_with_zero_w.abs().max().item() < 1e-6, \
        "Zero w should produce zero force"


# ---------------------------------------------------------------------------
# 5. Single agent: zero force
# ---------------------------------------------------------------------------

def test_no_self_interaction():
    """A lone agent has no interaction partners → zero acceleration."""
    func = SocialForceFunc(n_agents=1, r_thresh=10.0)
    y = torch.tensor([[1.0, 2.0, 0.5, -0.3]])   # (1, 4)
    t_scalar = torch.tensor(0.0)

    with torch.no_grad():
        dydt = func(t_scalar, y)

    # d(pos)/dt = vel ✓, d(vel)/dt = 0 (no neighbors)
    assert torch.allclose(dydt[0, :2], y[0, 2:]), "dx/dt should equal velocity"
    assert torch.allclose(dydt[0, 2:], torch.zeros(2)), "No force for single agent"


# ---------------------------------------------------------------------------
# 6. Beyond threshold: no force
# ---------------------------------------------------------------------------

def test_beyond_threshold():
    """Agents farther apart than r_thresh should produce zero force on each other."""
    r_thresh = 3.0
    func = SocialForceFunc(n_agents=2, r_thresh=r_thresh)
    func.log_w.data = torch.log(torch.tensor(10.0))

    # Agents 5 m apart (> r_thresh=3)
    y = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
    ])
    t_scalar = torch.tensor(0.0)

    with torch.no_grad():
        dydt = func(t_scalar, y)

    acc = dydt[:, 2:]
    assert acc.abs().max().item() < 1e-6, \
        "Agents beyond r_thresh should have zero acceleration"
