"""
Phase 1 toy data verification script.

3 シナリオで SocialForceODELayer の動作を確認する。

Scenario A: 反発の可視化
    2 エージェントを近くに配置し、斥力で離れていく軌跡を描画する。

Scenario B: 学習による係数回復
    既知の (w_true, sigma_true) で生成した軌跡を教師データとし、
    ランダム初期値からの勾配降下で係数を回復できるか検証する。
    → autograd が正しく機能することの実証。

Scenario C: 三角配置の平衡確認
    3 エージェントを正三角形に配置し、対称性から合力がゼロになることを確認する。

使い方:
    python scripts/verify_phase1.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

try:
    import matplotlib.pyplot as plt
    PLOT = True
except ImportError:
    PLOT = False
    print("[warning] matplotlib not found. Skipping plots.")


from src.models.social_force_ode import SocialForceODELayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_time(n_steps: int, dt: float = 0.04) -> torch.Tensor:
    return torch.linspace(0.0, dt * n_steps, n_steps + 1)


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Scenario A: 反発の可視化
# ---------------------------------------------------------------------------

def scenario_a():
    print_section("Scenario A: 2 エージェントの反発軌跡")

    layer = SocialForceODELayer(n_agents=2, r_thresh=10.0, method="euler")

    # w=2.0, sigma=1.5 に設定
    with torch.no_grad():
        layer.ode_func.log_w.copy_(torch.log(torch.tensor(2.0)))
        layer.ode_func.log_sigma.copy_(torch.log(torch.tensor(1.5)))

    # 初期状態: 2 エージェントが 1 m 離れて静止
    y0 = torch.tensor([
        [-0.5, 0.0, 0.0, 0.0],
        [ 0.5, 0.0, 0.0, 0.0],
    ])
    t = make_time(n_steps=50, dt=0.04)   # 2 秒間

    with torch.no_grad():
        y_traj = layer(y0, t)   # (51, 2, 4)

    pos = y_traj[:, :, :2]   # (51, 2, 2)
    dist = (pos[:, 0, :] - pos[:, 1, :]).norm(dim=-1)   # (51,)

    print(f"  初期エージェント間距離:  {dist[0].item():.4f} m")
    print(f"  最終エージェント間距離:  {dist[-1].item():.4f} m")
    print(f"  離れた → {'OK' if dist[-1] > dist[0] else 'FAIL'}")

    if PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for i, color in enumerate(["tab:blue", "tab:orange"]):
            ax.plot(pos[:, i, 0].numpy(), pos[:, i, 1].numpy(),
                    color=color, label=f"Agent {i}")
            ax.plot(*pos[0, i].numpy(), "o", color=color)
            ax.plot(*pos[-1, i].numpy(), "s", color=color)
        ax.set_title("Scenario A: Trajectories (o=start, s=end)")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(); ax.set_aspect("equal"); ax.grid(True)

        ax = axes[1]
        ax.plot(t.numpy(), dist.numpy(), color="tab:green")
        ax.set_title("Inter-agent distance over time")
        ax.set_xlabel("t [s]"); ax.set_ylabel("distance [m]")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("scenario_a.png", dpi=100)
        print("  [saved] scenario_a.png")
        plt.close()


# ---------------------------------------------------------------------------
# Scenario B: 勾配降下による係数回復
# ---------------------------------------------------------------------------

def scenario_b():
    print_section("Scenario B: 勾配降下で (w, sigma) を回復")

    torch.manual_seed(42)

    # 教師データ生成用モデル (w=3.0, sigma=2.0 で固定)
    teacher = SocialForceODELayer(n_agents=3, r_thresh=10.0, method="euler")
    with torch.no_grad():
        teacher.ode_func.log_w.copy_(torch.log(torch.tensor(3.0)))
        teacher.ode_func.log_sigma.copy_(torch.log(torch.tensor(2.0)))

    y0 = torch.tensor([
        [0.0,  0.0, 0.0, 0.0],
        [1.5,  0.0, 0.0, 0.0],
        [0.75, 1.3, 0.0, 0.0],
    ])
    t = make_time(n_steps=20, dt=0.04)

    with torch.no_grad():
        y_true = teacher(y0, t)   # (21, 3, 4)  教師軌跡

    # 学習モデル: ランダム初期値
    student = SocialForceODELayer(n_agents=3, r_thresh=10.0, method="euler")
    optimizer = optim.Adam(student.parameters(), lr=0.05)

    print(f"  True   log_w={teacher.ode_func.log_w.item():.4f}  "
          f"log_sigma={teacher.ode_func.log_sigma.item():.4f}")
    print(f"  Init   log_w={student.ode_func.log_w.item():.4f}  "
          f"log_sigma={student.ode_func.log_sigma.item():.4f}")
    print()

    losses = []
    log_ws = []
    log_sigmas = []

    for step in range(200):
        optimizer.zero_grad()
        y_pred = student(y0, t)
        loss = (y_pred[..., :2] - y_true[..., :2]).pow(2).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        log_ws.append(student.ode_func.log_w.item())
        log_sigmas.append(student.ode_func.log_sigma.item())

        if step % 40 == 0 or step == 199:
            print(f"  step {step:3d}  loss={loss.item():.6f}  "
                  f"log_w={student.ode_func.log_w.item():.4f}  "
                  f"log_sigma={student.ode_func.log_sigma.item():.4f}")

    final_w = torch.exp(student.ode_func.log_w).item()
    final_sigma = torch.exp(student.ode_func.log_sigma).item()
    print(f"\n  Learned  w={final_w:.4f} (true=3.0)  sigma={final_sigma:.4f} (true=2.0)")

    w_err = abs(final_w - 3.0) / 3.0
    s_err = abs(final_sigma - 2.0) / 2.0
    print(f"  Relative error:  w={w_err*100:.1f}%  sigma={s_err*100:.1f}%")
    print(f"  Autograd through odeint: {'OK' if w_err < 0.2 else 'check'}")

    if PLOT:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].semilogy(losses)
        axes[0].set_title("Training loss"); axes[0].set_xlabel("step")
        axes[0].grid(True)

        axes[1].axhline(teacher.ode_func.log_w.item(), color="red",
                        linestyle="--", label="true log_w")
        axes[1].plot(log_ws, label="learned log_w")
        axes[1].set_title("log_w convergence"); axes[1].set_xlabel("step")
        axes[1].legend(); axes[1].grid(True)

        axes[2].axhline(teacher.ode_func.log_sigma.item(), color="red",
                        linestyle="--", label="true log_sigma")
        axes[2].plot(log_sigmas, label="learned log_sigma")
        axes[2].set_title("log_sigma convergence"); axes[2].set_xlabel("step")
        axes[2].legend(); axes[2].grid(True)

        plt.tight_layout()
        plt.savefig("scenario_b.png", dpi=100)
        print("  [saved] scenario_b.png")
        plt.close()


# ---------------------------------------------------------------------------
# Scenario C: 正三角形の対称平衡
# ---------------------------------------------------------------------------

def scenario_c():
    print_section("Scenario C: 正三角形配置での合力対称性")

    layer = SocialForceODELayer(n_agents=3, r_thresh=10.0, method="euler")
    with torch.no_grad():
        layer.ode_func.log_w.copy_(torch.log(torch.tensor(1.0)))
        layer.ode_func.log_sigma.copy_(torch.log(torch.tensor(1.0)))

    # 正三角形の頂点（一辺 2 m）
    r = 2.0 / (3 ** 0.5)   # 外接円半径
    import math
    y0 = torch.tensor([
        [r * math.cos(math.radians(90)),   r * math.sin(math.radians(90)),  0.0, 0.0],
        [r * math.cos(math.radians(210)),  r * math.sin(math.radians(210)), 0.0, 0.0],
        [r * math.cos(math.radians(330)),  r * math.sin(math.radians(330)), 0.0, 0.0],
    ])

    t_scalar = torch.tensor(0.0)
    with torch.no_grad():
        dydt = layer.ode_func(t_scalar, y0)

    # 対称性より各エージェントへの合力は中心方向（斥力なので外向き）
    acc = dydt[:, 2:]   # (3, 2)
    force_norms = acc.norm(dim=-1)

    print(f"  各エージェントの加速度ノルム: {force_norms.tolist()}")
    print(f"  3 エージェントの加速度ノルムが等しい → "
          f"{'OK' if force_norms.std() < 1e-5 else 'FAIL'}")

    # 合力の x, y 成分の合計はゼロ（運動量保存）
    total_force = acc.sum(dim=0)
    print(f"  全合力 (運動量保存): {total_force.tolist()}")
    print(f"  全合力がゼロ → {'OK' if total_force.norm() < 1e-5 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GNN-ODE-SFM Phase 1: 動作検証スクリプト")
    print(f"PyTorch version: {torch.__version__}")

    try:
        import torchdiffeq
        print(f"torchdiffeq version: {torchdiffeq.__version__}")
    except ImportError:
        print("[ERROR] torchdiffeq がインストールされていません。")
        print("        pip install torchdiffeq を実行してください。")
        sys.exit(1)

    scenario_a()
    scenario_b()
    scenario_c()

    print("\n" + "="*60)
    print("  Phase 1 verification 完了")
    print("="*60)
