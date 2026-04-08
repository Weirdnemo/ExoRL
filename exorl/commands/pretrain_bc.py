from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import exorl.core.planet_io


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavioural cloning pretraining")
    parser.add_argument(
        "--demos", required=True, help="Path to .npz demo file from generate_demos.py"
    )
    parser.add_argument("--out", default="bc_model/", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of data held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-dim", type=int, default=18)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Defer torch/SB3 imports so the module remains importable without `[rl]`.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split

    from stable_baselines3 import SAC

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading demonstrations: {args.demos}")
    data = np.load(args.demos, allow_pickle=True)
    obs_all = torch.tensor(data["observations"], dtype=torch.float32)
    act_all = torch.tensor(data["actions"], dtype=torch.float32)
    N = len(obs_all)

    print(f"  {N:,} (obs, action) pairs")
    print(f"  obs:  {tuple(obs_all.shape)}")
    print(f"  act:  {tuple(act_all.shape)}")
    print(f"  device: {device}\n")

    n_val = max(1, int(N * args.val_frac))
    n_train = N - n_val
    dataset = TensorDataset(obs_all, act_all)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    print(f"  Train: {n_train:,}  Val: {n_val:,}  Batches/epoch: {len(train_loader)}\n")

    class BCPolicy(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int, hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, act_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)

    model = BCPolicy(args.obs_dim, 3, args.hidden).to(device)
    optim_ = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            tl += float(loss.item())
        tl /= max(len(train_loader), 1)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vl += float(loss_fn(pred, yb).item())
        vl /= max(len(val_loader), 1)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), str(Path(args.out) / "bc_policy_best.pt"))

        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(f"epoch={ep+1:>4d}/{args.epochs}  train={tl:.6f}  val={vl:.6f}")

    torch.save(model.state_dict(), str(Path(args.out) / "bc_policy_final.pt"))
    with open(Path(args.out) / "bc_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Export as SB3-compatible zip by creating a SAC model and copying the actor.
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv
    from exorl.core.env import OrbitalInsertionEnv

    def _make():
        return OrbitalInsertionEnv(
            obs_dim=args.obs_dim,
            use_science_atmosphere=args.obs_dim >= 18,
            use_science_j2=args.obs_dim >= 18,
            attach_star=args.obs_dim >= 18,
        )

    env = DummyVecEnv([_make])
    sac = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1,
        batch_size=1,
        policy_kwargs=dict(net_arch=[args.hidden, args.hidden]),
        verbose=0,
        seed=args.seed,
    )

    # Match the SAC actor MLP layers (latent_pi) and mu head
    with torch.no_grad():
        actor = sac.policy.actor
        # First two layers correspond to latent_pi[0] and latent_pi[2]
        actor.latent_pi[0].weight.copy_(model.net[0].weight)
        actor.latent_pi[0].bias.copy_(model.net[0].bias)
        actor.latent_pi[2].weight.copy_(model.net[2].weight)
        actor.latent_pi[2].bias.copy_(model.net[2].bias)
        actor.mu.weight.copy_(model.net[4].weight)
        actor.mu.bias.copy_(model.net[4].bias)

    out_zip = Path(args.out) / "bc_policy"
    sac.save(str(out_zip))

    dt = time.time() - t0
    print(f"\nSaved: {out_zip}.zip")
    print(f"Done in {dt:.1f}s")


if __name__ == "__main__":
    main()

