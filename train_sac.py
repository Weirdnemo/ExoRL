"""
train_sac.py — SAC baseline training for OrbitalInsertionEnv.

Trains a Soft Actor-Critic agent on planetary orbital insertion.
Supports three experimental conditions:

  fixed     : single planet (Earth by default), fastest convergence
  random    : random planets, no curriculum
  curriculum: random planets ordered easy → hard by habitability score

Usage
-----
    # Quick test (1k steps, ~30 seconds)
    python train_sac.py --mode fixed --steps 1000 --tag test

    # Earth baseline (recommended first run, ~30 min)
    python train_sac.py --mode fixed --steps 200000 --tag earth_baseline

    # Full curriculum run (overnight)
    python train_sac.py --mode curriculum --steps 500000 --tag curriculum_v1

    # Random planets (generalisation baseline)
    python train_sac.py --mode random --steps 500000 --tag random_v1

Outputs (saved to ./training_runs/<tag>/)
------------------------------------------
    config.json         — full experiment config + env fingerprints
    learning_curve.csv  — step, ep_reward, success_rate, fuel_eff, hab_bonus
    learning_curve.png  — publication-ready training figure
    model_final.zip     — trained SB3 SAC model
    model_best.zip      — best checkpoint (by eval reward)
    eval_results.json   — final evaluation over 50 episodes
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SAC training for OrbitalInsertionEnv")
parser.add_argument("--mode",    choices=["fixed","random","curriculum"],
                    default="fixed",   help="Training condition")
parser.add_argument("--planet",  default="earth",
                    help="Preset planet name (only used with --mode fixed)")
parser.add_argument("--steps",   type=int, default=200_000,
                    help="Total environment steps")
parser.add_argument("--eval-freq",    type=int, default=10_000,
                    help="Evaluate every N steps")
parser.add_argument("--eval-episodes",type=int, default=20,
                    help="Episodes per evaluation")
parser.add_argument("--tag",     default="run",
                    help="Experiment name (used for output directory)")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--lr",      type=float, default=3e-4, help="Learning rate")
parser.add_argument("--batch",   type=int,   default=256,  help="Batch size")
parser.add_argument("--buffer",  type=int,   default=100_000, help="Replay buffer size")
parser.add_argument("--hidden",  type=int,   default=256,  help="Hidden layer size")
parser.add_argument("--no-science", action="store_true",
                    help="Disable science stack (legacy 10-dim obs)")
parser.add_argument("--pretrain", default=None,
                    help="Path to BC-pretrained .zip model for warm start")
args = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.planet_io   # trigger Planet patches

from core.env import OrbitalInsertionEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Output directory ──────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name  = f"{args.tag}_{args.mode}_{timestamp}"
out_dir   = Path("training_runs") / run_name
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\nRun: {run_name}")
print(f"Output: {out_dir}/")

# ── Environment factory ───────────────────────────────────────────────────────
obs_dim = 10 if args.no_science else 18

def make_env(seed_offset=0):
    """Create a single wrapped env for training."""
    def _init():
        kwargs = dict(
            obs_dim                = obs_dim,
            use_science_atmosphere = not args.no_science,
            use_science_j2         = not args.no_science,
            attach_star            = not args.no_science,
            generator_seed         = args.seed + seed_offset,
        )
        if args.mode == "fixed":
            kwargs["planet_preset"]    = args.planet
            kwargs["randomize_planet"] = False
            kwargs["curriculum_mode"]  = False
        elif args.mode == "random":
            # Cycle all 5 preset planets + random to prevent catastrophic forgetting
            all_presets = ["earth", "mars", "venus", "moon", "titan"]
            kwargs["planet_preset"]    = all_presets[seed_offset % len(all_presets)]
            kwargs["randomize_planet"] = (seed_offset % 2 == 1)
            kwargs["curriculum_mode"]  = False
        else:  # curriculum
            kwargs["randomize_planet"]      = True
            kwargs["curriculum_mode"]       = True
            kwargs["curriculum_pool_size"]  = 300
            kwargs["curriculum_easy_first"] = True

        env = OrbitalInsertionEnv(**kwargs)
        env = Monitor(env)
        return env
    return _init

train_env = DummyVecEnv([make_env(0)])
eval_env  = DummyVecEnv([make_env(1)])

# ── Logging callback ──────────────────────────────────────────────────────────
class MetricsCallback(BaseCallback):
    """
    Logs per-episode metrics to CSV:
      step, ep_reward, ep_length, success, fuel_eff, hab_bonus, planet_hab
    """

    def __init__(self, out_dir: Path, eval_freq: int, eval_episodes: int,
                 eval_env, verbose=1):
        super().__init__(verbose)
        self.out_dir       = out_dir
        self.eval_freq     = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_env      = eval_env

        self.csv_path = out_dir / "learning_curve.csv"
        self._csv = open(self.csv_path, "w")
        self._csv.write(
            "step,ep_reward_mean,ep_reward_std,success_rate,"
            "fuel_efficiency_mean,hab_bonus_mean,planet_hab_mean\n"
        )
        self._last_eval = 0
        self.eval_history = []   # (step, mean_r, success_rate)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval) >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        """Run deterministic evaluation and log results."""
        rewards, successes, fuel_effs, hab_bonuses, planet_habs = [], [], [], [], []

        for _ in range(self.eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_r = 0.0
            ep_info = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                ep_r += float(reward[0])
                if info[0]:
                    ep_info = info[0]
            rewards.append(ep_r)
            successes.append(float(ep_info.get("success", False)))
            fuel_effs.append(float(ep_info.get("fuel_fraction", 0.5)))
            hab_bonuses.append(float(ep_info.get("hab_bonus", 0.0)))
            planet_habs.append(float(ep_info.get("habitability", 0.0)))

        mean_r   = float(np.mean(rewards))
        std_r    = float(np.std(rewards))
        succ_r   = float(np.mean(successes))
        fuel_eff = float(np.mean(fuel_effs))
        hab_b    = float(np.mean(hab_bonuses))
        p_hab    = float(np.mean(planet_habs))

        self.eval_history.append((self.num_timesteps, mean_r, succ_r))

        line = (f"{self.num_timesteps},{mean_r:.4f},{std_r:.4f},"
                f"{succ_r:.4f},{fuel_eff:.4f},{hab_b:.4f},{p_hab:.4f}\n")
        self._csv.write(line)
        self._csv.flush()

        elapsed = time.time() - self._start_time
        fps     = self.num_timesteps / max(elapsed, 1)
        eta_s   = (self._total_steps - self.num_timesteps) / max(fps, 1)

        print(f"  step={self.num_timesteps:>8d}  "
              f"reward={mean_r:>8.1f}  "
              f"success={succ_r:>5.1%}  "
              f"fuel={fuel_eff:>5.2f}  "
              f"hab={p_hab:>5.3f}  "
              f"fps={fps:>5.0f}  "
              f"eta={eta_s/60:>5.1f}min")

    def _on_training_start(self):
        self._start_time  = time.time()
        self._total_steps = self.locals.get("total_timesteps", args.steps)
        print(f"\n{'─'*80}")
        print(f"  {'step':>8s}  {'reward':>8s}  {'success':>7s}  "
              f"{'fuel':>5s}  {'hab':>5s}  {'fps':>5s}  {'eta':>8s}")
        print(f"{'─'*80}")

    def _on_training_end(self):
        self._csv.close()
        total_time = time.time() - self._start_time
        print(f"{'─'*80}")
        print(f"  Training complete: {total_time:.0f}s  "
              f"({self.num_timesteps/total_time:.0f} fps avg)")


# ── SAC hyperparameters ───────────────────────────────────────────────────────
policy_kwargs = dict(
    net_arch = [args.hidden, args.hidden],   # 2-layer MLP
)

model = SAC(
    policy         = "MlpPolicy",
    env            = train_env,
    learning_rate  = args.lr,
    buffer_size    = args.buffer,
    batch_size     = args.batch,
    gamma          = 0.99,
    tau            = 0.005,
    ent_coef       = "auto",
    learning_starts= 2_000,
    train_freq     = 1,
    gradient_steps = 1,
    policy_kwargs  = policy_kwargs,
    verbose        = 0,
    seed           = args.seed,
)

n_params = sum(p.numel() for p in model.policy.parameters())
print(f"SAC policy: {n_params:,} parameters  hidden={args.hidden}×2  obs={obs_dim}  act=3")

# Warm start: load BC-pretrained actor weights into SAC policy
if args.pretrain and os.path.exists(args.pretrain):
    import torch
    print(f"Loading BC pretrained weights: {args.pretrain}")
    bc_model = SAC.load(args.pretrain)
    try:
        with torch.no_grad():
            dst = model.policy.actor
            src_actor = bc_model.policy.actor
            dst.latent_pi[0].weight.copy_(src_actor.latent_pi[0].weight)
            dst.latent_pi[0].bias.copy_(  src_actor.latent_pi[0].bias)
            dst.latent_pi[2].weight.copy_(src_actor.latent_pi[2].weight)
            dst.latent_pi[2].bias.copy_(  src_actor.latent_pi[2].bias)
            dst.mu.weight.copy_(           src_actor.mu.weight)
            dst.mu.bias.copy_(             src_actor.mu.bias)
        print("  ✓ BC actor weights loaded — warm start active")
    except Exception as e:
        print(f"  ✗ BC weight transfer failed: {e}  — training from scratch")
elif args.pretrain:
    print(f"  Warning: --pretrain path not found: {args.pretrain}")
print(f"Training:   {args.steps:,} steps  eval_freq={args.eval_freq:,}  seed={args.seed}")
print(f"Mode:       {args.mode}" +
      (f"  planet={args.planet}" if args.mode == "fixed" else ""))

# ── Save config ───────────────────────────────────────────────────────────────
config = {
    "run_name":      run_name,
    "timestamp":     timestamp,
    "mode":          args.mode,
    "planet":        args.planet if args.mode == "fixed" else None,
    "total_steps":   args.steps,
    "eval_freq":     args.eval_freq,
    "eval_episodes": args.eval_episodes,
    "seed":          args.seed,
    "obs_dim":       obs_dim,
    "science_stack": not args.no_science,
    "lr":            args.lr,
    "batch_size":    args.batch,
    "buffer_size":   args.buffer,
    "hidden_size":   args.hidden,
    "n_parameters":  n_params,
    "sb3_version":   __import__("stable_baselines3").__version__,
}
with open(out_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# ── Best model callback ───────────────────────────────────────────────────────
best_cb = EvalCallback(
    eval_env,
    best_model_save_path = str(out_dir),
    log_path             = str(out_dir),
    eval_freq            = args.eval_freq,
    n_eval_episodes      = args.eval_episodes,
    deterministic        = True,
    render               = False,
    verbose              = 0,
)
best_cb.best_model_save_path = str(out_dir / "model_best")

metrics_cb = MetricsCallback(
    out_dir       = out_dir,
    eval_freq     = args.eval_freq,
    eval_episodes = args.eval_episodes,
    eval_env      = eval_env,
)

# ── Train ─────────────────────────────────────────────────────────────────────
t0 = time.time()
model.learn(
    total_timesteps = args.steps,
    callback        = [metrics_cb],
    progress_bar    = False,
    reset_num_timesteps = True,
)
train_time = time.time() - t0

# ── Save final model ──────────────────────────────────────────────────────────
model.save(str(out_dir / "model_final"))
print(f"\nSaved: {out_dir}/model_final.zip")

# ── Final evaluation ──────────────────────────────────────────────────────────
print(f"\nRunning final evaluation (50 episodes)...")
final_rewards, final_successes, final_fuels, final_habs = [], [], [], []
eval_env_single = make_env(99)()

for ep in range(50):
    obs, _ = eval_env_single.reset()
    done = trunc = False
    ep_r = 0.0
    info = {}
    while not (done or trunc):
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, rew, done, trunc, info = eval_env_single.step(action[0])
        ep_r += rew
    final_rewards.append(ep_r)
    final_successes.append(float(info.get("success", False)))
    final_fuels.append(float(info.get("fuel_fraction", 0.5)))
    final_habs.append(float(info.get("habitability", 0.0)))

eval_results = {
    "n_episodes":         50,
    "mean_reward":        float(np.mean(final_rewards)),
    "std_reward":         float(np.std(final_rewards)),
    "success_rate":       float(np.mean(final_successes)),
    "mean_fuel_fraction": float(np.mean(final_fuels)),
    "mean_hab_score":     float(np.mean(final_habs)),
    "train_time_s":       train_time,
    "steps_per_second":   args.steps / train_time,
}
with open(out_dir / "eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\n{'═'*60}")
print(f"  Final evaluation (50 episodes):")
print(f"  Success rate:    {eval_results['success_rate']:>6.1%}")
print(f"  Mean reward:     {eval_results['mean_reward']:>8.1f}")
print(f"  Fuel remaining:  {eval_results['mean_fuel_fraction']:>6.1%}")
print(f"  Mean hab score:  {eval_results['mean_hab_score']:>6.3f}")
print(f"  Train time:      {train_time:.0f}s  ({eval_results['steps_per_second']:.0f} fps)")
print(f"{'═'*60}")

# ── Learning curve figure ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    steps_l, rewards_l, success_l, fuel_l, hab_l = [], [], [], [], []
    with open(out_dir / "learning_curve.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps_l.append(int(row["step"]))
            rewards_l.append(float(row["ep_reward_mean"]))
            success_l.append(float(row["success_rate"]) * 100)
            fuel_l.append(float(row["fuel_efficiency_mean"]))
            hab_l.append(float(row["planet_hab_mean"]))

    if len(steps_l) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.14,
                            wspace=0.35)

        BLUE   = "#0072B2"
        GREEN  = "#009E73"
        ORANGE = "#E69F00"

        for ax in axes:
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="#eeeeee", lw=0.8, zorder=0)
            ax.tick_params(labelsize=9)

        axes[0].plot(steps_l, rewards_l, color=BLUE, lw=1.5)
        axes[0].set_xlabel("Environment steps", fontsize=10)
        axes[0].set_ylabel("Mean episode reward", fontsize=10)
        axes[0].set_title("Episode reward", fontsize=10, fontweight="bold")

        axes[1].plot(steps_l, success_l, color=GREEN, lw=1.5)
        axes[1].set_xlabel("Environment steps", fontsize=10)
        axes[1].set_ylabel("Success rate (%)", fontsize=10)
        axes[1].set_ylim(0, 105)
        axes[1].set_title("Insertion success rate", fontsize=10, fontweight="bold")

        axes[2].plot(steps_l, fuel_l, color=ORANGE, lw=1.5)
        axes[2].set_xlabel("Environment steps", fontsize=10)
        axes[2].set_ylabel("Fuel remaining (fraction)", fontsize=10)
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title("Fuel efficiency", fontsize=10, fontweight="bold")

        mode_label = f"mode={args.mode}" + (f" planet={args.planet}" if args.mode=="fixed" else "")
        fig.suptitle(
            f"SAC training — OrbitalInsertionEnv  [{mode_label}]  "
            f"final success={eval_results['success_rate']:.1%}",
            fontsize=10, fontweight="bold"
        )

        fig.savefig(str(out_dir / "learning_curve.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir}/learning_curve.png")
except Exception as e:
    print(f"Figure generation skipped: {e}")

print(f"\nAll outputs in: {out_dir}/")
print("Done.")
