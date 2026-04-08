"""
generate_demos.py — Expert demonstration generator for imitation learning.

Runs the periapsis-timed controller across all preset planets and random
planets, collecting (observation, action) pairs as training data for
Behavioural Cloning (BC).

The expert controller achieves:
  Earth / Mars / Venus / Moon / Titan : 100% success
  Random planets                      : ~88% success

Usage
-----
    # Quick dataset (100 episodes, ~2 min)
    python generate_demos.py --episodes 100 --out demos/demos_100.npz

    # Full BC pretraining dataset (1000 episodes, ~20 min)
    python generate_demos.py --episodes 1000 --out demos/demos_1000.npz

    # Only preset planets
    python generate_demos.py --episodes 200 --presets-only --out demos/presets.npz

Output format (compressed .npz)
--------------------------------
    observations : float32  (N, 18)
    actions      : float32  (N, 3)
    episode_ids  : int32    (N,)      episode index for each step
    planet_names : str      (E,)      planet name per episode
    successes    : bool     (E,)      success flag per episode
    rewards      : float32  (E,)      total reward per episode
    metadata     : dict               generation config
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Generate expert demonstrations")
parser.add_argument(
    "--episodes", type=int, default=500, help="Number of episodes to collect"
)
parser.add_argument("--out", default="demos/demos.npz", help="Output .npz file path")
parser.add_argument(
    "--presets-only",
    action="store_true",
    help="Only use the 5 preset planets (no random)",
)
parser.add_argument("--obs-dim", type=int, default=18)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--max-steps",
    type=int,
    default=5000,
    help="Max steps per demo episode (higher = more success on big planets)",
)
parser.add_argument(
    "--speed-ratio",
    type=float,
    default=1.08,
    help="Initial speed as fraction of circular orbit speed",
)
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planet_rl.core.planet_io
from planet_rl.core.env import OrbitalInsertionEnv

# ── Expert controller ──────────────────────────────────────────────────────────


def periapsis_controller(env) -> np.ndarray:
    """
    Burn retrograde ONLY when near periapsis.

    Physical basis: continuous retrograde burn while rising lowers periapsis
    toward the surface (crash). You must coast to the next periapsis pass and
    only burn there. This is the standard multi-pass orbital insertion strategy.

    Near-periapsis criterion:
      - |v_radial| < window  (moving nearly tangentially)
      - v > v_circ           (faster than circular = at periapsis, not apoapsis)
      - window = max(200, v_circ × 0.03)  (adaptive to planet size)
    """
    state = env.state
    planet = env.planet
    mu = 6.674e-11 * planet.mass
    pos = state.position
    vel = state.velocity
    r = float(np.linalg.norm(pos))
    v = float(np.linalg.norm(vel))
    r_hat = pos / r
    v_r = float(np.dot(vel, r_hat))
    v_circ = math.sqrt(mu / r)

    window = max(200.0, v_circ * 0.03)
    near_peri = abs(v_r) < window and v > v_circ

    if near_peri and v > v_circ + 20:
        return np.array([1.0, 0.0, 1.0], dtype=np.float32)  # retrograde
    return np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # coast


# ── Planet schedule ────────────────────────────────────────────────────────────

PRESETS = ["earth", "mars", "venus", "moon", "titan"]


def make_env(preset=None, seed=0):
    """Create one episode environment."""
    randomize = preset is None
    return OrbitalInsertionEnv(
        obs_dim=args.obs_dim,
        planet_preset=preset if not randomize else "earth",
        randomize_planet=randomize,
        generator_seed=seed,
        initial_altitude=0,
        initial_speed_ratio=args.speed_ratio,
        max_steps=args.max_steps,
    )


def planet_schedule(n_episodes: int, presets_only: bool):
    """
    Yield (preset_or_None, seed) for each episode.
    Mix: 50% preset planets (10% each), 50% random if not presets_only.
    """
    rng = np.random.RandomState(args.seed)
    for ep in range(n_episodes):
        if presets_only:
            yield PRESETS[ep % len(PRESETS)], ep
        else:
            if ep % 2 == 0:
                yield PRESETS[ep // 2 % len(PRESETS)], ep
            else:
                yield None, rng.randint(0, 100_000)


# ── Collection loop ────────────────────────────────────────────────────────────

Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

obs_buf = []
act_buf = []
ep_ids = []
p_names = []
successes = []
rewards = []

n_success = 0
n_total = 0
t0 = time.time()

if not args.quiet:
    print(f"Generating {args.episodes} expert demonstrations")
    print(
        f"  obs_dim={args.obs_dim}  speed_ratio={args.speed_ratio}  "
        f"max_steps={args.max_steps}"
    )
    print(f"  {'presets only' if args.presets_only else 'presets + random (50/50)'}")
    print(f"  Output: {args.out}\n")
    print(
        f"  {'ep':>5}  {'planet':<12}  {'ok':>4}  {'steps':>6}  "
        f"{'reward':>8}  {'running_%':>10}"
    )
    print("  " + "─" * 55)

for ep, (preset, seed) in enumerate(planet_schedule(args.episodes, args.presets_only)):
    env = make_env(preset, seed)
    obs, _ = env.reset()

    ep_obs = []
    ep_act = []
    ep_r = 0.0
    done = trunc = False

    while not (done or trunc):
        action = periapsis_controller(env)
        ep_obs.append(obs.copy())
        ep_act.append(action.copy())
        obs, r, done, trunc, info = env.step(action)
        ep_r += r

    success = bool(info.get("success", False))
    planet_name = info.get("planet", preset or "random")

    # Only keep successful episodes for BC training
    if success:
        for i, (o, a) in enumerate(zip(ep_obs, ep_act)):
            obs_buf.append(o)
            act_buf.append(a)
            ep_ids.append(ep)
        p_names.append(planet_name)
        successes.append(True)
        rewards.append(ep_r)
        n_success += 1
    else:
        p_names.append(planet_name)
        successes.append(False)
        rewards.append(ep_r)

    n_total += 1
    elapsed = time.time() - t0
    rate = n_total / elapsed

    if not args.quiet and (ep % 25 == 0 or ep < 5):
        pct = n_success / n_total * 100
        print(
            f"  {ep + 1:5d}  {planet_name:<12}  "
            f"{'✓' if success else '✗':>4}  {len(ep_obs):>6}  "
            f"{ep_r:>8.1f}  {pct:>9.1f}%"
        )

# ── Save dataset ───────────────────────────────────────────────────────────────

obs_arr = np.array(obs_buf, dtype=np.float32)
act_arr = np.array(act_buf, dtype=np.float32)
ep_arr = np.array(ep_ids, dtype=np.int32)
suc_arr = np.array(successes, dtype=bool)
rew_arr = np.array(rewards, dtype=np.float32)

np.savez_compressed(
    args.out,
    observations=obs_arr,
    actions=act_arr,
    episode_ids=ep_arr,
    planet_names=np.array(p_names),
    successes=suc_arr,
    rewards=rew_arr,
)

elapsed = time.time() - t0
if not args.quiet:
    print(f"\n{'─' * 57}")
    print(
        f"  Episodes:   {n_total}  ({n_success} successful, "
        f"{n_success / n_total * 100:.0f}%)"
    )
    print(f"  Pairs (obs,act): {len(obs_arr):,}  (from successful episodes only)")
    print(f"  obs shape:  {obs_arr.shape}")
    print(f"  act shape:  {act_arr.shape}")
    print(f"  Time:       {elapsed:.0f}s  ({n_total / elapsed:.1f} eps/s)")
    print(f"  Saved:      {args.out}  ({os.path.getsize(args.out) // 1024} kB)")
