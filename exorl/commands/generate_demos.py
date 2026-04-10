from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import exorl.core.planet_io
from exorl.core.env import OrbitalInsertionEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate expert demonstrations")
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--out", default="demos/demos.npz", help="Output .npz file path"
    )
    parser.add_argument(
        "--presets-only",
        action="store_true",
        help="Only use the 5 preset planets (no random)",
    )
    parser.add_argument("--obs-dim", type=int, default=18)
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Fast demo mode: obs_dim=10 + simplified planets + no science stack",
    )
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
    return parser


def periapsis_controller(env) -> np.ndarray:
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


PRESETS = ["earth", "mars", "venus", "moon", "titan"]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    def make_env(preset=None, seed=0):
        randomize = preset is None
        return OrbitalInsertionEnv(
            lite_mode=bool(args.lite),
            obs_dim=10 if args.lite else args.obs_dim,
            use_science_atmosphere=not args.lite,
            use_science_j2=not args.lite,
            attach_star=not args.lite,
            planet_preset=preset if not randomize else "earth",
            randomize_planet=randomize,
            generator_seed=seed,
            initial_altitude=0,
            initial_speed_ratio=args.speed_ratio,
            max_steps=args.max_steps,
        )

    def planet_schedule(n_episodes: int, presets_only: bool):
        rng = np.random.RandomState(args.seed)
        for ep in range(n_episodes):
            if presets_only:
                yield PRESETS[ep % len(PRESETS)], ep
            else:
                if ep % 2 == 0:
                    yield PRESETS[ep // 2 % len(PRESETS)], ep
                else:
                    yield None, int(rng.randint(0, 100_000))

    Path(Path(args.out).parent or ".").mkdir(parents=True, exist_ok=True)

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    ep_ids: list[int] = []
    p_names: list[str] = []
    successes: list[bool] = []
    rewards: list[float] = []

    n_success = 0
    n_total = 0
    t0 = time.time()

    if not args.quiet:
        print(f"Generating {args.episodes} expert demonstrations")
        print(
            f"  obs_dim={10 if args.lite else args.obs_dim}  speed_ratio={args.speed_ratio}  "
            f"max_steps={args.max_steps}  lite={args.lite}"
        )
        print(
            f"  {'presets only' if args.presets_only else 'presets + random (50/50)'}"
        )
        print(f"  Output: {args.out}\n")
        print(
            f"  {'ep':>5}  {'planet':<12}  {'ok':>4}  {'steps':>6}  "
            f"{'reward':>8}  {'running_%':>10}"
        )
        print("  " + "─" * 55)

    for ep, (preset, seed) in enumerate(
        planet_schedule(args.episodes, args.presets_only)
    ):
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
            ep_r += float(r)

        success = bool(info.get("success", False))
        planet_name = info.get("planet", preset or "random")

        if success:
            for o, a in zip(ep_obs, ep_act):
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

        if not args.quiet and (ep % 51 == 0 or ep < 5):
            pct = n_success / max(n_total, 1) * 100
            print(
                f"  {ep + 1:5d}  {planet_name:<12}  "
                f"{'✓' if success else '✗':>4}  {len(ep_obs):>6}  "
                f"{ep_r:>8.1f}  {pct:>9.1f}%"
            )

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
            f"{n_success / max(n_total, 1) * 100:.0f}%)"
        )
        print(f"  Pairs (obs,act): {len(obs_arr):,}  (successful episodes only)")
        print(f"  obs shape:  {obs_arr.shape}")
        print(f"  act shape:  {act_arr.shape}")
        print(
            f"  Time:       {elapsed:.0f}s  ({n_total / max(elapsed, 1e-9):.1f} eps/s)"
        )
        print(f"  Saved:      {args.out}")


if __name__ == "__main__":
    main()
