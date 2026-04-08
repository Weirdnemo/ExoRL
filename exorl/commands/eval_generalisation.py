from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import planet_rl.core.planet_io
from planet_rl.core.env import OrbitalInsertionEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot generalisation evaluation")
    parser.add_argument("--model", required=True, help="Path to trained .zip model")
    parser.add_argument(
        "--planets",
        nargs="+",
        default=["earth", "mars", "venus", "moon", "titan"],
        help="Preset planets to evaluate on",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per planet")
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also evaluate a random policy for comparison",
    )
    parser.add_argument(
        "--kepler",
        action="store_true",
        help="Also evaluate on Kepler catalog RL candidates",
    )
    parser.add_argument(
        "--out", default=None, help="Output directory (default: next to model)"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Evaluate using the lite env settings (obs_dim=10, no science stack).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from stable_baselines3 import SAC

    print(f"Loading model: {args.model}")
    model = SAC.load(args.model)
    print(f"  Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    out_dir = Path(args.out) if args.out else Path(args.model).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 10 if args.lite else 18
    sci = not args.lite

    def evaluate_on_planet(planet_name: str, n_episodes: int, policy, is_random=False):
        env = OrbitalInsertionEnv(
            lite_mode=bool(args.lite),
            obs_dim=obs_dim,
            planet_preset=planet_name,
            randomize_planet=False,
            initial_altitude=0,
            use_science_atmosphere=sci,
            use_science_j2=sci,
            attach_star=sci,
        )

        rewards, successes, fuels, lengths = [], [], [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = trunc = False
            ep_r = 0.0
            ep_len = 0
            info = {}
            while not (done or trunc):
                if is_random:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(obs.reshape(1, -1), deterministic=True)
                    action = action[0]
                obs, rew, done, trunc, info = env.step(action)
                ep_r += float(rew)
                ep_len += 1

            rewards.append(ep_r)
            successes.append(float(info.get("success", False)))
            fuels.append(float(info.get("fuel_fraction", 0.5)))
            lengths.append(ep_len)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_fuel_fraction": float(np.mean(fuels)),
            "mean_ep_len": float(np.mean(lengths)),
        }

    results = {
        "model": str(args.model),
        "episodes_per_planet": args.episodes,
        "planets": {},
        "lite": bool(args.lite),
    }

    for p in args.planets:
        print(f"Evaluating: {p}  (episodes={args.episodes})")
        results["planets"][p] = evaluate_on_planet(p, args.episodes, model)

    if args.random_baseline:
        results["random_baseline"] = {}
        for p in args.planets:
            print(f"Evaluating random baseline: {p}")
            results["random_baseline"][p] = evaluate_on_planet(
                p, args.episodes, model, is_random=True
            )

    out_json = out_dir / "generalisation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()

