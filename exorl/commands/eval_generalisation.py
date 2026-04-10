from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import exorl.core.planet_io
from exorl.core.env import OrbitalInsertionEnv


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


def _hohmann_fuel_fraction(planet) -> float | None:
    """Estimate propellant fraction for a 500 km -> 300 km Hohmann transfer."""
    try:
        approach_alt = 500_000.0
        target_alt = 300_000.0
        dv1, dv2 = planet.hohmann_delta_v(approach_alt, target_alt)
        hohmann_dv = float(dv1 + dv2)

        # Match env defaults used by OrbitalInsertionEnv.
        Isp = 320.0
        g0 = 9.80665
        wet_mass = 1000.0
        propellant_mass = 700.0

        m_after = wet_mass * np.exp(-hohmann_dv / (Isp * g0))
        hohmann_fuel_frac = (wet_mass - m_after) / propellant_mass
        return float(np.clip(hohmann_fuel_frac, 0.0, 1.0))
    except Exception:
        return None


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

    def evaluate_on_planet(
        planet_name_or_obj,
        n_episodes: int,
        policy,
        *,
        is_preset: bool = True,
        is_random: bool = False,
        planet_fingerprint: str | None = None,
    ):
        if is_preset:
            env = OrbitalInsertionEnv(
                lite_mode=bool(args.lite),
                obs_dim=obs_dim,
                planet_preset=planet_name_or_obj,
                randomize_planet=False,
                initial_altitude=0,
                use_science_atmosphere=sci,
                use_science_j2=sci,
                attach_star=sci,
            )
        else:
            env = OrbitalInsertionEnv(
                lite_mode=bool(args.lite),
                obs_dim=obs_dim,
                planet=planet_name_or_obj,
                randomize_planet=False,
                curriculum_mode=False,
                use_science_atmosphere=sci,
                use_science_j2=sci,
                attach_star=False if args.lite else False,
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
            # New env info emits fuel_kg; convert to fraction of initial propellant mass.
            fuel_kg = float(info.get("fuel_kg", 700.0))
            fuels.append(float(np.clip(fuel_kg / 700.0, 0.0, 1.0)))
            lengths.append(ep_len)

        from exorl.core.generator import PRESETS

        if is_preset:
            p = PRESETS[planet_name_or_obj]()
        else:
            p = planet_name_or_obj
        hohmann_fuel_frac = _hohmann_fuel_fraction(p)

        return {
            "planet": str(planet_name_or_obj) if is_preset else getattr(p, "name", "?"),
            "fingerprint": planet_fingerprint,
            "n_episodes": int(n_episodes),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_fuel_remaining": float(np.mean(fuels)),
            "mean_ep_length": float(np.mean(lengths)),
            "hohmann_fuel_frac": hohmann_fuel_frac,
            "fuel_efficiency_vs_hohmann": (
                float(np.mean(fuels)) / hohmann_fuel_frac
                if hohmann_fuel_frac and hohmann_fuel_frac > 0
                else None
            ),
        }

    results = {
        "model": str(args.model),
        "preset_results": [],
        "kepler_results": [],
        "lite": bool(args.lite),
    }

    header = (
        f"{'Planet':<10}  {'Success':>8}  {'Reward':>10}  "
        f"{'Fuel':>6}  {'Hohmann':>8}  {'Efficiency':>10}"
    )
    print(header)
    print("─" * len(header))

    for p in args.planets:
        print(f"Evaluating: {p}  (episodes={args.episodes})")
        r = evaluate_on_planet(p, args.episodes, model, is_preset=True)
        results["preset_results"].append(r)
        eff = r["fuel_efficiency_vs_hohmann"]
        print(
            f"  {p:<10}  "
            f"{r['success_rate']:>7.1%}  "
            f"{r['mean_reward']:>10.1f}  "
            f"{r['mean_fuel_remaining']:>6.2f}  "
            f"{(r['hohmann_fuel_frac'] or 0):>8.2f}  "
            f"{'N/A' if eff is None else f'{eff:.2f}x':>10}"
        )

    if args.random_baseline:
        results["random_baseline"] = []
        for p in args.planets:
            print(f"Evaluating random baseline: {p}")
            r = evaluate_on_planet(
                p, max(args.episodes // 2, 1), model, is_preset=True, is_random=True
            )
            results["random_baseline"].append(r)
            print(f"  {p:<10}  {r['success_rate']:>7.1%}  {r['mean_reward']:>10.1f}")

    if args.kepler:
        print("\nKepler catalog evaluation...")
        from exorl.core.kepler_catalog import KeplerCatalog

        catalog = KeplerCatalog()
        candidates = catalog.rl_training_candidates(min_score=0.35)[:5]
        print(f"  Top {len(candidates)} RL candidates:")

        for entry in candidates:
            try:
                planet_obj = entry.to_planet()
                fp = (
                    planet_obj.fingerprint
                    if hasattr(planet_obj, "fingerprint")
                    else None
                )
                r = evaluate_on_planet(
                    planet_obj,
                    max(args.episodes // 2, 1),
                    model,
                    is_preset=False,
                    planet_fingerprint=fp,
                )
                r["planet"] = entry.name
                results["kepler_results"].append(r)
                print(
                    f"  {entry.name:<22}  success={r['success_rate']:>6.1%}  "
                    f"reward={r['mean_reward']:>8.1f}"
                )
            except Exception as e:
                print(f"  {entry.name}: ERROR - {e}")

    if results["preset_results"]:
        preset_success = np.mean([r["success_rate"] for r in results["preset_results"]])
        print(f"\n{'═' * 60}")
        print("  Zero-shot generalisation summary")
        print(f"  Planets evaluated:  {args.planets}")
        print(f"  Mean success rate:  {preset_success:.1%}")
        print(f"  Model:              {Path(args.model).name}")
        print(f"{'═' * 60}")

    out_json = out_dir / "generalisation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
