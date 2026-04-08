"""
eval_generalisation.py — Zero-shot generalisation evaluation.

Loads a trained model and evaluates it on each of the 5 solar system
preset planets without any retraining. Measures:
  - Insertion success rate per planet
  - Fuel consumption vs analytical Hohmann optimum
  - Episode reward
  - Comparison to a random policy baseline

This is the key experiment for the generalisation paper:
  "Can a policy trained on random/curriculum planets zero-shot to
   Earth, Mars, Venus, Moon, and Titan?"

Usage
-----
    # Evaluate a trained model on all presets
    python scripts/eval_generalisation.py --model training_runs/<run>/model_final.zip

    # Evaluate on specific planets only
    python scripts/eval_generalisation.py --model path/to/model.zip --planets earth mars

    # Also compare against a random policy
    python scripts/eval_generalisation.py --model path/to/model.zip --random-baseline

    # Evaluate the Kepler catalog targets
    python scripts/eval_generalisation.py --model path/to/model.zip --kepler

Outputs
-------
    generalisation_results.json   — full results table
    generalisation_table.png      — publication-ready table figure
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

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
    "--out", default=None, help="Output path (default: next to model file)"
)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import SAC

import planet_rl.core.planet_io
from planet_rl.core.env import OrbitalInsertionEnv

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {args.model}")
model = SAC.load(args.model)
print(f"  Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

out_dir = Path(args.out) if args.out else Path(args.model).parent
out_dir.mkdir(parents=True, exist_ok=True)

# ── Evaluation helper ─────────────────────────────────────────────────────────


def evaluate_on_planet(
    planet_name_or_obj, n_episodes, policy, is_preset=True, planet_fingerprint=None
):
    """
    Run n_episodes on a fixed planet and return metrics dict.
    policy: callable(obs) → action, or "random"
    """
    if is_preset:
        env = OrbitalInsertionEnv(
            obs_dim=18,
            planet_preset=planet_name_or_obj,
            randomize_planet=False,
            initial_altitude=0,  # must match training env
            use_science_atmosphere=True,
            use_science_j2=True,
            attach_star=True,
        )
    else:
        env = OrbitalInsertionEnv(
            obs_dim=18,
            planet=planet_name_or_obj,
            randomize_planet=False,
            curriculum_mode=False,
            use_science_atmosphere=True,
            use_science_j2=True,
            attach_star=False,
        )

    rewards, successes, fuels, lengths = [], [], [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = trunc = False
        ep_r = 0.0
        ep_len = 0
        ep_info = {}

        while not (done or trunc):
            if policy == "random":
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs.reshape(1, -1), deterministic=True)
                action = action[0]

            obs, rew, done, trunc, ep_info = env.step(action)
            ep_r += rew
            ep_len += 1

        rewards.append(ep_r)
        successes.append(float(ep_info.get("success", False)))
        # fuel_fraction not in info dict — compute from fuel_kg
        fuel_kg = float(ep_info.get("fuel_kg", 700.0))
        fuels.append(min(fuel_kg / 700.0, 1.0))  # normalise by default wet_mass
        lengths.append(ep_len)

    # Compute Hohmann optimal fuel fraction for comparison
    from planet_rl.core.generator import PRESETS

    if is_preset:
        p = PRESETS[planet_name_or_obj]()
    else:
        p = planet_name_or_obj

    # Hohmann ΔV from approach altitude to target 300 km
    try:
        approach_alt = 500_000
        target_alt = 300_000
        dv1, dv2 = p.hohmann_delta_v(approach_alt, target_alt)
        hohmann_dv = dv1 + dv2
        # Convert to fuel fraction: m_prop/m_wet via Tsiolkovsky
        Isp = 320.0
        g0 = 9.80665
        wet_mass = 1000.0
        m_after = wet_mass * np.exp(-(hohmann_dv) / (Isp * g0))
        hohmann_fuel_frac = (wet_mass - m_after) / (wet_mass * 0.7)
        hohmann_fuel_frac = max(0, min(1, hohmann_fuel_frac))
    except Exception:
        hohmann_fuel_frac = None

    return {
        "planet": planet_name_or_obj if is_preset else getattr(p, "name", "?"),
        "fingerprint": planet_fingerprint,
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_fuel_remaining": float(np.mean(fuels)),
        "mean_ep_length": float(np.mean(lengths)),
        "hohmann_fuel_frac": hohmann_fuel_frac,
        "fuel_efficiency_vs_hohmann": (
            float(np.mean(fuels)) / hohmann_fuel_frac
            if hohmann_fuel_frac and hohmann_fuel_frac > 0
            else None
        ),
    }


# ── Evaluate on presets ───────────────────────────────────────────────────────
print(
    f"\nEvaluating on {len(args.planets)} preset planet(s), "
    f"{args.episodes} episodes each...\n"
)

results = {"model": str(args.model), "preset_results": [], "kepler_results": []}

header = (
    f"{'Planet':<10}  {'Success':>8}  {'Reward':>10}  "
    f"{'Fuel':>6}  {'Hohmann':>8}  {'Efficiency':>10}"
)
print(header)
print("─" * len(header))

for planet_name in args.planets:
    r = evaluate_on_planet(planet_name, args.episodes, model, is_preset=True)
    results["preset_results"].append(r)

    eff = r["fuel_efficiency_vs_hohmann"]
    print(
        f"  {planet_name:<10}  "
        f"{r['success_rate']:>7.1%}  "
        f"{r['mean_reward']:>10.1f}  "
        f"{r['mean_fuel_remaining']:>6.2f}  "
        f"{r['hohmann_fuel_frac'] or 0:>8.2f}  "
        f"{'N/A' if eff is None else f'{eff:.2f}x':>10}"
    )

# ── Random baseline ───────────────────────────────────────────────────────────
if args.random_baseline:
    print("\nRandom policy baseline:")
    random_results = []
    for planet_name in args.planets:
        r = evaluate_on_planet(
            planet_name, args.episodes // 2, "random", is_preset=True
        )
        random_results.append(r)
        print(
            f"  {planet_name:<10}  {r['success_rate']:>7.1%}  {r['mean_reward']:>10.1f}"
        )
    results["random_baseline"] = random_results

# ── Kepler catalog evaluation ─────────────────────────────────────────────────
if args.kepler:
    print("\nKepler catalog evaluation...")
    from planet_rl.core.kepler_catalog import KeplerCatalog

    catalog = KeplerCatalog()
    candidates = catalog.rl_training_candidates(min_score=0.35)[:5]
    print(f"  Top {len(candidates)} RL candidates:")

    for entry in candidates:
        try:
            planet_obj = entry.to_planet()
            fp = planet_obj.fingerprint if hasattr(planet_obj, "fingerprint") else None
            r = evaluate_on_planet(
                planet_obj,
                args.episodes // 2,
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
            print(f"  {entry.name}: ERROR — {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
preset_success = np.mean([r["success_rate"] for r in results["preset_results"]])
print(f"\n{'═' * 60}")
print(f"  Zero-shot generalisation summary")
print(f"  Planets evaluated:  {args.planets}")
print(f"  Mean success rate:  {preset_success:.1%}")
print(f"  Model:              {Path(args.model).name}")
print(f"{'═' * 60}")

# Save results
out_path = out_dir / "generalisation_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# ── Figure ────────────────────────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planet_names = [r["planet"] for r in results["preset_results"]]
    success_rates = [r["success_rate"] * 100 for r in results["preset_results"]]
    mean_rewards = [r["mean_reward"] for r in results["preset_results"]]

    BLUE = "#0072B2"
    ORANGE = "#E69F00"
    GREEN = "#009E73"

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.16, wspace=0.38)

    x = np.arange(len(planet_names))

    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#eeeeee", lw=0.8, zorder=0)
        ax.tick_params(labelsize=9)

    bars = axes[0].bar(
        x, success_rates, color=BLUE, alpha=0.85, zorder=3, edgecolor="white", lw=0.5
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p.title() for p in planet_names], fontsize=9)
    axes[0].set_ylabel("Success rate (%)", fontsize=10)
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Insertion success — zero-shot", fontsize=10, fontweight="bold")
    for bar, val in zip(bars, success_rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bars2 = axes[1].bar(
        x, mean_rewards, color=ORANGE, alpha=0.85, zorder=3, edgecolor="white", lw=0.5
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.title() for p in planet_names], fontsize=9)
    axes[1].set_ylabel("Mean episode reward", fontsize=10)
    axes[1].set_title("Episode reward — zero-shot", fontsize=10, fontweight="bold")

    if args.random_baseline and "random_baseline" in results:
        rand_sr = [r["success_rate"] * 100 for r in results["random_baseline"]]
        axes[0].plot(
            x, rand_sr, "o--", color="#CC3311", ms=5, lw=1.2, label="Random policy"
        )
        axes[0].legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"Zero-shot generalisation  |  {Path(args.model).parent.name}",
        fontsize=10,
        fontweight="bold",
    )
    fig.savefig(str(out_dir / "generalisation_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/generalisation_table.png")

except Exception as e:
    print(f"Figure skipped: {e}")

print("\nDone.")
