"""
population_demo.py — Generate a planet population and produce all analysis figures.

Usage
-----
    python population_demo.py              # 500 planets, default settings
    python population_demo.py --n 1000    # larger population
    python population_demo.py --seed 7    # different random seed
    python population_demo.py --fast      # 100 planets, quick test

Outputs (saved to ./examples/)                      (raw data, openable in Excel)
"""

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Planet population analysis")
parser.add_argument("--n", type=int, default=500, help="Number of planets")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--out", type=str, default="figures/science_figures", help="Output directory"
)
parser.add_argument(
    "--csv", type=str, default="examples/csv-data", help="CSV output location"
)
parser.add_argument("--fast", action="store_true", help="Quick test (100 planets)")
parser.add_argument("--no-atm", action="store_true", help="Skip atmosphere (faster)")
parser.add_argument(
    "--save-csv", action="store_true", default=True, help="Save population data to CSV"
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="Load existing CSV instead of generating  e.g. --load population_500.csv",
)
args = parser.parse_args()

if args.fast:
    args.n = 100

os.makedirs(args.out, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
print("Loading modules...")
from exorl.core.population import PlanetPopulation
from exorl.visualization import (
    apply_journal_style,
    plot_correlation_heatmap,
    plot_habitability_distribution,
    plot_mass_radius,
    plot_population_dashboard,
    save_figure,
)

apply_journal_style()

# ── Load or generate population ──────────────────────────────────────────────
if args.load:
    if not os.path.exists(args.load):
        print(f"Error: file not found: {args.load}")
        sys.exit(1)
    print(f"\nLoading population from {args.load}...")
    t0 = time.time()
    pop = PlanetPopulation.load(args.load)
    print(f"Loaded {len(pop)} planets in {time.time() - t0:.2f}s")
else:
    print(f"\nGenerating {args.n} planets  (seed={args.seed})...")
    t0 = time.time()
    pop = PlanetPopulation.generate(
        n=args.n,
        seed=args.seed,
        atmosphere_enabled=not args.no_atm,
        oblateness_enabled=True,
        magnetic_field_enabled=True,
        attach_random_star=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(
        f"\nGenerated {len(pop)} planets in {elapsed:.1f}s  ({elapsed / args.n * 1000:.0f} ms/planet)"
    )

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(pop.summary())
print("=" * 55)

# ── RL training stats ─────────────────────────────────────────────────────────
stats = pop.rl_training_stats()
print("\nRL training distribution:")
print(
    f"  Gravity range (5-95th pct):  {stats['gravity_p5']:.1f} – {stats['gravity_p95']:.1f} m/s²"
)
print(
    f"  Radius range  (5-95th pct):  {stats['radius_p5']:.2f} – {stats['radius_p95']:.2f} R⊕"
)
print(
    f"  Mass range    (5-95th pct):  {stats['mass_p5']:.3f} – {stats['mass_p95']:.1f} M⊕"
)
print(
    f"  Hab score mean / std:         {stats['hab_score_mean']:.3f} / {stats['hab_score_std']:.3f}"
)
print(f"  Fraction score > 0.5:         {100 * stats['frac_habitable']:.1f}%")
print(f"  Fraction with active dynamo:  {100 * stats['frac_with_dynamo']:.1f}%")
print(f"  Median transit depth:         {stats['transit_ppm_median']:.0f} ppm")
print(f"  Median J2:                    {stats['j2_median']:.2e}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
if args.save_csv and not args.load:
    csv_path = os.path.join(args.csv, f"population_{len(pop)}.csv")
    pop.save(csv_path)
    print(f"\nSaved raw data: {csv_path}")
elif args.load:
    print(f"\nUsing loaded data — CSV save skipped (already have the source file)")

# ── Figures ───────────────────────────────────────────────────────────────────
print("\nProducing figures...")

# Figure 15 — Mass-radius diagram
print("  [1/4] Mass-radius diagram...")
fig = plot_mass_radius(pop, colour_by="hab_score", show_composition_lines=True)
save_figure(fig, "fig15_mass_radius", args.out)
plt.close(fig)

# Figure 16 — Habitability distribution
print("  [2/4] Habitability distribution...")
fig = plot_habitability_distribution(pop)
save_figure(fig, "fig16_habitability_distribution", args.out)
plt.close(fig)

# Figure 17 — Correlation heatmap
print("  [3/4] Correlation heatmap...")
fig = plot_correlation_heatmap(pop)
save_figure(fig, "fig17_correlation_heatmap", args.out)
plt.close(fig)

# Figure 18 — Dashboard (saves itself)
print("  [4/4] Population dashboard...")
fig = plot_population_dashboard(pop, args.out, "fig18_population_dashboard")
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nAll figures saved to ./{args.out}/")
print()
for fn in sorted(os.listdir(args.out)):
    if fn.startswith("fig1") and fn.endswith(".png"):
        kb = os.path.getsize(os.path.join(args.out, fn)) // 1024
        print(f"  {fn:<50s}  {kb:4d} kB")

print("\nDone.")
