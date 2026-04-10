from __future__ import annotations

import argparse
from pathlib import Path

from exorl.core.population import PlanetPopulation
from exorl.visualization.visualizer import (
    plot_correlation_heatmap,
    plot_habitability_distribution,
    plot_mass_radius,
    save_figure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate publication-style figures")
    parser.add_argument(
        "--type",
        choices=["mass-radius", "habitability", "correlation"],
        required=True,
        help="Figure type to generate",
    )
    parser.add_argument(
        "--population",
        required=True,
        help="Path to population CSV file",
    )
    parser.add_argument(
        "--out",
        default="figures/science_figures",
        help="Output directory for generated figure files",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output filename stem (without extension)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pop = PlanetPopulation.load(args.population)

    if args.type == "mass-radius":
        fig = plot_mass_radius(pop)
        default_name = "mass_radius"
    elif args.type == "habitability":
        fig = plot_habitability_distribution(pop)
        default_name = "habitability_distribution"
    else:
        fig = plot_correlation_heatmap(pop)
        default_name = "correlation_heatmap"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = args.name or default_name
    save_figure(fig, filename=filename, output_dir=str(out_dir))
