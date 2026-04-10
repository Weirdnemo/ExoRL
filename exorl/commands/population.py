from __future__ import annotations

import argparse
from pathlib import Path

from exorl.core.population import PlanetPopulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Planet population generation/inspection")
    parser.add_argument("--n", type=int, default=500, help="Population size to generate")
    parser.add_argument("--seed", type=int, default=0, help="Generation seed")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path for generated population (e.g. pop.csv)",
    )
    parser.add_argument(
        "--load",
        default=None,
        help="Load an existing population CSV instead of generating",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable generation progress prints",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.load:
        pop = PlanetPopulation.load(args.load)
        print(pop.summary())
        return

    pop = PlanetPopulation.generate(n=args.n, seed=args.seed, verbose=not args.no_verbose)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pop.save(str(out_path))
        print(f"Saved: {out_path}")
    print(pop.summary())
