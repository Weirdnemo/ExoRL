from __future__ import annotations

import argparse

from exorl.core.generator import PRESETS, PlanetGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Planet inspection helpers")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Preset planet to inspect",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a random planet instead of using --preset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for --random mode")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable planet summary",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.random:
        planet = PlanetGenerator(seed=args.seed).generate(name=f"Random-{args.seed}")
    else:
        preset = args.preset or "earth"
        planet = PRESETS[preset]()

    # Summary is the default output for this command.
    print(planet.summary())
