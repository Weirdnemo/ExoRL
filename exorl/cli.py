from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="exorl",
        description="ExoRL convenience CLI.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # New user-facing CLI surface.
    sub.add_parser("train", help="Run SAC training")
    sub.add_parser("demo", help="Generate expert demos")
    sub.add_parser("eval", help="Evaluate trained model generalisation")
    sub.add_parser("planet", help="Inspect preset or random planet summary")
    sub.add_parser("population", help="Generate/load population CSV summaries")
    sub.add_parser("figure", help="Generate population figures from CSV")

    # Legacy aliases kept for backward compatibility.
    sub.add_parser("train-sac", help="Run SAC training")
    sub.add_parser("generate-demos", help="Generate expert demos")
    sub.add_parser("pretrain-bc", help="Run BC pretraining")
    sub.add_parser("eval-generalisation", help="Evaluate generalisation")

    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        raise SystemExit(2)

    # Parse only the subcommand; forward the rest to the underlying command parser.
    cmd = argv[0]
    rest = argv[1:]

    if cmd in ("-h", "--help"):
        build_parser().print_help()
        raise SystemExit(0)

    if cmd in ("train", "train-sac"):
        from exorl.commands.train import main as _main
    elif cmd in ("demo", "generate-demos"):
        from exorl.commands.demo import main as _main
    elif cmd == "pretrain-bc":
        from exorl.commands.pretrain_bc import main as _main
    elif cmd in ("eval", "eval-generalisation"):
        from exorl.commands.eval import main as _main
    elif cmd == "planet":
        from exorl.commands.planet import main as _main
    elif cmd == "population":
        from exorl.commands.population import main as _main
    elif cmd == "figure":
        from exorl.commands.figure import main as _main
    else:
        build_parser().parse_args([cmd])
        raise SystemExit(2)

    _main(rest)
    raise SystemExit(0)


if __name__ == "__main__":
    main()

