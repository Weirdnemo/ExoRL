from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="planet-rl",
        description="Planet-RL convenience CLI.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

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

    if cmd == "train-sac":
        from planet_rl.commands.train_sac import main as _main

        _main(rest)
        raise SystemExit(0)
    elif cmd == "generate-demos":
        from planet_rl.commands.generate_demos import main as _main

        _main(rest)
        raise SystemExit(0)
    elif cmd == "pretrain-bc":
        from planet_rl.commands.pretrain_bc import main as _main

        _main(rest)
        raise SystemExit(0)
    elif cmd == "eval-generalisation":
        from planet_rl.commands.eval_generalisation import main as _main

        _main(rest)
        raise SystemExit(0)
    else:
        # Let argparse show consistent help + error code.
        build_parser().parse_args([cmd])
        raise SystemExit(2)


if __name__ == "__main__":
    main()

