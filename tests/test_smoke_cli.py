from __future__ import annotations

from exorl import cli


def test_cli_parser_accepts_legacy_commands() -> None:
    parser = cli.build_parser()
    for cmd in ("train-sac", "generate-demos", "pretrain-bc", "eval-generalisation"):
        args = parser.parse_args([cmd])
        assert args.cmd == cmd


def test_cli_parser_accepts_v2_commands() -> None:
    parser = cli.build_parser()
    for cmd in ("train", "demo", "eval", "planet", "population", "figure"):
        args = parser.parse_args([cmd])
        assert args.cmd == cmd
