from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    from exorl.commands.eval_generalisation import main as _main

    _main(argv)
