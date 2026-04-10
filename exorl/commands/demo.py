from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    from exorl.commands.generate_demos import main as _main

    _main(argv)
