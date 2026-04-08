"""
Command entrypoints that back the `planet-rl` CLI.

These modules are import-safe (no argparse side effects on import) and can be
invoked either as:
  - console script: `planet-rl <subcommand> -- ...`
  - module: `python -m planet_rl.commands.<command> --help`

The legacy `scripts/*.py` files call into these entrypoints for backwards
compatibility.
"""

