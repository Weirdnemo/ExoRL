"""
Command entrypoints that back the `exorl` CLI.

These modules are import-safe (no argparse side effects on import) and can be
invoked either as:
  - console script: `exorl <subcommand> ...`
  - module: `python -m exorl.commands.<command> --help`

The legacy `scripts/*.py` files call into these entrypoints for backwards
compatibility.
"""

