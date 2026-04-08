"""
Legacy entrypoint (backwards compatible).

Prefer:
  - `exorl eval-generalisation ...`
  - `python -m exorl.commands.eval_generalisation ...`
"""

from exorl.commands.eval_generalisation import main


if __name__ == "__main__":
    main()

