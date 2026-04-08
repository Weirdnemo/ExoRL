"""
Legacy entrypoint (backwards compatible).

Prefer:
  - `planet-rl eval-generalisation -- ...`
  - `python -m planet_rl.commands.eval_generalisation ...`
"""

from planet_rl.commands.eval_generalisation import main


if __name__ == "__main__":
    main()

