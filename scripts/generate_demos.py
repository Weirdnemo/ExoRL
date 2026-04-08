"""
Legacy entrypoint (backwards compatible).

Prefer:
  - `planet-rl generate-demos -- ...`
  - `python -m planet_rl.commands.generate_demos ...`
"""

from planet_rl.commands.generate_demos import main


if __name__ == "__main__":
    main()

