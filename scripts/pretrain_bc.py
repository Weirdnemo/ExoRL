"""
Legacy entrypoint (backwards compatible).

Prefer:
  - `planet-rl pretrain-bc -- ...`
  - `python -m planet_rl.commands.pretrain_bc ...`
"""

from planet_rl.commands.pretrain_bc import main


if __name__ == "__main__":
    main()

