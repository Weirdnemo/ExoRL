"""
Legacy entrypoint (backwards compatible).

Prefer:
  - `exorl pretrain-bc ...`
  - `python -m exorl.commands.pretrain_bc ...`
"""

from exorl.commands.pretrain_bc import main


if __name__ == "__main__":
    main()

