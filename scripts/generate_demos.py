"""
Legacy entrypoint (backwards compatible).

Prefer:
  - `exorl generate-demos ...`
  - `python -m exorl.commands.generate_demos ...`
"""

from exorl.commands.generate_demos import main


if __name__ == "__main__":
    main()

