"""Allow ``python -m finaudit``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
