"""让 ``python -m bkt_em`` 可用。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
