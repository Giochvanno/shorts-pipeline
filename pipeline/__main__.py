"""Entry point: ``python -m pipeline ...``. The commands live in pipeline/cli.py."""

import sys

from pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
