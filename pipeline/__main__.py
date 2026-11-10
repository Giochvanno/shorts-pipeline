"""CLI entry point: ``python -m pipeline ...``.

Subcommands (run, soak, ...) are added in later steps.
"""

import argparse
import sys
from importlib.metadata import version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="AI pipeline for YouTube Shorts.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('shorts-pipeline')}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
