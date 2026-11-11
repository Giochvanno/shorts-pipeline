"""CLI entry point: ``python -m pipeline ...``.

More subcommands (run, soak, ...) are added in later steps.
"""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from pipeline.config import DEFAULT_CONFIG_DIR, ConfigError, list_series, load_channel, load_series


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
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check-config", help="validate config/channel.yaml and every series")
    check.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    return parser


def check_config(config_dir: Path) -> int:
    try:
        channel = load_channel(config_dir)
        print(f"ok  channel.yaml  ({channel.channel.name}, upload: {channel.upload.privacy})")
        for series_id in list_series(config_dir):
            series = load_series(series_id, config_dir)
            print(f"ok  series/{series_id}.yaml  ({series.title})")
    except ConfigError as e:
        print(f"error  {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-config":
        return check_config(args.config_dir)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
