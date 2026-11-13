"""Command line: check-config, run, resume, status.

python -m pipeline run --series fact-of-day
python -m pipeline run --series fact-of-day --chaos 0.3 --seed 7
python -m pipeline resume --video 3
python -m pipeline status
"""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from pydantic import ValidationError

from pipeline.adapters.chaos import ChaosConfig
from pipeline.adapters.notify.console import ConsoleNotifier
from pipeline.adapters.registry import ALL_FAKE, build_adapters
from pipeline.config import (
    DEFAULT_CONFIG_DIR,
    PORTS,
    ConfigError,
    ModeConfig,
    list_series,
    load_channel,
    load_series,
    load_topics,
)
from pipeline.costs import Ledger
from pipeline.env import load_env
from pipeline.logs import get_logger, log_to_file
from pipeline.runner import RetryPolicy, Runner, VideoReport
from pipeline.state import StateStore, StepStatus, UnknownVideoError

DB_NAME = "pipeline.db"
LOG_NAME = "pipeline.log.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description="AI pipeline for YouTube Shorts.")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('shorts-pipeline')}"
    )
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check-config", help="validate config/channel.yaml and every series")
    check.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)

    def add_run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--runs-dir", type=Path, default=Path("runs"))
        p.add_argument("--mode", choices=["fake"], help="force every port to fake for this run")
        p.add_argument(
            "--real",
            action="append",
            default=[],
            choices=list(PORTS),
            help="make this port real for this run (repeatable)",
        )
        p.add_argument(
            "--chaos",
            type=float,
            default=0.0,
            metavar="RATE",
            help="fake mode only: share of adapter calls that fail (half as many return garbage)",
        )
        p.add_argument("--seed", type=int, default=0, help="seed for --chaos, to replay a run")
        p.add_argument("--max-attempts", type=int, default=3)
        p.add_argument(
            "--retry-delay",
            type=float,
            default=2.0,
            metavar="SECONDS",
            help="first backoff pause; doubles on each retry",
        )

    run = sub.add_parser("run", help="make one new video")
    run.add_argument("--series", required=True)
    run.add_argument("--topic", help="default: the first unused topic of the series")
    add_run_options(run)

    resume = sub.add_parser("resume", help="continue a failed video from the failed step")
    resume.add_argument("--video", type=int, required=True)
    add_run_options(resume)

    status = sub.add_parser("status", help="list videos and where each one stands")
    status.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser


# --- commands -------------------------------------------------------------------


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


def pick_topic(store: StateStore, series_id: str, topics: list[str]) -> str | None:
    used = {t.lower() for t in store.used_topics(series_id)}
    return next((t for t in topics if t.lower() not in used), None)


def _runner(args: argparse.Namespace, store: StateStore) -> Runner | None:
    channel = load_channel()
    mode = ALL_FAKE if args.mode == "fake" else channel.mode
    if args.real:
        try:  # validate again: some ports only make sense together
            mode = ModeConfig.model_validate(
                {**mode.model_dump(), **dict.fromkeys(args.real, "real")}
            )
        except ValidationError as e:
            reason = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
            print(f"error  {reason}", file=sys.stderr)
            return None
    if args.chaos > 0 and mode.real_ports:
        print("error  --chaos works with fake ports only", file=sys.stderr)
        return None
    if mode.real_ports:
        # Keys for real providers; the shell's own variables win over the file.
        loaded = load_env(Path.cwd() / ".env")
        if loaded:
            print(f"secrets from .env: {', '.join(loaded)}")
    chaos = None
    if args.chaos > 0:
        chaos = ChaosConfig(
            fail_rate=args.chaos, garbage_rate=args.chaos / 2, max_latency_s=0.02, seed=args.seed
        )
    try:
        adapters = build_adapters(channel.models, mode=mode, chaos=chaos)
    except NotImplementedError as e:
        print(f"error  {e}", file=sys.stderr)
        return None
    ports = ", ".join(f"{port}={getattr(mode, port)}" for port in PORTS)
    banner = f"adapters: {ports}" + ("" if mode.real_ports else " ($0)")
    if chaos is not None:
        banner += (
            f" | chaos: {chaos.fail_rate:.0%} fail, {chaos.garbage_rate:.0%} garbage,"
            f" seed {args.seed}"
        )
    print(banner)
    return Runner(
        store=store,
        adapters=adapters,
        runs_dir=args.runs_dir,
        policy=RetryPolicy(max_attempts=args.max_attempts, base_delay_s=args.retry_delay),
        notifier=ConsoleNotifier(),
        budget=channel.budget,
        language=channel.channel.language,
    )


def _print_report(report: VideoReport, series_id: str, topic: str) -> None:
    print(f"video {report.video_id}  {series_id}  {topic!r}")
    for step in report.steps:
        tries = (
            ""
            if step.attempts == 0
            else f"{step.attempts} {'try' if step.attempts == 1 else 'tries'}"
        )
        note = "cached" if step.cached and step.status == StepStatus.DONE else ""
        line = f"  {step.step:<8} {step.status:<8} {tries:<8} ${step.cost_usd:.3f}  {note}"
        print(line.rstrip())
        if step.error:
            print(f"           {step.error}")
    where = f" -> {report.output}" if report.output else ""
    print(f"{report.status} in {report.elapsed_s:.1f} s, paid ${report.cost_usd:.3f}{where}")


def run_command(args: argparse.Namespace) -> int:
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    handler = log_to_file(args.runs_dir / LOG_NAME)
    try:
        with StateStore(args.runs_dir / DB_NAME) as store:
            runner = _runner(args, store)
            if runner is None:
                return 2
            if args.command == "run":
                series_id = args.series
                load_series(series_id)  # fail early on a bad series
                topic = args.topic or pick_topic(store, series_id, load_topics(series_id))
                if topic is None:
                    print(
                        f"error  every topic in config/topics/{series_id}.yaml is used: "
                        "add more or pass --topic",
                        file=sys.stderr,
                    )
                    return 2
                video_id = runner.create_video(series_id, topic)
            else:
                video_id = args.video
                try:
                    record = store.video(video_id)
                except UnknownVideoError as e:
                    print(f"error  {e}", file=sys.stderr)
                    return 2
                series_id, topic = record.series_id, record.topic
            report = runner.run_video(video_id)
            _print_report(report, series_id, topic)
            return 0 if report.status == StepStatus.DONE else 1
    except ConfigError as e:
        print(f"error  {e}", file=sys.stderr)
        return 2
    finally:
        get_logger().removeHandler(handler)
        handler.close()


def status_command(args: argparse.Namespace) -> int:
    db = args.runs_dir / DB_NAME
    if not db.exists():
        print("no videos yet")
        return 0
    with StateStore(db) as store:
        videos = store.videos()
        if not videos:
            print("no videos yet")
            return 0
        budget = load_channel().budget
        spend = Ledger(store, budget).spend()
        print(
            f"this month: ${spend.month_usd:.3f} of ${budget.max_monthly_usd:.2f}"
            f" ({spend.month_calls} calls), per video limit ${budget.max_cost_per_video_usd:.2f}"
        )
        for video in videos:
            steps = store.steps(video.id)
            current = next((s for s in steps if s.status != StepStatus.DONE), None)
            where = "" if current is None else f"  at {current.name} ({current.status})"
            state = store.video_status(video.id)
            paid = store.video_cost(video.id)
            print(
                f"{video.id:>4}  {state:<8} ${paid:.3f}  {video.series_id}  {video.topic!r}{where}"
            )
            if current is not None and current.error:
                print(f"        last error: {current.error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-config":
        return check_config(args.config_dir)
    if args.command in {"run", "resume"}:
        return run_command(args)
    if args.command == "status":
        return status_command(args)
    parser.print_help()
    return 0
