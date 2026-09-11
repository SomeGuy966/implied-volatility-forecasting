"""Command-line entry point: ``vixcast [options]`` or ``python -m vixcast``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import DEFAULT_LEVELS, DEFAULT_MODELS, Config
from .models import AVAILABLE_MODELS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vixcast",
        description="Walk-forward VIX forecasting with HAR / GARCH models and rigorous evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default="^VIX")
    p.add_argument("--data-start", default="2010-01-01", help="first date of history to download")
    p.add_argument("--test-start", default="2024-01-01", help="first forecast *target* date")
    p.add_argument(
        "--test-end", default=None, help="last forecast target date (default: end of data)"
    )
    p.add_argument("-H", "--horizon", type=int, default=1, help="forecast horizon in trading days")
    p.add_argument(
        "--window",
        type=int,
        default=None,
        help="rolling estimation window in trading days (default: expanding from --data-start)",
    )
    p.add_argument("--min-train", type=int, default=260, help="minimum history before forecasting")
    p.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"comma-separated subset of {', '.join(AVAILABLE_MODELS)}",
    )
    p.add_argument(
        "--levels",
        default=",".join(str(x) for x in DEFAULT_LEVELS),
        help="comma-separated central interval levels",
    )
    p.add_argument(
        "--n-jobs", type=int, default=-1, help="worker processes for GARCH fits (-1 = all cores)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--refresh-data", action="store_true", help="re-download prices and overwrite the cache"
    )
    p.add_argument("--no-plots", action="store_true", help="skip figure generation")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        ticker=args.ticker,
        data_start=args.data_start,
        test_start=args.test_start,
        test_end=args.test_end,
        horizon=args.horizon,
        window=args.window,
        min_train=args.min_train,
        levels=tuple(float(x) for x in args.levels.split(",") if x),
        models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
        n_jobs=args.n_jobs,
        seed=args.seed,
        refresh_data=args.refresh_data,
        make_plots=not args.no_plots,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    from .pipeline import run  # deferred so ``--help`` stays fast

    outputs = run(config_from_args(args))
    print(outputs.summary)
    print("Artifacts:")
    for path in outputs.files:
        print(f"  {path}")
    return 0
