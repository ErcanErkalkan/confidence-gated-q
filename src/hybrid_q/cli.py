from __future__ import annotations

import argparse

from .experiment import run_config
from .statistics import aggregate


def run_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_config(args.config))


def aggregate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate(args.input, args.output)
