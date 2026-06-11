from __future__ import annotations

import argparse

from src.hybrid_q.experiment import run_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = run_config(args.config)
    print(output)


if __name__ == "__main__":
    main()
