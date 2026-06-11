from __future__ import annotations

import argparse

from src.hybrid_q.statistics import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate(args.input, args.output)


if __name__ == "__main__":
    main()
