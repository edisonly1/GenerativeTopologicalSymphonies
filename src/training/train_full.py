"""CLI entry point for the full model stack."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_model.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Full-model scaffold ready. Implement training loop using {args.config}.")


if __name__ == "__main__":
    main()
