"""Train or dry-run the MusicTransformer benchmark baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.benchmarks import get_benchmark_model_spec, validate_benchmark_model_config
from training.train_baseline import load_config, run_baseline_training


def parse_args() -> argparse.Namespace:
    spec = get_benchmark_model_spec("music_transformer")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=spec.default_config)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-pieces", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    validate_benchmark_model_config(config, expected_model="music_transformer")
    result = run_baseline_training(
        config,
        config_path=config_path,
        dry_run=args.dry_run,
        max_steps_override=args.max_steps,
        limit_pieces=args.limit_pieces,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
