"""CLI entry for SFT training/inference helpers."""

import argparse

from sft_infer import run_demo_inference
from sft_train import run_sft_training


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT workflow entry")
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="infer",
        help="Run training pipeline or inference demo.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_sft_training()
    else:
        run_demo_inference()


if __name__ == "__main__":
    main()
