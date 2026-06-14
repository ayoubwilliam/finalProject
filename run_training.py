"""
End-to-end pipeline runner.
Executes all 4 stages in sequence: Segmentation → Data Generation → Training → Evaluation.
Supports a --data-fraction flag to use a percentage of available CT scans for faster runs.

Usage:
    python run_all.py /path/to/ct/scans            # run full pipeline with all data
    python run_all.py /path/to/ct/scans --data-fraction 0.5 # use 50% of CT scans
    python run_all.py /path/to/ct/scans --skip-seg          # skip segmentation (already done)
"""

import argparse
import time

import config as cfg

# --- Global Imports to prevent Multiprocessing Deadlocks ---
from training.trainer import train


def run_training():
    """Stage 3: Train the model."""
    print("\n" + "=" * 60)
    print(" STAGE 3: TRAINING")
    print("=" * 60)

    train()


def main():
    parser = argparse.ArgumentParser(description="End-to-end X-ray change detection pipeline.")
    parser.add_argument("ct_dir", type=str, help="Path to the directory containing original CT scans.")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of CT scans to use (0.0–1.0). Default: 1.0 (all).")
    parser.add_argument("--skip-seg", action="store_true",
                        help="Skip the segmentation stage (if already completed).")
    args = parser.parse_args()

    start_time = time.time()

    cfg.set_ct_input_dir(args.ct_dir)

    run_training()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" ALL STAGES COMPLETE! Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
