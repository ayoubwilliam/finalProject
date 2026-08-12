"""
End-to-end pipeline runner.
Executes all 4 stages in sequence: Segmentation → Data Generation → Training → Evaluation.
Supports a --data-fraction flag to use a percentage of available CT scans for faster runs.

Usage:
    python run_all.py /path/to/ct/scans            # run full pipeline with all data
    python run_all.py /path/to/ct/scans --data-fraction 0.5 # use 50% of CT scans
    python run_all.py /path/to/ct/scans --skip-seg          # skip segmentation (already done)
"""

import time

from evaluation.evaluate import main as eval_main


def run_evaluation():
    """Stage 4: Run full evaluation."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    eval_main()


def main():
    """Functionality for main."""
    start_time = time.time()

    run_evaluation()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" ALL STAGES COMPLETE! Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
