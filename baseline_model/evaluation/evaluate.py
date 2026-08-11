"""
Master evaluation runner.
Executes the full evaluation pipeline: PW Dice → PW AUC → CC Metrics → CC Dice → Visuals → CC Visuals.
"""

import config as cfg
from evaluation.pw_dice import run_pw_dice
from evaluation.pw_auc import run_pw_auc
from evaluation.cc_metrics import run_cc_metrics
from evaluation.cc_dice import run_cc_dice
from evaluation.cc_visuals import run_cc_visuals
from evaluation.visuals import run_standard_visuals
from evaluation.angle_metrics import run_angle_metrics


def main():
    """Runs the entire evaluation pipeline sequentially."""
    print("=" * 50)
    print(" STARTING COMPLETE EVALUATION PIPELINE")
    print(f" Model: {cfg.SELECTED_MODEL} | Device: {cfg.DEVICE}")
    print("=" * 50)

    print("\n>>> STEP 0: Running Angle Error Evaluation...")
    try:
        run_angle_metrics()
    except Exception as e:
        print(f"[ERROR] Angle Metrics evaluation failed: {e}")

    print("\n>>> STEP 1: Running Pixel-Wise Metrics (PW)...")
    try:
        run_pw_dice()
        run_pw_auc()
    except Exception as e:
        print(f"[ERROR] PW Metrics evaluation failed: {e}")

    print("\n>>> STEP 2: Running Connected Components Metrics (CC)...")
    try:
        run_cc_metrics()
        run_cc_dice()
    except Exception as e:
        print(f"[ERROR] CC Metrics evaluation failed: {e}")

    print("\n>>> STEP 3: Generating Visual Samples (Standard & CC)...")
    try:
        run_standard_visuals()
        run_cc_visuals()
    except Exception as e:
        print(f"[ERROR] Visuals generation failed: {e}")

    print("\n" + "=" * 50)
    print(" EVALUATION PIPELINE COMPLETELY FINISHED!")
    print(f" PW Results: {cfg.EVAL_PW_DIR}")
    print(f" CC Results: {cfg.EVAL_CC_DIR}")
    print(f" Visuals:    {cfg.EVAL_VISUALS_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
