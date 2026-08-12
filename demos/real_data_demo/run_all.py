"""
Module: run_all.py
Provides functionality for run_all.
"""

import os
import subprocess
import sys

def main():
    """Functionality for main."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    tube_script = os.path.join(base_dir, "tube_pkg", "run_tube.py")
    consolidation_script = os.path.join(base_dir, "consolidation_pkg", "run_consolidation.py")
    
    print("=" * 60)
    print("STARTING FULL PIPELINE BATCH INFERENCE (ALL PATIENTS)")
    print("=" * 60)
    
    # 1. Run Tube
    print("\n>>> RUNNING TUBE MODEL INFERENCE <<<")
    try:
        subprocess.run([sys.executable, tube_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Tube inference failed with exit code {e.returncode}")
        return
        
    # 2. Run Consolidation
    print("\n>>> RUNNING CONSOLIDATION MODEL INFERENCE <<<")
    try:
        subprocess.run([sys.executable, consolidation_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Consolidation inference failed with exit code {e.returncode}")
        return

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE. ALL OUTPUTS SAVED.")
    print("=" * 60)

if __name__ == "__main__":
    main()
