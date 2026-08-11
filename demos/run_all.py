import os
import subprocess
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Define outputs directory
    output_dir = os.path.join(base_dir, "output")
    demo1_out = os.path.join(output_dir, "demo1_inference")
    demo2_out = os.path.join(output_dir, "demo2_model_generation")
    demo3_out = os.path.join(output_dir, "demo3_baseline")
    
    os.makedirs(demo1_out, exist_ok=True)
    os.makedirs(demo2_out, exist_ok=True)
    os.makedirs(demo3_out, exist_ok=True)
    
    print("=" * 60)
    print("STARTING ALL DEMOS")
    print(f"Outputs will be saved in: {output_dir}")
    print("=" * 60)
    
    # 1. Real and Synthetic Inference
    print("\n" + "=" * 60)
    print("DEMO 1: Running models on Real & Synthetic Data")
    print("=" * 60)
    
    real_demo_script = os.path.join(base_dir, "real_data_demo", "run_all.py")
    synthetic_demo_script = os.path.join(base_dir, "synthetic_data_demo", "run_all.py")
    
    if os.path.exists(real_demo_script):
        print(">>> Real Data Inference <<<")
        subprocess.run([sys.executable, real_demo_script], check=True)
    else:
        print("Real data demo script not found.")
        
    if os.path.exists(synthetic_demo_script):
        print(">>> Synthetic Data Inference <<<")
        subprocess.run([sys.executable, synthetic_demo_script], check=True)
    else:
        print("Synthetic data demo script not found.")
        
    # 2. Main Model Pipeline (Datagen -> Train -> Eval)
    print("\n" + "=" * 60)
    print("DEMO 2: Main Model Generation Pipeline")
    print("=" * 60)
    
    main_model_script = os.path.join(project_root, "main_model", "run_all.py")
    if os.path.exists(main_model_script):
        # We would ideally pass the output directory or input CTs here if needed
        subprocess.run([sys.executable, main_model_script], check=True)
    else:
        print("Main model run_all.py not found.")
        
    # 3. Baseline Model Pipeline
    print("\n" + "=" * 60)
    print("DEMO 3: Baseline Model Pipeline")
    print("=" * 60)
    
    baseline_script = os.path.join(project_root, "baseline_model", "run_all.py")
    if os.path.exists(baseline_script):
        subprocess.run([sys.executable, baseline_script], check=True)
    else:
        print("Baseline model run_all.py not found.")

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE.")
    print("=" * 60)

if __name__ == "__main__":
    main()
