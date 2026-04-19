"""
Master Script: Run Complete ML Pipeline
Executes all steps from data ingestion to final evaluation
"""
import sys
import os
import time
import importlib.util

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from utils import print_section_header


def run_pipeline():
    """Execute complete ML pipeline"""
    
    print("\n" + "="*70)
    print("  REAL-TIME PUBLIC TRANSPORT DELAY PREDICTION")
    print("  Complete ML Pipeline Execution")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    try:
        # Step 1: Data Ingestion
        print_section_header("STEP 1/7: Data Ingestion")
        import importlib.util
        spec = importlib.util.spec_from_file_location("step_01", "/app/ml_project/src/01_data_ingestion.py")
        step_01 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_01)
        step_01.main()
        print("✅ Step 1 Complete\n")
        
    except Exception as e:
        print(f"❌ Step 1 Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Step 2: Preprocessing
        print_section_header("STEP 2/7: Preprocessing")
        spec = importlib.util.spec_from_file_location("step_02", "/app/ml_project/src/02_preprocessing.py")
        step_02 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_02)
        step_02.main()
        print("✅ Step 2 Complete\n")
        
    except Exception as e:
        print(f"❌ Step 2 Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Step 3: Feature Engineering
        print_section_header("STEP 3/7: Feature Engineering")
        spec = importlib.util.spec_from_file_location("step_03", "/app/ml_project/src/03_feature_engineering.py")
        step_03 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_03)
        step_03.main()
        print("✅ Step 3 Complete\n")
        
    except Exception as e:
        print(f"❌ Step 3 Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Step 4: Statistical Models
        print_section_header("STEP 4/7: Statistical Models")
        spec = importlib.util.spec_from_file_location("step_04", "/app/ml_project/src/04_statistical_models.py")
        step_04 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_04)
        step_04.main()
        print("✅ Step 4 Complete\n")
        
    except Exception as e:
        print(f"❌ Step 4 Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Step 5: TensorFlow LSTM
        print_section_header("STEP 5/7: TensorFlow LSTM")
        spec = importlib.util.spec_from_file_location("step_05", "/app/ml_project/src/05_tensorflow_lstm.py")
        step_05 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_05)
        step_05.main()
        print("✅ Step 5 Complete\n")
        
    except Exception as e:
        print(f"❌ Step 5 Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Step 6: PyTorch LSTM
        print_section_header("STEP 6/7: PyTorch LSTM")
        spec = importlib.util.spec_from_file_location("step_06", "/app/ml_project/src/06_pytorch_lstm.py")
        step_06 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_06)
        step_06.main()
        print("✅ Step 6 Complete\n")
        
    except Exception as e:
        print(f"❌ Step 6 Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Step 7: Evaluation & Comparison
        print_section_header("STEP 7/7: Evaluation & Comparison")
        spec = importlib.util.spec_from_file_location("step_07", "/app/ml_project/src/07_evaluation.py")
        step_07 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_07)
        step_07.main()
        print("✅ Step 7 Complete\n")
        
    except Exception as e:
        print(f"❌ Step 7 Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("  🎉 COMPLETE ML PIPELINE EXECUTED SUCCESSFULLY!")
    print("="*70)
    print(f"\n⏱️  Total Execution Time: {total_time/60:.2f} minutes")
    print(f"\n📁 Outputs saved to:")
    print("   - Models: /app/ml_project/models/")
    print("   - Visualizations: /app/ml_project/outputs/visualizations/")
    print("   - Reports: /app/ml_project/outputs/reports/")
    print("\n📊 Check final_report.txt for complete analysis")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_pipeline()
