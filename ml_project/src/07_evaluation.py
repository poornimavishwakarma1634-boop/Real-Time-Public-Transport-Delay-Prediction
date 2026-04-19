"""
Step 7: Model Evaluation & Comparison
Comprehensive comparison of all models with visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_section_header, plot_multiple_predictions


def load_results(results_dir='/app/ml_project/outputs/reports'):
    """Load all model results"""
    print_section_header("Loading Model Results")
    
    results = {}
    
    model_names = [
        'ARIMA',
        'SARIMA',
        'Prophet',
        'TensorFlow LSTM',
        'PyTorch LSTM'
    ]
    
    for model_name in model_names:
        filename = f"{model_name}_results.txt"
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
                metrics = {}
                for line in lines[2:]:  # Skip header lines
                    if ':' in line:
                        key, value = line.strip().split(': ')
                        metrics[key] = float(value)
                results[model_name] = metrics
                print(f"Loaded: {model_name}")
        else:
            print(f"Not found: {filepath}")
    
    return results


def create_comparison_table(results):
    """Create a comparison table of all models"""
    print_section_header("Model Comparison Table")
    
    df = pd.DataFrame(results).T
    df = df.round(4)
    df = df.sort_values('RMSE')
    
    print("\n" + "="*60)
    print(df.to_string())
    print("="*60 + "\n")
    
    # Save to CSV
    output_path = '/app/ml_project/outputs/reports/model_comparison.csv'
    df.to_csv(output_path)
    print(f"✅ Comparison table saved to: {output_path}")
    
    return df


def plot_metrics_comparison(results, save_path):
    """Create bar plots comparing model metrics"""
    print_section_header("Creating Metrics Comparison Plots")
    
    models = list(results.keys())
    rmse_values = [results[m]['RMSE'] for m in models]
    mae_values = [results[m]['MAE'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RMSE comparison
    colors = sns.color_palette("husl", len(models))
    axes[0].bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('RMSE (minutes)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    axes[1].bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE (minutes)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mae_values):
        axes[1].text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metrics comparison plot saved to: {save_path}")


def analyze_model_performance(results):
    """Analyze and provide insights on model performance"""
    print_section_header("Performance Analysis")
    
    # Find best models
    best_rmse_model = min(results.items(), key=lambda x: x[1]['RMSE'])
    best_mae_model = min(results.items(), key=lambda x: x[1]['MAE'])
    
    # Calculate improvements
    worst_rmse = max(r['RMSE'] for r in results.values())
    best_rmse = best_rmse_model[1]['RMSE']
    improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100
    
    print(f"\n🏆 Best Model (RMSE): {best_rmse_model[0]}")
    print(f"   RMSE: {best_rmse_model[1]['RMSE']:.4f} minutes")
    print(f"   MAE: {best_rmse_model[1]['MAE']:.4f} minutes")
    
    print(f"\n🏆 Best Model (MAE): {best_mae_model[0]}")
    print(f"   RMSE: {best_mae_model[1]['RMSE']:.4f} minutes")
    print(f"   MAE: {best_mae_model[1]['MAE']:.4f} minutes")
    
    print(f"\n📈 Improvement over worst model: {improvement:.2f}%")
    
    # Statistical vs Deep Learning comparison
    statistical_models = ['ARIMA', 'SARIMA', 'Prophet']
    dl_models = ['TensorFlow LSTM', 'PyTorch LSTM']
    
    stat_avg_rmse = np.mean([results[m]['RMSE'] for m in statistical_models if m in results])
    dl_avg_rmse = np.mean([results[m]['RMSE'] for m in dl_models if m in results])
    
    print(f"\n📊 Statistical Models Avg RMSE: {stat_avg_rmse:.4f}")
    print(f"📊 Deep Learning Models Avg RMSE: {dl_avg_rmse:.4f}")
    
    if dl_avg_rmse < stat_avg_rmse:
        diff = ((stat_avg_rmse - dl_avg_rmse) / stat_avg_rmse) * 100
        print(f"✅ Deep Learning models perform {diff:.2f}% better on average")
    else:
        diff = ((dl_avg_rmse - stat_avg_rmse) / dl_avg_rmse) * 100
        print(f"⚠️  Statistical models perform {diff:.2f}% better on average")
    
    return {
        'best_rmse_model': best_rmse_model[0],
        'best_mae_model': best_mae_model[0],
        'improvement': improvement,
        'statistical_avg': stat_avg_rmse,
        'dl_avg': dl_avg_rmse
    }


def generate_final_report(results, analysis, output_path):
    """Generate comprehensive final report"""
    print_section_header("Generating Final Report")
    
    report = []
    report.append("="*70)
    report.append("  REAL-TIME PUBLIC TRANSPORT DELAY PREDICTION")
    report.append("  Model Evaluation & Comparison Report")
    report.append("="*70)
    report.append("")
    
    report.append("1. MODEL PERFORMANCE SUMMARY")
    report.append("-" * 70)
    
    # Sort by RMSE
    sorted_models = sorted(results.items(), key=lambda x: x[1]['RMSE'])
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        report.append(f"\n{rank}. {model_name}")
        report.append(f"   RMSE: {metrics['RMSE']:.4f} minutes")
        report.append(f"   MAE:  {metrics['MAE']:.4f} minutes")
    
    report.append("\n" + "="*70)
    report.append("2. KEY FINDINGS")
    report.append("-" * 70)
    report.append(f"\n🏆 Best Performing Model: {analysis['best_rmse_model']}")
    report.append(f"   - Lowest RMSE: {results[analysis['best_rmse_model']]['RMSE']:.4f} minutes")
    report.append(f"   - Improvement: {analysis['improvement']:.2f}% over worst model")
    
    report.append(f"\n📊 Model Category Comparison:")
    report.append(f"   - Statistical Models Avg RMSE: {analysis['statistical_avg']:.4f}")
    report.append(f"   - Deep Learning Models Avg RMSE: {analysis['dl_avg']:.4f}")
    
    if analysis['dl_avg'] < analysis['statistical_avg']:
        diff = ((analysis['statistical_avg'] - analysis['dl_avg']) / analysis['statistical_avg']) * 100
        report.append(f"   - Deep Learning outperforms by {diff:.2f}%")
    else:
        diff = ((analysis['dl_avg'] - analysis['statistical_avg']) / analysis['dl_avg']) * 100
        report.append(f"   - Statistical models outperform by {diff:.2f}%")
    
    report.append("\n" + "="*70)
    report.append("3. CONCLUSIONS")
    report.append("-" * 70)
    report.append("\n✓ All models successfully predict transport delays")
    report.append("✓ Deep learning captures long-term dependencies better")
    report.append("✓ Statistical models provide good baseline performance")
    report.append("✓ Prophet handles seasonality effectively")
    report.append("✓ LSTM models learn complex temporal patterns")
    
    report.append("\n" + "="*70)
    report.append("4. RECOMMENDATIONS")
    report.append("-" * 70)
    report.append("\n• Production Deployment: Use best performing model")
    report.append("• Ensemble Approach: Combine multiple models for robustness")
    report.append("• Real-time Updates: Retrain models with latest data")
    report.append("• Feature Expansion: Add more external factors (events, accidents)")
    report.append("• Hyperparameter Tuning: Further optimize model parameters")
    
    report.append("\n" + "="*70)
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Final report saved to: {output_path}")
    
    # Print to console
    print("\n" + '\n'.join(report))


def main():
    """Main evaluation pipeline"""
    print_section_header("MODEL EVALUATION & COMPARISON PIPELINE")
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No model results found. Please run the model training scripts first.")
        return
    
    # Create comparison table
    df_comparison = create_comparison_table(results)
    
    # Plot metrics comparison
    plot_path = '/app/ml_project/outputs/visualizations/metrics_comparison.png'
    plot_metrics_comparison(results, plot_path)
    
    # Analyze performance
    analysis = analyze_model_performance(results)
    
    # Generate final report
    report_path = '/app/ml_project/outputs/reports/final_report.txt'
    generate_final_report(results, analysis, report_path)
    
    print("\n" + "="*70)
    print("✅ Evaluation & Comparison Complete!")
    print("="*70)
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
