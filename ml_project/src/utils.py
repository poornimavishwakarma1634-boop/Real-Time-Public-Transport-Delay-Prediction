"""
Utility functions for ML project
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate RMSE and MAE metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'RMSE': rmse,
        'MAE': mae
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str, save_path: str = None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label='Actual', linewidth=2, alpha=0.7)
    plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Delay (minutes)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    plt.close()


def plot_multiple_predictions(predictions_dict: dict, y_true: np.ndarray, 
                              save_path: str = None):
    """Plot multiple model predictions together"""
    plt.figure(figsize=(16, 8))
    plt.plot(y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        plt.plot(y_pred, label=model_name, linewidth=2, alpha=0.7, 
                color=colors[idx % len(colors)])
    
    plt.title('Model Comparison: Actual vs Predicted Delays', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=13)
    plt.ylabel('Delay (minutes)', fontsize=13)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {save_path}")
    plt.close()


def create_sequences(data: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training (sliding window approach)
    
    Args:
        data: Time series data
        n_steps: Number of time steps to look back
        
    Returns:
        X: Input sequences
        y: Target values
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


def save_model_results(model_name: str, metrics: dict, save_dir: str):
    """Save model metrics to file"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{model_name}_results.txt")
    
    with open(filepath, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"✅ Results saved to {filepath}")


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")
