"""
Step 2: Time-Series Preprocessing
Handles resampling, imputation, and stationarity checks
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_section_header


def load_raw_data(filepath):
    print_section_header("Loading Raw Data")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def resample_data(df, freq='15min'):
    """Resample time-series data to fixed intervals"""
    print_section_header(f"Resampling Data to {freq} intervals")
    df = df.set_index('timestamp')
    resampled = df['delay_minutes'].resample(freq).mean()
    print(f"Resampled to {len(resampled)} records")
    return resampled


def handle_missing_values(series, method='interpolate'):
    print_section_header("Handling Missing Values")
    missing_count = int(series.isna().sum())
    total = len(series)
    pct = (missing_count / total) * 100
    print(f"Missing values: {missing_count} ({pct:.2f}%)")
    
    if method == 'interpolate':
        filled = series.interpolate(method='linear')
    elif method == 'forward_fill':
        filled = series.ffill()
    else:
        filled = series.bfill()
    
    remaining = int(filled.isna().sum())
    if remaining > 0:
        filled = filled.bfill().ffill()
    
    print(f"Imputation complete using {method}")
    print(f"Remaining missing: {int(filled.isna().sum())}")
    return filled


def check_stationarity(series, name='Series'):
    print_section_header(f"Stationarity Test: {name}")
    result = adfuller(series.dropna(), autolag='AIC')
    
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"P-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    is_stationary = result[1] <= 0.05
    if is_stationary:
        print(f"\nSeries is STATIONARY (p-value <= 0.05)")
    else:
        print(f"\nSeries is NON-STATIONARY (p-value > 0.05)")
    
    return is_stationary, result


def decompose_series(series, model='additive', period=96):
    """period=96 for daily seasonality with 15-min intervals (24*60/15)"""
    print_section_header("Time-Series Decomposition")
    try:
        decomposition = seasonal_decompose(
            series.dropna(), model=model, period=period, extrapolate_trend='freq'
        )
        fig = decomposition.plot()
        fig.set_size_inches(14, 10)
        plt.tight_layout()
        output_path = '/app/ml_project/outputs/visualizations/time_series_decomposition.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Decomposition complete ({model} model)")
        print(f"Plot saved to: {output_path}")
        return decomposition
    except Exception as e:
        print(f"Decomposition failed: {e}")
        return None


def save_processed_data(series, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({'timestamp': series.index, 'delay_minutes': series.values})
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    print_section_header("TIME-SERIES PREPROCESSING PIPELINE")
    
    raw_data_path = '/app/ml_project/data/raw/gtfs_data.csv'
    df = load_raw_data(raw_data_path)
    resampled = resample_data(df, freq='15min')
    filled_series = handle_missing_values(resampled, method='interpolate')
    is_stationary, adf_result = check_stationarity(filled_series, 'Delay Series')
    decomposition = decompose_series(filled_series, model='additive', period=96)
    
    output_path = '/app/ml_project/data/processed/processed_delays.csv'
    save_processed_data(filled_series, output_path)
    
    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    return filled_series, is_stationary


if __name__ == "__main__":
    series, is_stationary = main()
