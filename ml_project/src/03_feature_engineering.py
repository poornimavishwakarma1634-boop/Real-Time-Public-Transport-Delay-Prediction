"""
Step 3: Feature Engineering for Bangalore & Kalaburagi Transport
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_section_header


def load_processed_data(filepath):
    print_section_header("Loading Processed Data")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} records")
    return df


def create_temporal_features(df):
    print_section_header("Creating Temporal Features")
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_morning_rush'] = ((df['hour'] >= 8) & (df['hour'] < 10)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] < 20)).astype(int)
    df['is_rush_hour'] = ((df['is_morning_rush'] == 1) | (df['is_evening_rush'] == 1)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    print("Created temporal features: hour, day_of_week, is_weekend, is_rush_hour, cyclical encodings")
    return df


def create_lag_features(df, target_col='delay_minutes', n_lags=8):
    """8 lags * 15min = 2 hours of history"""
    print_section_header(f"Creating Lag Features (n={n_lags})")
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df[target_col].shift(i)
    print(f"Created {n_lags} lag features (lag_1 to lag_{n_lags})")
    return df


def create_rolling_features(df, target_col='delay_minutes', windows=[4, 8, 16]):
    """4=1hr, 8=2hr, 16=4hr with 15-min intervals"""
    print_section_header("Creating Rolling Window Features")
    df = df.copy()
    for window in windows:
        mins = window * 15
        df[f'rolling_mean_{mins}min'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{mins}min'] = df[target_col].rolling(window=window, min_periods=1).std()
    print(f"Created rolling features for windows: {[w*15 for w in windows]} minutes")
    return df


def add_weather_features(df):
    print_section_header("Adding Karnataka Weather Features")
    df = df.copy()
    np.random.seed(42)
    month = df.index.month
    # Karnataka monsoon: June-September
    rain_prob = np.where(month.isin([6, 7, 8, 9]), 0.5, 
                np.where(month.isin([10, 11]), 0.3, 0.15))
    df['weather_rain'] = (np.random.random(len(df)) < rain_prob).astype(int)
    print(f"Added weather features: {int(df['weather_rain'].sum())} rainy periods")
    return df


def drop_na_rows(df):
    print_section_header("Cleaning Dataset")
    initial = len(df)
    df = df.dropna()
    print(f"Dropped {initial - len(df)} rows with missing values")
    print(f"Final dataset: {len(df)} records")
    return df


def save_features(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"\nFeatures saved to: {output_path}")
    print(f"Total features: {len(df.columns)}")


def main():
    print_section_header("FEATURE ENGINEERING PIPELINE")
    data_path = '/app/ml_project/data/processed/processed_delays.csv'
    df = load_processed_data(data_path)
    df = create_temporal_features(df)
    df = create_lag_features(df, n_lags=8)
    df = create_rolling_features(df, windows=[4, 8, 16])
    df = add_weather_features(df)
    df = drop_na_rows(df)
    save_features(df, '/app/ml_project/data/processed/features.csv')
    print("\n" + "="*70)
    print("Feature Engineering Complete!")
    print("="*70)
    return df


if __name__ == "__main__":
    df = main()
