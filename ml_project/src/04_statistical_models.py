"""
Step 4: Statistical Models (ARIMA, SARIMA, Prophet)
Baseline time-series forecasting for Bangalore & Kalaburagi transport
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import pickle
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_section_header, calculate_metrics, plot_predictions, save_model_results


def load_data(filepath):
    print_section_header("Loading Data for Statistical Modeling")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} records")
    return df


def train_test_split(series, test_size=0.2):
    print_section_header("Train-Test Split")
    split_idx = int(len(series) * (1 - test_size))
    train = series[:split_idx]
    test = series[split_idx:]
    print(f"Train: {len(train)} | Test: {len(test)}")
    return train, test


def train_arima(train, order=(5, 1, 2)):
    print_section_header(f"Training ARIMA {order}")
    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
        print(f"ARIMA trained. AIC: {fitted.aic:.2f}")
        return fitted
    except Exception as e:
        print(f"ARIMA failed: {e}")
        return None


def train_sarima(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 96)):
    """seasonal_order s=96 for daily seasonality (15-min intervals, 24*4=96)"""
    print_section_header(f"Training SARIMA {order} x {seasonal_order}")
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=50)
        print(f"SARIMA trained. AIC: {fitted.aic:.2f}")
        return fitted
    except Exception as e:
        print(f"SARIMA failed: {e}")
        # Fallback
        try:
            model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                           enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=30)
            print(f"SARIMA fallback trained. AIC: {fitted.aic:.2f}")
            return fitted
        except Exception as e2:
            print(f"SARIMA fallback also failed: {e2}")
            return None


def train_prophet(train_series):
    print_section_header("Training Prophet Model")
    try:
        df_prophet = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05
        )
        model.fit(df_prophet)
        print("Prophet trained successfully")
        return model
    except Exception as e:
        print(f"Prophet failed: {e}")
        return None


def evaluate_model(model, train, test, model_name, model_type='arima'):
    print_section_header(f"Evaluating {model_name}")
    try:
        if model_type == 'prophet':
            future = pd.DataFrame({'ds': test.index})
            forecast = model.predict(future)
            predictions = forecast['yhat'].values
        else:
            predictions = np.array(model.forecast(steps=len(test)))
        
        metrics = calculate_metrics(test.values, predictions)
        print(f"RMSE: {metrics['RMSE']:.4f} min | MAE: {metrics['MAE']:.4f} min")
        
        plot_path = f'/app/ml_project/outputs/visualizations/{model_name.lower().replace(" ", "_")}_predictions.png'
        plot_predictions(test.values, predictions, f'{model_name}: Actual vs Predicted', save_path=plot_path)
        save_model_results(model_name, metrics, '/app/ml_project/outputs/reports')
        
        return predictions, metrics
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None, None


def save_model(model, model_name, save_dir='/app/ml_project/models/statistical'):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def main():
    print_section_header("STATISTICAL MODELS PIPELINE")
    
    df = load_data('/app/ml_project/data/processed/processed_delays.csv')
    series = df['delay_minutes']
    train, test = train_test_split(series, test_size=0.2)
    
    results = {}
    
    # ARIMA
    arima = train_arima(train, order=(5, 1, 2))
    if arima:
        preds, metrics = evaluate_model(arima, train, test, 'ARIMA', 'arima')
        if metrics:
            results['ARIMA'] = {'predictions': preds.tolist(), 'metrics': metrics}
            save_model(arima, 'ARIMA')
    
    # SARIMA (use smaller sample for speed)
    train_small = train[-2000:] if len(train) > 2000 else train
    sarima = train_sarima(train_small, order=(1, 1, 1), seasonal_order=(1, 0, 1, 96))
    if sarima:
        preds, metrics = evaluate_model(sarima, train_small, test, 'SARIMA', 'arima')
        if metrics:
            results['SARIMA'] = {'predictions': preds.tolist(), 'metrics': metrics}
            save_model(sarima, 'SARIMA')
    
    # Prophet
    prophet_model = train_prophet(train)
    if prophet_model:
        preds, metrics = evaluate_model(prophet_model, train, test, 'Prophet', 'prophet')
        if metrics:
            results['Prophet'] = {'predictions': preds.tolist(), 'metrics': metrics}
            save_model(prophet_model, 'Prophet')
    
    # Summary
    print("\n" + "="*70)
    print(" STATISTICAL MODELS SUMMARY")
    print("="*70)
    for name, r in results.items():
        print(f"{name}: RMSE={r['metrics']['RMSE']:.4f}, MAE={r['metrics']['MAE']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()
