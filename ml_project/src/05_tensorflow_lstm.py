"""
Step 5: TensorFlow LSTM Model for transport delay prediction
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_section_header, calculate_metrics, plot_predictions, save_model_results, create_sequences


def load_data(filepath):
    print_section_header("Loading Data for TensorFlow LSTM")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} records")
    return df


def prepare_lstm_data(series, n_steps=16, test_size=0.2):
    """n_steps=16 (4 hours of history with 15-min intervals)"""
    print_section_header(f"Preparing LSTM Data (n_steps={n_steps})")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = create_sequences(scaled, n_steps)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Shape: {X.shape}")
    return X_train, X_test, y_train, y_test, scaler


def build_model(n_steps, n_features=1):
    print_section_header("Building TensorFlow LSTM")
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.summary()
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=64):
    print_section_header("Training TensorFlow LSTM")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=1
    )
    print(f"Training Complete! Final val_loss: {history.history['val_loss'][-1]:.6f}")
    return model, history


def evaluate(model, X_test, y_test, scaler, model_name='TensorFlow LSTM'):
    print_section_header(f"Evaluating {model_name}")
    preds_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    metrics = calculate_metrics(y_true.flatten(), predictions.flatten())
    print(f"RMSE: {metrics['RMSE']:.4f} min | MAE: {metrics['MAE']:.4f} min")
    
    plot_path = f'/app/ml_project/outputs/visualizations/{model_name.lower().replace(" ", "_")}_predictions.png'
    plot_predictions(y_true.flatten(), predictions.flatten(), f'{model_name}: Actual vs Predicted', save_path=plot_path)
    save_model_results(model_name, metrics, '/app/ml_project/outputs/reports')
    
    return predictions.flatten(), metrics


def save_lstm(model, scaler, model_name, save_dir='/app/ml_project/models/tensorflow'):
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}.keras'))
    with open(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved to: {save_dir}")


def main():
    print_section_header("TENSORFLOW LSTM PIPELINE")
    df = load_data('/app/ml_project/data/processed/processed_delays.csv')
    n_steps = 16
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df['delay_minutes'], n_steps=n_steps)
    model = build_model(n_steps=n_steps)
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=64)
    predictions, metrics = evaluate(model, X_test, y_test, scaler)
    save_lstm(model, scaler, 'TensorFlow LSTM')
    print("\n" + "="*70)
    print(f"TensorFlow LSTM Complete! RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    print("="*70)
    return model, predictions, metrics


if __name__ == "__main__":
    model, predictions, metrics = main()
