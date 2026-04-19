"""
Model Retraining Scheduler
Periodically retrains Prophet and LSTM models with the latest data,
and generates new live feed observations.
"""
import os
import sys
import asyncio
import logging
import threading
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Add ml src to path
sys.path.insert(0, '/app/ml_project/src')

MODELS_DIR = Path('/app/ml_project/models')
DATA_DIR = Path('/app/ml_project/data')
REPORTS_DIR = Path('/app/ml_project/outputs/reports')


def _retrain_prophet(series: pd.Series) -> dict:
    """Retrain Prophet on the full series and save"""
    from prophet import Prophet
    import warnings
    warnings.filterwarnings('ignore')

    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    m = Prophet(
        yearly_seasonality=False, weekly_seasonality=True,
        daily_seasonality=True, seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    m.fit(df)

    save_dir = MODELS_DIR / 'statistical'
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'prophet.pkl', 'wb') as f:
        pickle.dump(m, f)

    # Evaluate on last 20%
    split = int(len(series) * 0.8)
    test = series.iloc[split:]
    future = pd.DataFrame({'ds': test.index})
    fc = m.predict(future)
    rmse = np.sqrt(((test.values - fc['yhat'].values) ** 2).mean())
    mae = np.abs(test.values - fc['yhat'].values).mean()
    logger.info(f"Prophet retrained — RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return {'model': 'Prophet', 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}


def _retrain_tf_lstm(series: pd.Series, n_steps=16, epochs=15) -> dict:
    """Retrain TensorFlow LSTM and save"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from utils import create_sequences

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = create_sequences(scaled, n_steps)
    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs,
              batch_size=64, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

    preds = scaler.inverse_transform(model.predict(Xte, verbose=0))
    ytrue = scaler.inverse_transform(yte)
    rmse = np.sqrt(((ytrue.flatten() - preds.flatten()) ** 2).mean())
    mae = np.abs(ytrue.flatten() - preds.flatten()).mean()

    save_dir = MODELS_DIR / 'tensorflow'
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / 'tensorflow_lstm.keras'))
    with open(save_dir / 'tensorflow_lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"TF LSTM retrained — RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return {'model': 'TensorFlow LSTM', 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}


def _retrain_pytorch_lstm(series: pd.Series, n_steps=16, epochs=15) -> dict:
    """Retrain PyTorch LSTM and save"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    from utils import create_sequences

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = create_sequences(scaled, n_steps)
    split = int(len(X) * 0.8)

    Xtr = torch.FloatTensor(X[:split])
    ytr = torch.FloatTensor(y[:split])
    Xte = torch.FloatTensor(X[split:])
    yte_np = y[split:]

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, 2, dropout=0.2, batch_first=True)
            self.fc1 = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.2)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc2(self.drop(self.relu(self.fc1(out))))

    model = LSTMModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xte)
            vl = loss_fn(val_pred, torch.FloatTensor(yte_np)).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds = scaler.inverse_transform(model(Xte).numpy())
    ytrue = scaler.inverse_transform(yte_np)
    rmse = np.sqrt(((ytrue.flatten() - preds.flatten()) ** 2).mean())
    mae = np.abs(ytrue.flatten() - preds.flatten()).mean()

    save_dir = MODELS_DIR / 'pytorch'
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_dir / 'pytorch_lstm.pth'))
    with open(save_dir / 'pytorch_lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"PyTorch LSTM retrained — RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return {'model': 'PyTorch LSTM', 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}


def _update_comparison(results: list):
    """Write updated model_comparison.csv"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df = df.set_index('model')
    df.to_csv(REPORTS_DIR / 'model_comparison.csv')
    logger.info("Updated model_comparison.csv")


def retrain_all():
    """Run full retraining cycle (blocking)"""
    logger.info("=== Retraining cycle started ===")
    csv = DATA_DIR / 'processed' / 'processed_delays.csv'
    if not csv.exists():
        logger.warning("No processed data found, skipping retrain")
        return None

    df = pd.read_csv(csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    series = df['delay_minutes']

    results = []

    # Prophet
    try:
        results.append(_retrain_prophet(series))
    except Exception as e:
        logger.error(f"Prophet retrain failed: {e}")

    # TF LSTM
    try:
        results.append(_retrain_tf_lstm(series))
    except Exception as e:
        logger.error(f"TF LSTM retrain failed: {e}")

    # PyTorch LSTM
    try:
        results.append(_retrain_pytorch_lstm(series))
    except Exception as e:
        logger.error(f"PyTorch LSTM retrain failed: {e}")

    if results:
        _update_comparison(results)

    logger.info(f"=== Retraining complete: {len(results)} models ===")
    return results


# ── Scheduler Thread ─────────────────────────────────────────────────────

class RetrainingScheduler:
    """Background thread that periodically retrains models and generates feed data"""

    def __init__(self, db, live_feed, ml_svc, interval_minutes=60):
        self.db = db
        self.live_feed = live_feed
        self.ml_svc = ml_svc
        self.interval = interval_minutes * 60  # seconds
        self.feed_interval = 15 * 60  # generate feed every 15 min
        self._thread = None
        self._stop_event = threading.Event()
        self.last_retrain = None
        self.last_feed_gen = None
        self.retrain_count = 0
        self.status = "idle"

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Scheduler started (retrain every {self.interval//60}m, feed every {self.feed_interval//60}m)")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.status = "stopped"

    def _run(self):
        self.status = "running"
        last_retrain_ts = 0
        last_feed_ts = 0

        while not self._stop_event.is_set():
            now = time.time()

            # Generate live feed data every 15 minutes
            if now - last_feed_ts >= self.feed_interval:
                try:
                    self.live_feed.generate_batch_sync()
                    self.last_feed_gen = datetime.now(timezone.utc)
                    last_feed_ts = now
                    logger.info("Live feed batch generated")
                except Exception as e:
                    logger.error(f"Feed generation error: {e}")

            # Retrain models periodically
            if now - last_retrain_ts >= self.interval:
                try:
                    self.status = "retraining"
                    results = retrain_all()
                    if results:
                        self.ml_svc.reload_models()
                        self.retrain_count += 1
                        self.last_retrain = datetime.now(timezone.utc)
                    self.status = "running"
                    last_retrain_ts = now
                except Exception as e:
                    logger.error(f"Retrain cycle error: {e}")
                    self.status = "running"

            self._stop_event.wait(30)  # check every 30s

    def get_status(self):
        return {
            "status": self.status,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "last_feed_gen": self.last_feed_gen.isoformat() if self.last_feed_gen else None,
            "retrain_count": self.retrain_count,
            "interval_minutes": self.interval // 60,
            "feed_interval_minutes": self.feed_interval // 60,
            "models_loaded": self.ml_svc.get_available_models() if self.ml_svc else []
        }
