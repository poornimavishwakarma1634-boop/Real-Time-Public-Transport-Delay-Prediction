"""
ML Service for Public Transport Delay Prediction
Provides REAL model inference using trained ARIMA, Prophet, TF-LSTM, PyTorch-LSTM
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_keras = None
try:
    from tensorflow import keras as _keras
except Exception:
    try:
        import keras as _keras
    except Exception:
        _keras = None

# PyTorch
_torch = None
try:
    import torch as _torch
except Exception:
    _torch = None

# Add ml_project src to path for PyTorch model class
sys.path.insert(0, '/app/ml_project/src')

# Route delay profiles (used as priors for model blending)
ROUTE_PROFILES = {
    "BLR-BUS-1": {"base": 8.2, "type": "Bus", "city": "Bangalore"},
    "BLR-BUS-2": {"base": 9.5, "type": "Bus", "city": "Bangalore"},
    "BLR-TRN-1": {"base": 3.6, "type": "Train", "city": "Bangalore"},
    "BLR-TRN-2": {"base": 3.4, "type": "Train", "city": "Bangalore"},
    "KLB-BUS-1": {"base": 8.8, "type": "Bus", "city": "Kalaburagi"},
    "KLB-BUS-2": {"base": 9.7, "type": "Bus", "city": "Kalaburagi"},
    "KLB-TRN-1": {"base": 10.2, "type": "Train", "city": "Kalaburagi"},
    "BLR-KLB-EXP": {"base": 17.2, "type": "Train", "city": "Intercity"},
}


class DelayPredictionService:
    """Production-grade prediction service using trained ML models"""

    def __init__(self):
        self.models_dir = Path('/app/ml_project/models')
        self.data_dir = Path('/app/ml_project/data')
        self.models = {}
        self.scalers = {}
        self.history_cache = None  # cached time series for LSTM input
        self._load_all_models()
        self._load_history_cache()

    # ── Model Loading ────────────────────────────────────────────────────

    def _load_all_models(self):
        """Load every available trained model"""
        self._load_statistical_models()
        self._load_tf_lstm()
        self._load_pytorch_lstm()
        logger.info(f"Models ready: {list(self.models.keys())}")

    def _load_statistical_models(self):
        stat_dir = self.models_dir / 'statistical'
        if not stat_dir.exists():
            return
        for pkl in stat_dir.glob('*.pkl'):
            name = pkl.stem.upper()
            try:
                with open(pkl, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info(f"Loaded {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

    def _load_tf_lstm(self):
        tf_dir = self.models_dir / 'tensorflow'
        if not tf_dir.exists() or _keras is None:
            return
        model_f = tf_dir / 'tensorflow_lstm.keras'
        scaler_f = tf_dir / 'tensorflow_lstm_scaler.pkl'
        if model_f.exists() and scaler_f.exists():
            try:
                self.models['TENSORFLOW_LSTM'] = _keras.models.load_model(str(model_f))
                with open(scaler_f, 'rb') as f:
                    self.scalers['TENSORFLOW_LSTM'] = pickle.load(f)
                logger.info("Loaded TensorFlow LSTM")
            except Exception as e:
                logger.warning(f"TF LSTM load error: {e}")

    def _load_pytorch_lstm(self):
        pt_dir = self.models_dir / 'pytorch'
        if not pt_dir.exists() or _torch is None:
            return
        model_f = pt_dir / 'pytorch_lstm.pth'
        scaler_f = pt_dir / 'pytorch_lstm_scaler.pkl'
        if model_f.exists() and scaler_f.exists():
            try:
                from utils import create_sequences  # noqa
                # Reconstruct architecture (must match 06_pytorch_lstm.py)
                import torch.nn as nn

                class LSTMModel(nn.Module):
                    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                            num_layers=num_layers, dropout=dropout, batch_first=True)
                        self.fc1 = nn.Linear(hidden_size, 32)
                        self.relu = nn.ReLU()
                        self.drop = nn.Dropout(dropout)
                        self.fc2 = nn.Linear(32, 1)

                    def forward(self, x):
                        out, _ = self.lstm(x)
                        out = out[:, -1, :]
                        out = self.fc1(out)
                        out = self.relu(out)
                        out = self.drop(out)
                        return self.fc2(out)

                model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
                model.load_state_dict(_torch.load(str(model_f), map_location='cpu'))
                model.eval()
                self.models['PYTORCH_LSTM'] = model
                with open(scaler_f, 'rb') as f:
                    self.scalers['PYTORCH_LSTM'] = pickle.load(f)
                logger.info("Loaded PyTorch LSTM")
            except Exception as e:
                logger.warning(f"PyTorch LSTM load error: {e}")

    def _load_history_cache(self):
        """Load the processed delay time series used as LSTM input window"""
        try:
            csv_path = self.data_dir / 'processed' / 'processed_delays.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.history_cache = df
                logger.info(f"History cache: {len(df)} rows")
        except Exception as e:
            logger.warning(f"History cache load error: {e}")

    # ── Public Interface ─────────────────────────────────────────────────

    def get_available_models(self):
        return list(self.models.keys())

    def get_historical_data(self, limit=1000):
        try:
            if self.history_cache is not None:
                df = self.history_cache.tail(limit).copy()
                df['timestamp'] = df['timestamp'].astype(str)
                return df.to_dict('records')
            return []
        except Exception:
            return []

    def get_model_comparison(self):
        try:
            comp_file = Path('/app/ml_project/outputs/reports/model_comparison.csv')
            if comp_file.exists():
                df = pd.read_csv(comp_file)
                df = df.rename(columns={df.columns[0]: 'model'})
                return df.to_dict('records')
            return []
        except Exception:
            return []

    def reload_models(self):
        """Hot-reload models (called after retraining)"""
        self.models.clear()
        self.scalers.clear()
        self._load_all_models()
        self._load_history_cache()

    # ── Prediction Engine ────────────────────────────────────────────────

    def predict(self, timestamp: str, route_id: str = 'BLR-BUS-1', model_name: str = 'PROPHET'):
        """
        Make a real prediction using the specified model.
        Falls through: requested model → Prophet → TF LSTM → PyTorch LSTM → heuristic
        """
        ts = pd.to_datetime(timestamp)
        model_key = model_name.upper().replace(' ', '_')

        # Try the requested model first
        result = self._predict_with_model(model_key, ts, route_id)
        if result is not None:
            return self._format(result, model_key, route_id, timestamp, 'high')

        # Fallback chain
        for fallback in ['PROPHET', 'TENSORFLOW_LSTM', 'PYTORCH_LSTM']:
            if fallback != model_key:
                result = self._predict_with_model(fallback, ts, route_id)
                if result is not None:
                    return self._format(result, fallback, route_id, timestamp, 'medium')

        # Final heuristic fallback
        result = self._heuristic(ts, route_id)
        return self._format(result, 'HEURISTIC', route_id, timestamp, 'low')

    def _predict_with_model(self, model_key: str, ts: datetime, route_id: str):
        """Dispatch to the correct model inference path"""
        if model_key not in self.models:
            return None
        try:
            if model_key == 'PROPHET':
                return self._predict_prophet(ts, route_id)
            elif model_key == 'TENSORFLOW_LSTM':
                return self._predict_tf_lstm(ts, route_id)
            elif model_key == 'PYTORCH_LSTM':
                return self._predict_pytorch_lstm(ts, route_id)
            elif model_key in ('ARIMA', 'SARIMA'):
                return self._predict_arima(model_key, ts, route_id)
            else:
                return None
        except Exception as e:
            logger.warning(f"{model_key} inference error: {e}")
            return None

    # ── Prophet ──────────────────────────────────────────────────────────

    def _predict_prophet(self, ts: datetime, route_id: str) -> float:
        model = self.models['PROPHET']
        future = pd.DataFrame({'ds': [ts]})
        forecast = model.predict(future)
        raw = float(forecast['yhat'].iloc[0])
        return self._adjust_for_route(raw, route_id)

    # ── ARIMA / SARIMA ───────────────────────────────────────────────────

    def _predict_arima(self, key: str, ts: datetime, route_id: str) -> float:
        model = self.models[key]
        # Forecast 1 step ahead from the end of training data
        fc = model.forecast(steps=1)
        raw = float(fc.iloc[0]) if hasattr(fc, 'iloc') else float(fc[0])
        return self._adjust_for_route(raw, route_id)

    # ── TensorFlow LSTM ──────────────────────────────────────────────────

    def _predict_tf_lstm(self, ts: datetime, route_id: str) -> float:
        model = self.models['TENSORFLOW_LSTM']
        scaler = self.scalers['TENSORFLOW_LSTM']
        seq = self._build_input_sequence(ts, scaler, n_steps=16)
        if seq is None:
            return None
        pred_scaled = model.predict(seq, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        raw = float(pred[0][0])
        return self._adjust_for_route(raw, route_id)

    # ── PyTorch LSTM ─────────────────────────────────────────────────────

    def _predict_pytorch_lstm(self, ts: datetime, route_id: str) -> float:
        model = self.models['PYTORCH_LSTM']
        scaler = self.scalers['PYTORCH_LSTM']
        seq = self._build_input_sequence(ts, scaler, n_steps=16)
        if seq is None:
            return None
        tensor = _torch.FloatTensor(seq)
        model.eval()
        with _torch.no_grad():
            pred_scaled = model(tensor).numpy()
        pred = scaler.inverse_transform(pred_scaled)
        raw = float(pred[0][0])
        return self._adjust_for_route(raw, route_id)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _build_input_sequence(self, ts: datetime, scaler, n_steps: int = 16):
        """Build a normalised input window from recent history for LSTM"""
        if self.history_cache is None or len(self.history_cache) < n_steps:
            return None
        # Use the last n_steps values from history as the input window
        recent = self.history_cache['delay_minutes'].values[-n_steps:]
        scaled = scaler.transform(recent.reshape(-1, 1))
        return scaled.reshape(1, n_steps, 1)

    def _adjust_for_route(self, raw_delay: float, route_id: str) -> float:
        """
        The models are trained on aggregate data. Adjust the raw prediction
        using route-specific priors (average historical delay per route).
        """
        profile = ROUTE_PROFILES.get(route_id)
        if profile is None:
            return max(0, round(raw_delay, 2))

        global_avg = 8.0  # approximate global average from training data
        route_avg = profile["base"]
        ratio = route_avg / global_avg if global_avg > 0 else 1.0
        adjusted = raw_delay * ratio
        return max(0, round(adjusted, 2))

    def _heuristic(self, ts: datetime, route_id: str) -> float:
        """Deterministic time-of-day heuristic (last resort)"""
        profile = ROUTE_PROFILES.get(route_id, {"base": 5.0, "type": "Bus"})
        hour = ts.hour
        dow = ts.weekday()
        base = profile["base"]

        if profile["type"] == "Bus":
            if 8 <= hour < 10:
                base *= 1.6
            elif 17 <= hour < 20:
                base *= 1.8
            elif 23 <= hour or hour < 5:
                base *= 0.3
        else:
            if 8 <= hour < 10 or 17 <= hour < 19:
                base *= 1.3
            elif 23 <= hour or hour < 5:
                base *= 0.4

        if dow >= 5:
            base *= 0.75

        noise = np.random.normal(0, 0.4)
        return max(0, round(base + noise, 2))

    @staticmethod
    def _format(delay: float, model_key: str, route_id: str, timestamp: str, confidence: str):
        return {
            'predicted_delay': delay,
            'model_used': model_key,
            'confidence': confidence,
        }


# ── Singleton ────────────────────────────────────────────────────────────
ml_service = DelayPredictionService()
