# Real-Time Public Transport Delay Prediction - PRD

## Original Problem Statement
Build a Real-Time Public Transport Delay Prediction system for **Bangalore and Kalaburagi** cities using bus and train transport data. Implement time-series forecasting with statistical models (ARIMA, SARIMA, Prophet) and deep learning models (TensorFlow LSTM, PyTorch LSTM). Deliver standalone ML scripts + integrated full-stack web app.

## Architecture
- **ML Project**: `/app/ml_project/` - 7 Python scripts for complete data science pipeline
- **Backend**: FastAPI (port 8001) with ML prediction APIs, GTFS fetcher, scheduler, MongoDB
- **Frontend**: React with Recharts, dark Swiss design theme
- **Data Sources**: BMTC GTFS (GitHub), live feed simulator, MongoDB persistence

## What's Been Implemented

### Session 1 (MVP)
- [x] Synthetic GTFS data generation for Bangalore & Kalaburagi (90 days)
- [x] Time-series preprocessing (resampling, imputation, ADF stationarity)
- [x] Feature engineering (temporal, lag, rolling, weather)
- [x] ARIMA, SARIMA, Prophet, TF LSTM, PyTorch LSTM — all trained/evaluated
- [x] Model comparison report (Prophet best at RMSE 5.61)
- [x] Full-stack dashboard (route selector, predictions, charts, stats)

### Session 2 (Features)
- [x] **Real GTFS Integration**: Fetches BMTC GTFS from GitHub (4210 routes, 200 stops, 500 trips). Parses routes.txt, stops.txt, trips.txt. NEKRTC data via simulation.
- [x] **Live Feed Simulator**: Generates realistic delay observations every 15 min, stored in MongoDB `live_delays` collection. Backfills 48h on startup. Patterns include rush hour, weekend, monsoon weather, random incidents.
- [x] **Model Retraining Scheduler**: Background thread retrains Prophet + TF LSTM + PyTorch LSTM every 60 min. Auto-reloads models in serving layer. Tracks retrain count, last retrain time.
- [x] **Real Model Inference**: Replaced rule-based fallback with actual Prophet forecast, TF LSTM sequence prediction, PyTorch LSTM sequence prediction. Route-specific adjustment using historical delay profiles.

### API Endpoints (v2.0.0)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/predict | POST | Real ML prediction (Prophet/TF-LSTM/PyTorch-LSTM) |
| /api/models | GET | Available models list |
| /api/routes | GET | 8 routes (Bangalore + Kalaburagi) |
| /api/historical | GET | Historical delay data from CSV |
| /api/live-feed | GET | Real-time delays from MongoDB |
| /api/model-comparison | GET | RMSE/MAE for all models |
| /api/pipeline-status | GET | Models, scheduler, live feed status |
| /api/gtfs-status | GET | GTFS feed info (4210 routes) |
| /api/route-stats | GET | Per-route delay statistics |
| /api/scheduler/retrain | POST | Trigger manual retrain |

## Testing Results
- **Iteration 1**: 13/13 backend, 100% frontend
- **Iteration 2**: 15/15 backend, 100% frontend — all 3 new features verified

## Prioritized Backlog
### P1
- Integration with actual BMTC real-time GPS API (when available)
- More granular route-specific model training
- Ensemble model combining all predictions

### P2
- WebSocket for push-based live delay updates
- User accounts and favorite routes
- Multi-language support (Kannada, Hindi)
- Mobile-responsive improvements

### Future
- Push notifications for delay alerts
- Google Maps API integration
- Real weather API (OpenWeatherMap)
- A/B testing framework for model selection
