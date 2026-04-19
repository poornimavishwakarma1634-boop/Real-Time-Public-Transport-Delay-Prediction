# Real-Time Public Transport Delay Prediction

A comprehensive machine learning project for predicting public transport delays using time-series forecasting techniques.

## 🎯 Project Overview

This project implements multiple approaches to predict bus/train delays:
- **Statistical Models**: ARIMA, SARIMA, Facebook Prophet
- **Deep Learning Models**: TensorFlow LSTM, PyTorch LSTM

## 📁 Project Structure

```
ml_project/
├── data/
│   ├── raw/              # Original GTFS data
│   ├── processed/        # Cleaned and preprocessed data
│   └── external/         # Weather and other external data
│
├── models/               # Trained models
│   ├── statistical/      # ARIMA, SARIMA, Prophet
│   ├── tensorflow/       # TensorFlow LSTM
│   └── pytorch/          # PyTorch LSTM
│
├── outputs/
│   ├── visualizations/   # Plots and charts
│   └── reports/          # Performance reports
│
├── src/                  # Source code
│   ├── 01_data_ingestion.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_statistical_models.py
│   ├── 05_tensorflow_lstm.py
│   ├── 06_pytorch_lstm.py
│   ├── 07_evaluation.py
│   └── utils.py
│
└── run_pipeline.py       # Master script
```

## 🚀 Quick Start

### Run Complete Pipeline

```bash
cd /app/ml_project
python run_pipeline.py
```

### Run Individual Steps

```bash
# Step 1: Generate GTFS data
python src/01_data_ingestion.py

# Step 2: Preprocess time-series
python src/02_preprocessing.py

# Step 3: Engineer features
python src/03_feature_engineering.py

# Step 4: Train statistical models
python src/04_statistical_models.py

# Step 5: Train TensorFlow LSTM
python src/05_tensorflow_lstm.py

# Step 6: Train PyTorch LSTM
python src/06_pytorch_lstm.py

# Step 7: Evaluate and compare
python src/07_evaluation.py
```

## 📊 Features

### Time-Series Preprocessing
- Resampling to fixed intervals (5 minutes)
- Missing value imputation
- Stationarity testing (ADF test)
- Seasonal decomposition

### Feature Engineering
- **Temporal Features**: Hour, day of week, rush hours
- **Lag Features**: Previous 12 time steps (1 hour history)
- **Rolling Statistics**: 30-min, 1-hour, 2-hour windows
- **Weather Features**: Rain/clear conditions
- **Cyclical Encoding**: Sin/cos transformations

### Models Implemented

#### Statistical Models
1. **ARIMA** - AutoRegressive Integrated Moving Average
2. **SARIMA** - Seasonal ARIMA with daily patterns
3. **Prophet** - Facebook's forecasting tool

#### Deep Learning Models
4. **TensorFlow LSTM** - Keras implementation
5. **PyTorch LSTM** - PyTorch implementation

## 📈 Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

## 🎨 Visualizations

All visualizations are saved to `/app/ml_project/outputs/visualizations/`:
- Time-series decomposition
- Individual model predictions
- Model comparison plots
- Metrics comparison bar charts

## 📄 Reports

Final reports are saved to `/app/ml_project/outputs/reports/`:
- `model_comparison.csv` - Tabular comparison
- `final_report.txt` - Comprehensive analysis
- Individual model result files

## 🔧 Technical Details

### Data Format
- **Timestamp**: 5-minute intervals
- **Target**: `delay_minutes` (delay in minutes)
- **Features**: 20+ engineered features

### Model Architecture (LSTM)
- 3 LSTM layers (128 → 64 → 32 units)
- Dropout (0.2) for regularization
- Dense output layer
- Adam optimizer
- Early stopping with patience=10

### Train-Test Split
- Training: 80% of data
- Testing: 20% of data
- Time-series split (no shuffling)

## 📦 Dependencies

All dependencies are pre-installed:
- pandas, numpy
- scikit-learn
- statsmodels
- prophet
- tensorflow
- torch
- matplotlib, seaborn, plotly

## 🎓 Learning Objectives

This project covers:
- ✅ Time-series analysis and preprocessing
- ✅ Stationarity testing and differencing
- ✅ Feature engineering for temporal data
- ✅ Statistical forecasting (ARIMA family)
- ✅ Deep learning for sequences (LSTM/GRU)
- ✅ Model evaluation and comparison
- ✅ Production-ready ML pipeline

## 🚀 Next Steps

1. **Run the pipeline**: `python run_pipeline.py`
2. **Check visualizations**: Navigate to `outputs/visualizations/`
3. **Read final report**: `outputs/reports/final_report.txt`
4. **Integrate with API**: Use models for real-time predictions

## 📝 Notes

- Synthetic GTFS data is generated automatically
- All paths are absolute for easy execution
- Models are saved for future inference
- Scalers are preserved for production deployment

---

**Built for**: Real-Time Public Transport Delay Prediction Internship
**Tech Stack**: Python, TensorFlow, PyTorch, Statsmodels, Prophet
