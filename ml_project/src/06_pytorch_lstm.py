"""
Step 6: PyTorch LSTM Model for transport delay prediction
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_section_header, calculate_metrics, plot_predictions, save_model_results, create_sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        out = self.fc2(out)
        return out


def load_data(filepath):
    print_section_header("Loading Data for PyTorch LSTM")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} records")
    return df


def prepare_data(series, n_steps=16, test_size=0.2, batch_size=64):
    print_section_header(f"Preparing PyTorch Data (n_steps={n_steps})")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = create_sequences(scaled, n_steps)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    train_loader = DataLoader(TSDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TSDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return train_loader, test_loader, scaler, X_test, y_test


def train_model(model, train_loader, test_loader, epochs=30, lr=0.001):
    print_section_header("Training PyTorch LSTM")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val = float('inf')
    patience, patience_cnt = 5, 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            loss = criterion(out, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                val_loss += criterion(out, y_b).item()
        val_loss /= len(test_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_state)
                break
    
    if best_state:
        model.load_state_dict(best_state)
    print(f"Training Complete! Best val_loss: {best_val:.6f}")
    return model


def evaluate(model, X_test, y_test, scaler, model_name='PyTorch LSTM'):
    print_section_header(f"Evaluating {model_name}")
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy()
    
    predictions = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    metrics = calculate_metrics(y_true.flatten(), predictions.flatten())
    print(f"RMSE: {metrics['RMSE']:.4f} min | MAE: {metrics['MAE']:.4f} min")
    
    plot_path = f'/app/ml_project/outputs/visualizations/{model_name.lower().replace(" ", "_")}_predictions.png'
    plot_predictions(y_true.flatten(), predictions.flatten(), f'{model_name}: Actual vs Predicted', save_path=plot_path)
    save_model_results(model_name, metrics, '/app/ml_project/outputs/reports')
    
    return predictions.flatten(), metrics


def save_model_files(model, scaler, model_name, save_dir='/app/ml_project/models/pytorch'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}.pth'))
    with open(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved to: {save_dir}")


def main():
    print_section_header("PYTORCH LSTM PIPELINE")
    df = load_data('/app/ml_project/data/processed/processed_delays.csv')
    n_steps = 16
    train_loader, test_loader, scaler, X_test, y_test = prepare_data(df['delay_minutes'], n_steps=n_steps)
    
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_model(model, train_loader, test_loader, epochs=30, lr=0.001)
    predictions, metrics = evaluate(model, X_test, y_test, scaler)
    save_model_files(model, scaler, 'PyTorch LSTM')
    
    print("\n" + "="*70)
    print(f"PyTorch LSTM Complete! RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    print("="*70)
    return model, predictions, metrics


if __name__ == "__main__":
    model, predictions, metrics = main()
