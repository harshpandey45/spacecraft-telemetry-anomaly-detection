import numpy as np
import torch
import torch.nn as nn

def create_windows(data, window_size=250):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class TelemetryLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=80, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)