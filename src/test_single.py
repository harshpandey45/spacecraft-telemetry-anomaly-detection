import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from baseline import to_sequences, score_events
from model import create_windows, TelemetryLSTM
from threshold import smooth_errors, find_dynamic_threshold, prune_false_positives

# Agar src folder ke andar kharab (0B) image padi hai toh use automatic hata dega
if os.path.exists("src/p1_result.png"):
    os.remove("src/p1_result.png")

# Reproducibility ke liye random seed fix karte hain
torch.manual_seed(42)
np.random.seed(42)

chan_id = "P-1"
window_size = 250

# 1. Data load karein
labels = pd.read_csv("data/labeled_anomalies.csv")
row = labels[labels["chan_id"] == chan_id].iloc[0]
true_seqs = ast.literal_eval(row["anomaly_sequences"])

train = np.load(f"data/train/{chan_id}.npy")
test = np.load(f"data/test/{chan_id}.npy")

# 2. Sliding windows banayein
X_train, y_train = create_windows(train, window_size)
X_test, y_test = create_windows(test, window_size)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=64,
    shuffle=True
)

# 3. LSTM Model banayein aur Train karein (10 epochs)
model = TelemetryLSTM(input_dim=train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"Training LSTM on {chan_id}...")
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for bx, by in train_loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d}/10 | Loss: {total_loss/len(train_loader):.5f}")

# 4. Test data par prediction aur anomaly detection
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test)).numpy()

errors = smooth_errors(y_test, y_pred, span=30)
eps = find_dynamic_threshold(errors)
raw_seqs = to_sequences(errors >= eps, offset=window_size)
pred_seqs = prune_false_positives(raw_seqs, errors, offset=window_size)

tp, fp, fn = score_events(pred_seqs, true_seqs)
print(f"\nTrue Anomalies:      {true_seqs}")
print(f"Predicted Anomalies: {pred_seqs}")
print(f"Result -> Found (TP): {tp}/{len(true_seqs)} | False Alarms (FP): {fp}")

# 5. 3-Panel Visual Proof Graph banayein aur automatic save karein
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Top: Actual vs Predicted
axes[0].plot(range(window_size, len(test)), y_test, label="Actual Sensor", color="blue", lw=1)
axes[0].plot(range(window_size, len(test)), y_pred, label="LSTM Predicted", color="orange", lw=1, alpha=0.8)
axes[0].set_title(f"{chan_id}: Actual vs LSTM Predicted Telemetry")
axes[0].legend()

# Middle: Smoothed Error & Dynamic Threshold
axes[1].plot(range(window_size, len(test)), errors, label="Smoothed Error", color="purple", lw=1)
axes[1].axhline(eps, color="red", linestyle="--", label=f"Dynamic Threshold ({eps:.3f})")
axes[1].set_title("Prediction Error & Unsupervised Threshold")
axes[1].legend()

# Bottom: Ground Truth vs Detected Intervals
axes[2].plot(test[:, 0], color="gray", alpha=0.5, lw=0.8)
for s, e in true_seqs:
    axes[2].axvspan(s, e, color="red", alpha=0.35, label="True Anomaly")
for s, e in pred_seqs:
    axes[2].axvspan(s, e, color="green", alpha=0.35, label="LSTM Detected")
axes[2].set_title("Red = True Anomaly | Green = LSTM Detected")
handles, lbls = axes[2].get_legend_handles_labels()
by_label = dict(zip(lbls, handles))
axes[2].legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.savefig("p1_result.png", dpi=150)
print("\nGraph successfully saved as 'p1_result.png' in main project folder!")
plt.show()