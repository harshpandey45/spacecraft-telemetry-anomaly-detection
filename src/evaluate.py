import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import to_sequences, score_events
from model import create_windows, TelemetryLSTM
from threshold import smooth_errors, find_dynamic_threshold, prune_false_positives

torch.manual_seed(42)
np.random.seed(42)

def evaluate_all(window_size=250):
    labels = pd.read_csv("data/labeled_anomalies.csv")
    total_tp, total_fp, total_fn = 0, 0, 0

    print(f"Running LSTM + Dynamic Threshold Benchmark on {len(labels)} channels...\n")

    for idx, row in labels.iterrows():
        chan_id = row["chan_id"]
        true_seqs = ast.literal_eval(row["anomaly_sequences"])

        train = np.load(f"data/train/{chan_id}.npy")
        test = np.load(f"data/test/{chan_id}.npy")

        if len(train) <= window_size or len(test) <= window_size:
            continue

        # Fast CPU training ke liye recent 1500 steps lete hain
        train_slice = train[-1500:] if len(train) > 1500 else train
        X_train, y_train = create_windows(train_slice, window_size)
        X_test, y_test = create_windows(test, window_size)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=256,
            shuffle=True,
        )

        model = TelemetryLSTM(input_dim=train.shape[1], hidden_dim=64, num_layers=1, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(5):
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        # Batch-wise test prediction
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                batch_x = torch.tensor(X_test[i : i + 512])
                preds.append(model(batch_x).numpy())
        y_pred = np.concatenate(preds)

        errors = smooth_errors(y_test, y_pred, span=30)
        eps = find_dynamic_threshold(errors)
        raw_seqs = to_sequences(errors >= eps, offset=window_size)
        pred_seqs = prune_false_positives(raw_seqs, errors, offset=window_size)

        tp, fp, fn = score_events(pred_seqs, true_seqs)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if (idx + 1) % 10 == 0 or (idx + 1) == len(labels):
            print(f"Processed [{idx + 1:02d}/{len(labels)}] channels | Current TP: {total_tp}, FP: {total_fp}")

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n========================================================")
    print("   FINAL BENCHMARK COMPARISON (82 NASA CHANNELS)")
    print("========================================================")
    print(f"{'Metric':<22} | {'Z-Score Baseline':<18} | {'Our LSTM Pipeline'}")
    print("-" * 56)
    print(f"{'Anomalies Found (TP)':<22} | {'58 / 105':<18} | {total_tp} / {total_tp + total_fn}")
    print(f"{'False Alarms (FP)':<22} | {'1132':<18} | {total_fp}")
    print(f"{'Precision':<22} | {'0.049 (4.9%)':<18} | {precision:.3f} ({precision*100:.1f}%)")
    print(f"{'Recall':<22} | {'0.552 (55.2%)':<18} | {recall:.3f} ({recall*100:.1f}%)")
    print(f"{'F1-Score':<22} | {'0.090 (9.0%)':<18} | {f1:.3f} ({f1*100:.1f}%)")
    print("========================================================")

if __name__ == "__main__":
    evaluate_all()