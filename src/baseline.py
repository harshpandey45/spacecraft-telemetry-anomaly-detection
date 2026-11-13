import ast
import numpy as np
import pandas as pd

def to_sequences(flags, offset=0):
    """Lagatar aane wale True flags ko [start, end] intervals mein group karta hai."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [[int(s[0] + offset), int(s[-1] + offset)] for s in splits]

def score_events(pred_seqs, true_seqs):
    """Check karta hai ki humare pakde gaye intervals asli anomaly se overlap karte hain ya nahi."""
    hit = lambda a, b: a[0] <= b[1] and b[0] <= a[1]
    tp = sum(any(hit(p, t) for p in pred_seqs) for t in true_seqs)   # Sahi pakdi gayi anomalies
    fp = sum(not any(hit(p, t) for t in true_seqs) for p in pred_seqs) # Galat alarms (False Positives)
    fn = len(true_seqs) - tp                                           # Jo miss ho gayin (False Negatives)
    return tp, fp, fn

def run_baseline():
    labels = pd.read_csv("data/labeled_anomalies.csv")
    total_tp, total_fp, total_fn = 0, 0, 0

    for _, row in labels.iterrows():
        chan_id = row["chan_id"]
        true_seqs = ast.literal_eval(row["anomaly_sequences"])

        train = np.load(f"data/train/{chan_id}.npy")
        test = np.load(f"data/test/{chan_id}.npy")

        # Training data ka average (mean) aur failav (std) nikalte hain
        mean = train[:, 0].mean()
        std = train[:, 0].std() + 1e-8

        # Agar test value average se 3*std se zyada door hai, toh flag karo
        z_scores = np.abs(test[:, 0] - mean) / std
        pred_seqs = to_sequences(z_scores > 3.0)

        tp, fp, fn = score_events(pred_seqs, true_seqs)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)

    print("=== Z-SCORE BASELINE RESULTS (82 Channels) ===")
    print(f"Total Anomalies Found (TP): {total_tp} out of {total_tp + total_fn}")
    print(f"False Alarms (FP):          {total_fp}")
    print(f"Precision:                  {precision:.3f}")
    print(f"Recall:                     {recall:.3f}")

if __name__ == "__main__":
    run_baseline()