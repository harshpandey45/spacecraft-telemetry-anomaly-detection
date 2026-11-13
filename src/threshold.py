import numpy as np
import pandas as pd

def smooth_errors(y_true, y_pred, span=30):
    """
    1. raw_error: Jab sensor achanak spike kare (Point Anomaly)
    2. ctx_error: Jab oscillating sensor achanak freeze/flat ho jaye (Contextual Anomaly)
    """
    raw_error = np.abs(y_true - y_pred)
    roll_std = pd.Series(y_true).rolling(50).std().bfill().to_numpy()
    ctx_error = np.maximum(0.0, np.median(roll_std) - roll_std)
    combined = raw_error + 2.0 * ctx_error
    return pd.Series(combined).ewm(span=span).mean().to_numpy()

def find_dynamic_threshold(errors, z_range=np.arange(2.5, 4.0, 0.5)):
    """
    Bina ground-truth labels dekhe, error ke mean aur std se khud best cutoff line chunta hai.
    """
    mean_e = errors.mean()
    std_e = errors.std() + 1e-8
    best_thresh = mean_e + z_range[-1] * std_e
    best_score = -np.inf

    for z in z_range:
        eps = mean_e + z * std_e
        below = errors[errors < eps]
        above = errors >= eps

        if not above.any() or len(below) == 0:
            continue

        n_seq = int(np.sum(np.diff(above.astype(int)) == 1) + above[0])
        mean_drop = (mean_e - below.mean()) / (mean_e + 1e-8)
        std_drop = (std_e - below.std()) / (std_e + 1e-8)
        score = (mean_drop + std_drop) / (above.sum() + n_seq ** 2)

        if score > best_score:
            best_score = score
            best_thresh = eps

    return float(best_thresh)

def prune_false_positives(pred_seqs, errors, offset=250, min_drop=0.13):
    """
    NASA Telemanom Pruning: Sabse bade error peaks ko normal error se compare karta hai.
    Jahan >= 13% ka bada drop (cliff) aata hai, uske upar ke saare peaks ko rakhta hai.
    """
    if len(pred_seqs) == 0:
        return []

    peaks = [float(errors[s[0] - offset : s[1] - offset + 1].max()) for s in pred_seqs]
    order = np.argsort(peaks)[::-1]
    sorted_seqs = [pred_seqs[i] for i in order]
    sorted_peaks = [peaks[i] for i in order]

    normal_floor = float(np.percentile(errors, 95))
    all_levels = sorted_peaks + [normal_floor]

    last_cliff_idx = -1
    for i in range(len(sorted_peaks)):
        drop = (all_levels[i] - all_levels[i + 1]) / (all_levels[i] + 1e-8)
        if drop >= min_drop:
            last_cliff_idx = i

    if last_cliff_idx == -1:
        return [sorted_seqs[0]]
    return sorted_seqs[: last_cliff_idx + 1]