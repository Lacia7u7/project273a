import numpy as np
import pandas as pd

def decision_curve_analysis(y_true, y_prob):
    """Compute net benefit for various threshold probabilities."""
    results = []
    N = len(y_true)
    y_true = np.array(y_true)
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        # net benefit = TP/N - FP/N * (thr/(1-thr))
        net_benefit = tp/N - fp/N * (thr/(1-thr))
        results.append((thr, net_benefit))
    return pd.DataFrame(results, columns=["threshold","net_benefit"])
