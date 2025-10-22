import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def calibrate_probabilities(y_prob, y_true, method="isotonic"):
    """Train a calibration model (Platt scaling or isotonic) on given probabilities."""
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_prob, y_true)
        return iso
    elif method == "platt":
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(y_prob.reshape(-1,1), y_true)
        return lr
    else:
        return None

def apply_calibration(model, y_prob):
    """Apply a fitted calibration model (isotonic or logistic) to probabilities."""
    if model is None:
        return y_prob
    if isinstance(model, IsotonicRegression):
        return model.predict(y_prob)
    elif isinstance(model, LogisticRegression):
        return model.predict_proba(y_prob.reshape(-1,1))[:,1]
    else:
        return y_prob
