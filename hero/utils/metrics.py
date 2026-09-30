import numpy as np
from sklearn.metrics import average_precision_score


def mean_average_precision(labels, probs):
    """One-vs-rest average precision per class and their mean.

    Args:
        labels: (N,) integer labels.
        probs: (N, C) predicted class probabilities.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    aps = []
    for c in range(probs.shape[1]):
        target = labels == c
        aps.append(average_precision_score(target, probs[:, c]) if target.any() else float("nan"))
    return float(np.nanmean(aps)), aps
