"""
Set of custom metrics to evaluate the performance of the models

Author: smas
Last update: 04/01/2024
"""

import numpy as np


def hit_rate(y_true: np.array, y_pred: np.array, hit_radius) -> float:
    """
    This function calculates the hit rate

    Args:
        y_true: Array with the true values
        y_pred: Array with the predicted values
        hit_radius: Radius to consider a positive hit

    Returns:
        Hit rate
    """

    assert len(y_true) == len(y_pred), f"Length of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must be the same"

    hits = 0

    for target, prediction in zip(y_true, y_pred):
        if np.abs(prediction - target) <= hit_radius:
            hits += 1

    return hits / len(y_true)
