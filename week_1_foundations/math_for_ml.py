"""Numerical linear algebra and regression calculus."""

from typing import Tuple

import numpy as np


def matrix_operations(left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return matrix product and element-wise sum after shape validation."""
    left_array, right_array = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("both inputs must be two-dimensional")
    if left_array.shape[1] != right_array.shape[0]:
        raise ValueError("matrix dimensions are incompatible")
    return left_array @ right_array, left_array + right_array if left_array.shape == right_array.shape else left_array


def eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvalues for a square matrix."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return np.linalg.eigvals(array)


def mse_gradient(features: np.ndarray, targets: np.ndarray, weights: np.ndarray, bias: float = 0.0) -> Tuple[np.ndarray, float]:
    """Return analytical gradients of mean squared error for linear regression."""
    x, y, w = np.asarray(features, dtype=float), np.asarray(targets, dtype=float), np.asarray(weights, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or w.ndim != 1 or x.shape[0] != y.size or x.shape[1] != w.size:
        raise ValueError("features, targets, and weights have incompatible shapes")
    residual = x @ w + float(bias) - y
    return (2.0 / x.shape[0]) * x.T @ residual, float(2.0 * residual.mean())
