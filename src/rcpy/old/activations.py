from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    bx = beta * np.asarray(x)
    return np.exp(bx) / np.sum(np.exp(bx))


def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    bx = beta * np.asarray(x)
    return (1.0 / beta) * np.log(1.0 + np.exp(bx))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def identity(x: np.ndarray) -> np.ndarray:
    return np.asarray(x)


def relu(x: np.ndarray) -> np.ndarray:
    _x = np.asarray(x)
    return _x * (_x > 0)
