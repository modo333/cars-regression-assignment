"""Deterministic 1→8→1 neural network for educational cars regression."""
from dataclasses import dataclass

import numpy as np


@dataclass
class NeuralRegressor:
    hidden_units: int = 8
    epochs: int = 2500
    learning_rate: float = 0.01
    regularization: float = 0.001

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(x) != len(y) or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Need at least three finite (speed, dist) observations")
        self.x_mean, self.y_mean = float(x.mean()), float(y.mean())
        self.x_scale, self.y_scale = max(float(x.std()), 1e-8), max(float(y.std()), 1e-8)
        xn, yn = (x - self.x_mean) / self.x_scale, (y - self.y_mean) / self.y_scale
        indices = np.arange(1, self.hidden_units + 1)
        self.w1, self.b1 = 0.5 * np.sin(indices), np.zeros(self.hidden_units)
        self.w2, self.b2 = 0.5 * np.cos(indices), 0.0
        bias2 = np.array([0.0])
        params = [self.w1, self.b1, self.w2, bias2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        n = len(x)
        for step in range(1, self.epochs + 1):
            hidden = np.tanh(xn[:, None] * self.w1 + self.b1)
            residual = hidden @ self.w2 + bias2[0] - yn
            dout = 2 * residual / n
            grad_w2 = hidden.T @ dout + 2 * self.regularization * self.w2
            grad_b2 = np.array([dout.sum()])
            dz = dout[:, None] * self.w2 * (1 - hidden**2)
            grad_w1 = (dz * xn[:, None]).sum(axis=0) + 2 * self.regularization * self.w1
            grad_b1 = dz.sum(axis=0)
            for p, grad, first, second in zip(params, [grad_w1, grad_b1, grad_w2, grad_b2], m, v):
                first *= 0.9
                first += 0.1 * grad
                second *= 0.999
                second += 0.001 * grad**2
                p -= self.learning_rate * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        self.b2 = float(bias2[0])
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        hidden = np.tanh(((x - self.x_mean) / self.x_scale)[:, None] * self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2) * self.y_scale + self.y_mean
