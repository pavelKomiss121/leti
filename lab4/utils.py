"""
Вспомогательные функции: метрики и разбиение выборки.
"""
import numpy as np


def accuracy(y_true, pred_proba, threshold=0.5):
    """Точность: доля правильных предсказаний. pred_proba — выход НС (0..1)."""
    y_true = np.asarray(y_true).ravel()
    pred_proba = np.asarray(pred_proba).ravel()
    pred_class = (pred_proba >= threshold).astype(np.int64)
    y_int = (
        (np.asarray(y_true) != 0).astype(np.int64)
        if y_true.dtype == bool
        else np.asarray(y_true, dtype=np.int64)
    )
    return np.mean(pred_class == y_int)


def mse_loss(y_true, pred):
    """Среднеквадратичная ошибка."""
    y_true = np.asarray(y_true).ravel()
    pred = np.asarray(pred).ravel()
    return np.mean((y_true - pred) ** 2)


def split_70_30(X, Y, seed=42):
    """Разбиение на обучающую (70%) и тестовую (30%) выборки."""
    n = len(Y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    X, Y = X[idx], Y[idx]
    train_count = round(0.7 * n)
    return X[:train_count], Y[:train_count], X[train_count:], Y[train_count:]
