"""
Общие функции для классификаторов: подготовка данных, SMOTE, оценка, CV.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.base import clone

from course_work.config import COL_TARGET, TEST_SIZE, CV_FOLDS, RANDOM_STATE

# Список признаков, устанавливается один раз в run_task7 из main.py
_features: list = []


def set_features(features: list):
    """Задать список признаков для классификации (вызывается из run_task7)."""
    global _features
    _features = list(features)


def prepare_xy(df: pd.DataFrame):
    """Признаки для классификации: используются ранее выбранные (set_features). Цель: 0/1."""
    if not _features:
        raise RuntimeError("Признаки не заданы. Вызовите set_features() перед prepare_xy().")
    X = df[_features].copy()
    if X.isnull().any().any():
        X = X.fillna(X.median())
    y = (df[COL_TARGET] == "Yes").astype(int)
    return X, y


def smote_train(X_train, y_train):
    """SMOTE только на обучающей выборке (опционально)."""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    except ImportError:
        return X_train, y_train


def evaluate_with_cv(model, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS, scoring: str = "f1") -> dict:
    """StratifiedKFold CV: объективная оценка при дисбалансе и склонности к переобучению."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clone(model), X, y, cv=skf, scoring=scoring)
    return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


def train_and_evaluate(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    """Обучение модели и расчёт метрик."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
        "confusion": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_proba": y_proba,
    }


def get_train_test_split(X, y):
    """Единое разбиение train/test для всех классификаторов."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
