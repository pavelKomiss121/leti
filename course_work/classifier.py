"""
Классификаторы (Задание 7): логистическая регрессия, дерево решений, MLP.
Один файл — одна задача: обучение, метрики, ROC, сравнение.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COL_N = "Humidity3pm"
COL_M = "Rainfall"
COL_TARGET = "RainTomorrow"
TEST_SIZE = 0.3  # 70% train / 30% test, как в лабораторных
RANDOM_STATE = 42


def _prepare_xy(df: pd.DataFrame):
    """Признаки: только числовые (n, m и др.). Location и категориальные дропнуты. Цель: 0/1."""
    numeric = df.select_dtypes(include=[np.number])
    if COL_TARGET in numeric.columns:
        X = numeric.drop(columns=[COL_TARGET])
    else:
        X = numeric.copy()
    # Удалить столбец даты, если попал (например, как число)
    for c in ["Date", "Unnamed: 0"]:
        if c in X.columns:
            X = X.drop(columns=[c])
    y = (df[COL_TARGET] == "Yes").astype(int)
    return X, y


def _smote_train(X_train, y_train):
    """SMOTE только на обучающей выборке (опционально)."""
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    except ImportError:
        return X_train, y_train


def _train_and_evaluate(model, X_train, y_train, X_test, y_test, name: str) -> dict:
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


def run_classifiers(df: pd.DataFrame, use_smote: bool = True) -> None:
    """Обучение трёх классификаторов, метрики, сводная таблица."""
    X, y = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if use_smote:
        X_train_s, y_train = _smote_train(X_train_s, y_train)
        print("  Применён SMOTE на обучающей выборке.")

    # class_weight='balanced' повышает recall для класса Yes (дождь) при дисбалансе 78/22
    models = [
        (LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced"), "LogisticRegression"),
        (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"), "DecisionTree"),
        (MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=RANDOM_STATE), "MLP"),
    ]

    results = []
    for model, name in models:
        res = _train_and_evaluate(model, X_train_s, y_train, X_test_s, y_test, name)
        results.append(res)

        print(f"\n--- {name} ---")
        print("  Матрица ошибок:")
        print(res["confusion"])
        print(f"  Accuracy = {res['accuracy']:.4f}, Precision = {res['precision']:.4f}, Recall = {res['recall']:.4f}, F1 = {res['f1']:.4f}, AUC = {res['auc']:.4f}")

    # Сводная таблица
    print("\n--- Сводная таблица метрик ---")
    table = pd.DataFrame(
        [
            {
                "Модель": r["name"],
                "Accuracy": f"{r['accuracy']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "F1": f"{r['f1']:.4f}",
                "AUC": f"{r['auc']:.4f}",
            }
            for r in results
        ]
    )
    print(table.to_string(index=False))

    return results


def plot_roc_curves(df: pd.DataFrame, use_smote: bool = True) -> None:
    """ROC-кривые для всех трёх моделей на одном графике."""
    X, y = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if use_smote:
        X_train_s, y_train = _smote_train(X_train_s, y_train)

    fig, ax = plt.subplots(figsize=(8, 6))
    models = [
        (LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced"), "LogisticRegression"),
        (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"), "DecisionTree"),
        (MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=RANDOM_STATE), "MLP"),
    ]
    for model, name in models:
        model.fit(X_train_s, y_train)
        RocCurveDisplay.from_estimator(model, X_test_s, y_test, ax=ax, name=name)

    ax.set_title("ROC-кривые классификаторов")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task7_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task7_roc_curves.png")


def run_task7(df: pd.DataFrame, use_smote: bool = True) -> None:
    """Задание 7: классификаторы и оценка качества. По умолчанию SMOTE включён для борьбы с дисбалансом 78/22."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 7: классификаторы")
    print("=" * 60)

    print("\nПризнаки: числовые столбцы (n, m и др.), Location и категориальные не используются.")
    print("Балансировка: class_weight='balanced' (LogReg, Tree) + SMOTE на train по умолчанию.")
    run_classifiers(df, use_smote=use_smote)
    plot_roc_curves(df, use_smote=use_smote)

    print("\nЗадание 7 завершено. При дисбалансе классов ориентироваться на F1 и AUC, а не только на Accuracy.")
