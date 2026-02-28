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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COL_N = "Humidity3pm"
COL_M = "Rainfall"
COL_TARGET = "RainTomorrow"
TEST_SIZE = 0.3  # 70% train / 30% test, как в лабораторных
CV_FOLDS = 5
RANDOM_STATE = 42
# Ограничение глубины дерева против переобучения (можно пробовать 3, 5, 7, 10, 15)
TREE_MAX_DEPTH = 10


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


def _evaluate_with_cv(model, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS, scoring: str = "f1") -> dict:
    """StratifiedKFold CV: объективная оценка при дисбалансе и склонности к переобучению."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    # clone(model) — незафитированная копия (модель уже обучена на single split)
    scores = cross_val_score(clone(model), X, y, cv=skf, scoring=scoring)
    return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


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

    # class_weight='balanced' повышает recall для Yes; max_depth — против переобучения дерева
    models = [
        (LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced"), "LogisticRegression"),
        (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced", max_depth=TREE_MAX_DEPTH), "DecisionTree"),
        (MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=RANDOM_STATE), "MLP"),
    ]

    results = []
    for model, name in models:
        res = _train_and_evaluate(model, X_train_s, y_train, X_test_s, y_test, name)
        # 5-fold StratifiedKFold для объективной оценки (recall/F1 скачут при дисбалансе)
        cv_f1 = _evaluate_with_cv(model, X_train_s, y_train, cv=CV_FOLDS, scoring="f1")
        cv_auc = _evaluate_with_cv(model, X_train_s, y_train, cv=CV_FOLDS, scoring="roc_auc")
        res["cv_f1_mean"], res["cv_f1_std"] = cv_f1["mean"], cv_f1["std"]
        res["cv_auc_mean"], res["cv_auc_std"] = cv_auc["mean"], cv_auc["std"]
        results.append(res)

        print(f"\n--- {name} ---")
        print("  Матрица ошибок:")
        print(res["confusion"])
        print(f"  Single split: Accuracy = {res['accuracy']:.4f}, Precision = {res['precision']:.4f}, Recall = {res['recall']:.4f}, F1 = {res['f1']:.4f}, AUC = {res['auc']:.4f}")
        print(f"  5-fold CV:    F1 = {res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f},  AUC = {res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}")

    # Сводная таблица (single split + 5-fold CV)
    print("\n--- Сводная таблица метрик (single split 70/30 + 5-fold CV) ---")
    table = pd.DataFrame(
        [
            {
                "Модель": r["name"],
                "Accuracy": f"{r['accuracy']:.4f}",
                "F1 (split)": f"{r['f1']:.4f}",
                "F1 (CV)": f"{r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.4f}",
                "AUC (split)": f"{r['auc']:.4f}",
                "AUC (CV)": f"{r['cv_auc_mean']:.4f}±{r['cv_auc_std']:.4f}",
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
        (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced", max_depth=TREE_MAX_DEPTH), "DecisionTree"),
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

    plot_tree_max_depth_sweep(df, use_smote=use_smote)
    print("\nЗадание 7 завершено. При дисбалансе классов ориентироваться на F1 и AUC, а не только на Accuracy.")


def plot_tree_max_depth_sweep(df: pd.DataFrame, use_smote: bool = True) -> None:
    """Перебор max_depth для дерева: график F1 и AUC (5-fold CV) от глубины."""
    X, y = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if use_smote:
        X_train_s, y_train = _smote_train(X_train_s, y_train)

    depths = [3, 5, 7, 10, 15]
    f1_means, f1_stds = [], []
    auc_means, auc_stds = [], []

    for d in depths:
        tree = DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", max_depth=d
        )
        cv_f1 = _evaluate_with_cv(tree, X_train_s, y_train, cv=CV_FOLDS, scoring="f1")
        cv_auc = _evaluate_with_cv(tree, X_train_s, y_train, cv=CV_FOLDS, scoring="roc_auc")
        f1_means.append(cv_f1["mean"])
        f1_stds.append(cv_f1["std"])
        auc_means.append(cv_auc["mean"])
        auc_stds.append(cv_auc["std"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(depths, f1_means, yerr=f1_stds, capsize=4, marker="o")
    axes[0].set_xlabel("max_depth")
    axes[0].set_ylabel("F1 (5-fold CV)")
    axes[0].set_title("DecisionTree: F1 от глубины дерева")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(depths, auc_means, yerr=auc_stds, capsize=4, marker="o", color="C1")
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("AUC (5-fold CV)")
    axes[1].set_title("DecisionTree: AUC от глубины дерева")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task7_tree_max_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task7_tree_max_depth.png (перебор max_depth 3,5,7,10,15)")
