"""
MLP: перебор гиперпараметров (число нейронов, эпохи), сводная таблица и heatmap, затем обучение с лучшей конфигурацией.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from course_work.config import FIGURES_DIR, CV_FOLDS, RANDOM_STATE
from course_work.classifiers.common import prepare_xy, smote_train, evaluate_with_cv, get_train_test_split

# Сетка для перебора: архитектуры и число эпох (под 2–3 входа: малые сети).
# Один элемент — один скрытый слой. Два — два слоя, напр. (3, 2).
MLP_HIDDEN_LAYERS = [(3,), (5,), (10,), (20,), (10, 5), (3, 2)]
MLP_MAX_ITERS = [100, 500]


def _layer_repr(layers: tuple) -> str:
    """Строковое представление архитектуры для подписей."""
    return str(layers).replace(" ", "")


def sweep_mlp_hyperparams(df: pd.DataFrame, use_smote: bool = True) -> tuple:
    """
    Перебор hidden_layer_sizes x max_iter с 5-fold CV (F1 и AUC).
    Выводит сводную таблицу в консоль, сохраняет heatmap F1 -> task7_mlp_heatmap.png.
    Возвращает (best_hidden_layer_sizes, best_max_iter) для финального обучения.
    """
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if use_smote:
        X_train_s, y_train = smote_train(X_train_s, y_train)

    results = []
    f1_matrix = np.zeros((len(MLP_HIDDEN_LAYERS), len(MLP_MAX_ITERS)))
    auc_matrix = np.zeros((len(MLP_HIDDEN_LAYERS), len(MLP_MAX_ITERS)))
    n_combos = len(MLP_HIDDEN_LAYERS) * len(MLP_MAX_ITERS)
    combo = 0

    print("  Перебор архитектур × эпох (5-fold CV)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        for i, hidden in enumerate(MLP_HIDDEN_LAYERS):
            for j, max_iter in enumerate(MLP_MAX_ITERS):
                combo += 1
                print(f"    MLP {_layer_repr(hidden)} × {max_iter} эпох ({combo}/{n_combos})", flush=True)
                mlp = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    max_iter=max_iter,
                    random_state=RANDOM_STATE,
                )
                cv_f1 = evaluate_with_cv(mlp, X_train_s, y_train, cv=CV_FOLDS, scoring="f1")
                cv_auc = evaluate_with_cv(mlp, X_train_s, y_train, cv=CV_FOLDS, scoring="roc_auc")
                f1_matrix[i, j] = cv_f1["mean"]
                auc_matrix[i, j] = cv_auc["mean"]
                results.append({
                    "hidden_layer_sizes": _layer_repr(hidden),
                    "max_iter": max_iter,
                    "F1_mean": cv_f1["mean"],
                    "F1_std": cv_f1["std"],
                    "AUC_mean": cv_auc["mean"],
                    "AUC_std": cv_auc["std"],
                })

    results_df = pd.DataFrame(results)
    best_idx = results_df["F1_mean"].idxmax()
    best_row = results_df.loc[best_idx]
    best_hidden = MLP_HIDDEN_LAYERS[best_idx // len(MLP_MAX_ITERS)]
    best_max_iter = int(best_row["max_iter"])

    print("\n--- Сводка MLP: число нейронов (скрытый слой) × эпохи обучения (5-fold CV) ---")
    print(results_df.to_string(index=False))
    print(f"\nЛучшая конфигурация: hidden_layer_sizes={best_hidden}, max_iter={best_max_iter} (F1 = {best_row['F1_mean']:.4f} ± {best_row['F1_std']:.4f})")

    # Heatmap: F1 по (архитектура, max_iter)
    fig, ax = plt.subplots(figsize=(8, 5))
    row_labels = [_layer_repr(h) for h in MLP_HIDDEN_LAYERS]
    col_labels = [str(m) for m in MLP_MAX_ITERS]
    sns.heatmap(
        f1_matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        ax=ax,
    )
    ax.set_xlabel("max_iter (эпохи)")
    ax.set_ylabel("hidden_layer_sizes")
    ax.set_title("MLP: F1 (5-fold CV) — нейроны × эпохи")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task7_mlp_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task7_mlp_heatmap.png")

    return best_hidden, best_max_iter


def build_mlp(hidden_layer_sizes: tuple, max_iter: int):
    """Создать MLPClassifier с заданной архитектурой и числом эпох."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )
