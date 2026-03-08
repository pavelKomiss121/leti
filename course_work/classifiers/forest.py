"""
Случайный лес: перебор глубины деревьев и числа деревьев (n_estimators), затем обучение с лучшими параметрами.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from course_work.config import FIGURES_DIR, CV_FOLDS, RANDOM_STATE
from course_work.classifiers.common import (
    prepare_xy,
    smote_train,
    evaluate_with_cv,
    get_train_test_split,
)

# Сетка для перебора: глубина деревьев и размер леса (сокращено для скорости)
FOREST_DEPTHS = [4, 8, 16, 20]
FOREST_N_ESTIMATORS = [50, 300]


def find_optimal_forest_params(
    df: pd.DataFrame, use_smote: bool = True, criterion: str = "roc_auc"
) -> tuple:
    """
    Перебор max_depth × n_estimators с 5-fold CV (F1 и AUC).
    Строит heatmap F1 и AUC. Возвращает (best_depth, best_n_estimators).
    """
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if use_smote:
        X_train_s, y_train = smote_train(X_train_s, y_train)

    f1_matrix = np.zeros((len(FOREST_DEPTHS), len(FOREST_N_ESTIMATORS)))
    auc_matrix = np.zeros((len(FOREST_DEPTHS), len(FOREST_N_ESTIMATORS)))
    total = len(FOREST_DEPTHS) * len(FOREST_N_ESTIMATORS)
    idx = 0

    print("  Перебор глубины × число деревьев (5-fold CV)...")
    for i, d in enumerate(FOREST_DEPTHS):
        for j, n in enumerate(FOREST_N_ESTIMATORS):
            idx += 1
            print(f"    глубина {d}, n_estimators={n} ({idx}/{total})", flush=True)
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )
            cv_f1 = evaluate_with_cv(rf, X_train_s, y_train, cv=CV_FOLDS, scoring="f1")
            cv_auc = evaluate_with_cv(rf, X_train_s, y_train, cv=CV_FOLDS, scoring="roc_auc")
            f1_matrix[i, j] = cv_f1["mean"]
            auc_matrix[i, j] = cv_auc["mean"]

    if criterion == "f1":
        best_flat = np.argmax(f1_matrix)
    else:
        best_flat = np.argmax(auc_matrix)
    best_i = best_flat // len(FOREST_N_ESTIMATORS)
    best_j = best_flat % len(FOREST_N_ESTIMATORS)
    best_depth = FOREST_DEPTHS[best_i]
    best_n = FOREST_N_ESTIMATORS[best_j]
    best_auc = auc_matrix[best_i, best_j]
    best_f1 = f1_matrix[best_i, best_j]

    print(f"\n  Оптимальные параметры леса (по {criterion}): max_depth={best_depth}, n_estimators={best_n}")
    print(f"  (F1 = {best_f1:.4f}, AUC = {best_auc:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in [
        (axes[0], f1_matrix, "RandomForest: F1 (5-fold CV)"),
        (axes[1], auc_matrix, "RandomForest: AUC (5-fold CV)"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(FOREST_N_ESTIMATORS)))
        ax.set_yticks(range(len(FOREST_DEPTHS)))
        ax.set_xticklabels(FOREST_N_ESTIMATORS)
        ax.set_yticklabels(FOREST_DEPTHS)
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("max_depth")
        ax.set_title(title)
        for i in range(len(FOREST_DEPTHS)):
            for j in range(len(FOREST_N_ESTIMATORS)):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task7_forest_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Сохранено: figures/task7_forest_heatmap.png")

    return best_depth, best_n


def build_forest(max_depth: int, n_estimators: int = 200):
    """Создать RandomForestClassifier с заданной глубиной деревьев и числом деревьев."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
