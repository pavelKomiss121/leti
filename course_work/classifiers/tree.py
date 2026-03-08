"""
Дерево решений: поиск оптимальной глубины (1–20) через 5-fold CV, затем обучение с этой глубиной.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from course_work.config import FIGURES_DIR, CV_FOLDS, RANDOM_STATE
from course_work.classifiers.common import (
    prepare_xy,
    smote_train,
    evaluate_with_cv,
    get_train_test_split,
)


def find_optimal_tree_depth(
    df: pd.DataFrame, use_smote: bool = True, criterion: str = "roc_auc"
) -> int:
    """
    Перебор max_depth 1–20 с 5-fold StratifiedKFold (F1 и AUC).
    Строит график с errorbar (std по фолдам). По умолчанию глубина выбирается по AUC
    (лучше отражает обобщающую способность; по F1 часто выгодна максимальная глубина).
    criterion: "roc_auc" или "f1" — по какой метрике выбирать оптимальную глубину.
    """
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if use_smote:
        X_train_s, y_train = smote_train(X_train_s, y_train)

    depths = list(range(1, 21))
    f1_means, f1_stds = [], []
    auc_means, auc_stds = [], []

    print("  Перебор глубин 1–20 (5-fold CV по F1 и AUC)...")
    for d in depths:
        print(f"    глубина {d}/20", flush=True)
        tree = DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", max_depth=d
        )
        cv_f1 = evaluate_with_cv(tree, X_train_s, y_train, cv=CV_FOLDS, scoring="f1")
        cv_auc = evaluate_with_cv(tree, X_train_s, y_train, cv=CV_FOLDS, scoring="roc_auc")
        f1_means.append(cv_f1["mean"])
        f1_stds.append(cv_f1["std"])
        auc_means.append(cv_auc["mean"])
        auc_stds.append(cv_auc["std"])

    idx_best_f1 = max(range(len(depths)), key=lambda i: f1_means[i])
    idx_best_auc = max(range(len(depths)), key=lambda i: auc_means[i])
    if criterion == "f1":
        best_idx = idx_best_f1
        best_depth = depths[best_idx]
        print(f"\n--- Оптимальная глубина дерева (по F1, 5-fold CV): {best_depth} (F1 = {f1_means[best_idx]:.4f} ± {f1_stds[best_idx]:.4f})")
    else:
        best_idx = idx_best_auc
        best_depth = depths[best_idx]
        print(f"\n--- Оптимальная глубина дерева (по AUC, 5-fold CV): {best_depth} (AUC = {auc_means[best_idx]:.4f} ± {auc_stds[best_idx]:.4f})")
    print(f"  (по F1 лучшая глубина: {depths[idx_best_f1]}; по AUC лучшая глубина: {depths[idx_best_auc]})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(depths, f1_means, yerr=f1_stds, capsize=4, marker="o")
    axes[0].axvline(best_depth, color="gray", linestyle="--", alpha=0.7, label=f"оптимум = {best_depth}")
    axes[0].set_xlabel("max_depth")
    axes[0].set_ylabel("F1 (5-fold CV)")
    axes[0].set_title("DecisionTree: F1 от глубины дерева")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(depths, auc_means, yerr=auc_stds, capsize=4, marker="o", color="C1")
    axes[1].axvline(best_depth, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("AUC (5-fold CV)")
    axes[1].set_title("DecisionTree: AUC от глубины дерева")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task7_tree_max_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task7_tree_max_depth.png (перебор max_depth 1–20)")

    return best_depth


def build_tree(max_depth: int):
    """Создать DecisionTreeClassifier с заданной глубиной."""
    return DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", max_depth=max_depth
    )
