"""
Задание 7: сравнение классификаторов (сводная таблица, ROC-кривые).
Сначала поиск оптимальных гиперпараметров (глубина дерева, MLP), затем обучение всех моделей.
Признаки передаются из main.py (выбраны на шаге отбора признаков).
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from course_work.config import FIGURES_DIR, CV_FOLDS
from course_work.classifiers.common import (
    set_features,
    prepare_xy,
    smote_train,
    train_and_evaluate,
    evaluate_with_cv,
    get_train_test_split,
)
from course_work.classifiers.logistic import build_logistic
from course_work.classifiers.tree import find_optimal_tree_depth, build_tree
from course_work.classifiers.forest import build_forest, find_optimal_forest_params
from course_work.classifiers.neural import sweep_mlp_hyperparams, build_mlp


def run_classifiers(
    df: pd.DataFrame,
    use_smote: bool = True,
    best_depth: int = None,
    best_forest_depth: int = None,
    best_n_estimators: int = None,
    best_mlp_hidden: tuple = None,
    best_mlp_max_iter: int = None,
) -> list:
    """Обучение четырёх классификаторов с оптимальными параметрами."""
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if use_smote:
        X_train_s, y_train = smote_train(X_train_s, y_train)
        print("  Применён SMOTE на обучающей выборке.")

    models = [
        (build_logistic(), "LogisticRegression"),
        (build_tree(best_depth), "DecisionTree"),
        (build_forest(best_forest_depth, best_n_estimators), "RandomForest"),
        (build_mlp(best_mlp_hidden, best_mlp_max_iter), "MLP"),
    ]

    results = []
    for model, name in models:
        print(f"  Обучение и оценка: {name}...", flush=True)
        res = train_and_evaluate(model, X_train_s, y_train, X_test_s, y_test, name)
        cv_f1 = evaluate_with_cv(model, X_train_s, y_train, cv=CV_FOLDS, scoring="f1")
        cv_auc = evaluate_with_cv(model, X_train_s, y_train, cv=CV_FOLDS, scoring="roc_auc")
        res["cv_f1_mean"], res["cv_f1_std"] = cv_f1["mean"], cv_f1["std"]
        res["cv_auc_mean"], res["cv_auc_std"] = cv_auc["mean"], cv_auc["std"]
        results.append(res)

        y_pred = res["model"].predict(X_test_s)
        res["ap"] = average_precision_score(res["y_test"], res["y_proba"])

        print(f"\n--- {name} ---")
        print("  Матрица ошибок:")
        print(res["confusion"])
        print("  Отчёт по классам (устойчив к дисбалансу; смотреть на класс Yes):")
        print(classification_report(res["y_test"], y_pred, target_names=["No", "Yes"], zero_division=0))
        print(f"  AUC-PR (average precision) = {res['ap']:.4f}  (чем выше, тем лучше; учитывает дисбаланс)")
        print(f"  Single split: Accuracy = {res['accuracy']:.4f}, F1 = {res['f1']:.4f}, ROC-AUC = {res['auc']:.4f}")
        print(f"  5-fold CV:    F1 = {res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f},  ROC-AUC = {res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}")

    # Проверка 1: только объекты с фактическим «дождь» (Yes) — предсказывает ли модель Yes?
    mask_yes = (y_test.values == 1) if hasattr(y_test, "values") else (y_test == 1)
    n_yes = int(mask_yes.sum())
    if n_yes > 0:
        X_yes = X_test_s[mask_yes]
        print("\n--- Проверка на объектах с фактическим классом «дождь» (Yes) ---")
        print(f"  В тесте таких объектов: {n_yes}. Если модель «мухлюет» (всему ставит No), здесь предскажет в основном No.")
        for r in results:
            pred_on_yes = r["model"].predict(X_yes)
            frac_yes = pred_on_yes.mean()
            print(f"  {r['name']}: доля предсказаний «Yes» = {frac_yes:.1%}")
    else:
        print("\n  (В тесте нет объектов класса Yes — проверку пропускаем.)")

    # Проверка 2: только объекты с фактическим «нет дождя» (No) — не предсказывает ли модель лишний раз Yes?
    mask_no = (y_test.values == 0) if hasattr(y_test, "values") else (y_test == 0)
    n_no = int(mask_no.sum())
    if n_no > 0:
        X_no = X_test_s[mask_no]
        print("\n--- Проверка на объектах с фактическим классом «нет дождя» (No) ---")
        print(f"  В тесте таких объектов: {n_no}. Хорошая модель здесь в основном предскажет No; если часто Yes — ложные срабатывания.")
        for r in results:
            pred_on_no = r["model"].predict(X_no)
            frac_yes_on_no = pred_on_no.mean()
            print(f"  {r['name']}: доля предсказаний «Yes» = {frac_yes_on_no:.1%}")
    else:
        print("\n  (В тесте нет объектов класса No — проверку пропускаем.)")

    print("\n--- Сводная таблица метрик (single split 70/30 + 5-fold CV) ---")
    table = pd.DataFrame(
        [
            {
                "Модель": r["name"],
                "Accuracy": f"{r['accuracy']:.4f}",
                "F1 (split)": f"{r['f1']:.4f}",
                "F1 (CV)": f"{r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.4f}",
                "ROC-AUC (CV)": f"{r['cv_auc_mean']:.4f}±{r['cv_auc_std']:.4f}",
                "AUC-PR": f"{r['ap']:.4f}",
            }
            for r in results
        ]
    )
    print(table.to_string(index=False))
    print("  AUC-PR (average precision) — метрика для дисбаланса: чем выше, тем лучше предсказание класса «дождь».")
    return results


def plot_roc_curves(
    df: pd.DataFrame,
    use_smote: bool = True,
    best_depth: int = None,
    best_forest_depth: int = None,
    best_n_estimators: int = None,
    best_mlp_hidden: tuple = None,
    best_mlp_max_iter: int = None,
) -> None:
    """ROC-кривые для всех четырёх моделей."""
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if use_smote:
        X_train_s, y_train = smote_train(X_train_s, y_train)

    models = [
        (build_logistic(), "LogisticRegression"),
        (build_tree(best_depth), "DecisionTree"),
        (build_forest(best_forest_depth, best_n_estimators), "RandomForest"),
        (build_mlp(best_mlp_hidden, best_mlp_max_iter), "MLP"),
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, name in models:
        model.fit(X_train_s, y_train)
        RocCurveDisplay.from_estimator(model, X_test_s, y_test, ax=ax, name=name)

    ax.set_title("ROC-кривые классификаторов")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task7_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task7_roc_curves.png")


def run_task7(df: pd.DataFrame, features: list, use_smote: bool = True) -> None:
    """
    Задание 7: классификаторы и оценка качества.
    features — список признаков, уже выбранных на шаге отбора.
    """
    print("\n" + "=" * 60)
    print("Курсовая. Задание 7: классификаторы")
    print("=" * 60)

    set_features(features)
    print(f"\nПризнаки для классификации: {features}")
    print("Балансировка: class_weight='balanced' (LogReg, Tree, Forest) + SMOTE на train по умолчанию.")

    print("\n--- Шаг 1: поиск оптимальной глубины дерева (1–20) ---")
    best_depth = find_optimal_tree_depth(df, use_smote=use_smote)

    print("\n--- Шаг 2: перебор гиперпараметров Random Forest (глубина × число деревьев) ---")
    best_forest_depth, best_n_estimators = find_optimal_forest_params(df, use_smote=use_smote)

    print("\n--- Шаг 3: перебор гиперпараметров MLP (нейроны × эпохи) ---")
    best_mlp_hidden, best_mlp_max_iter = sweep_mlp_hyperparams(df, use_smote=use_smote)

    print("\n--- Шаг 4: обучение всех моделей с оптимальными параметрами ---")
    run_classifiers(
        df,
        use_smote=use_smote,
        best_depth=best_depth,
        best_forest_depth=best_forest_depth,
        best_n_estimators=best_n_estimators,
        best_mlp_hidden=best_mlp_hidden,
        best_mlp_max_iter=best_mlp_max_iter,
    )

    print("\n--- Шаг 5: ROC-кривые ---")
    plot_roc_curves(
        df,
        use_smote=use_smote,
        best_depth=best_depth,
        best_forest_depth=best_forest_depth,
        best_n_estimators=best_n_estimators,
        best_mlp_hidden=best_mlp_hidden,
        best_mlp_max_iter=best_mlp_max_iter,
    )

    print("\nЗадание 7 завершено. При дисбалансе классов ориентироваться на F1 и AUC, а не только на Accuracy.")
