"""
Лабораторная работа 3. Деревья и леса решений.
Запуск из корня leti: python -m lab3.main
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Папка для сохранения графиков
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

try:
    import lab1.DataGenerator as dg
except ImportError:
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    import lab1.DataGenerator as dg

try:
    import scikitplot as skplt
except ImportError:
    skplt = None

N = 1000


def sensitivity_specificity(Y_true, Y_pred):
    """
    Класс 0 = отсутствие признака, класс 1 = наличие признака.
    Чувствительность = TP / (TP + FN). Специфичность = TN / (TN + FP).
    """
    Y_true = np.asarray(Y_true).ravel()
    Y_pred = np.asarray(Y_pred).ravel()
    Y_true = (Y_true != 0).astype(int)
    Y_pred = (Y_pred != 0).astype(int)
    TP = np.sum((Y_true == 1) & (Y_pred == 1))
    TN = np.sum((Y_true == 0) & (Y_pred == 0))
    FP = np.sum((Y_true == 0) & (Y_pred == 1))
    FN = np.sum((Y_true == 1) & (Y_pred == 0))
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return sens, spec


def split_70_30(X, Y):
    """Разбиение на обучающую (70%) и тестовую (30%) выборки."""
    n = len(Y)
    train_count = round(0.7 * n)
    return (
        X[:train_count], Y[:train_count],
        X[train_count:], Y[train_count:]
    )


def run_dataset(Xtrain, Ytrain, Xtest, Ytest, suffix, title_suffix):
    """
    Обучает дерево и лес на выборке, считает метрики, строит ROC и гистограммы для леса.
    Возвращает словари с метриками для таблиц и AUC.
    """
    # Дерево решений (п.1-3)
    tree = DecisionTreeClassifier(random_state=0).fit(Xtrain, Ytrain)
    tree_pred_train = tree.predict(Xtrain)
    tree_pred_test = tree.predict(Xtest)
    tree_proba_train = tree.predict_proba(Xtrain)
    tree_proba_test = tree.predict_proba(Xtest)

    acc_tree_train = tree.score(Xtrain, Ytrain)
    acc_tree_test = tree.score(Xtest, Ytest)
    sens_tree_train, spec_tree_train = sensitivity_specificity(Ytrain, tree_pred_train)
    sens_tree_test, spec_tree_test = sensitivity_specificity(Ytest, tree_pred_test)

    # Случайный лес (п.4)
    forest = RandomForestClassifier(random_state=0).fit(Xtrain, Ytrain)
    forest_pred_train = forest.predict(Xtrain)
    forest_pred_test = forest.predict(Xtest)
    forest_proba_train = forest.predict_proba(Xtrain)
    forest_proba_test = forest.predict_proba(Xtest)

    acc_forest_train = forest.score(Xtrain, Ytrain)
    acc_forest_test = forest.score(Xtest, Ytest)
    sens_forest_train, spec_forest_train = sensitivity_specificity(Ytrain, forest_pred_train)
    sens_forest_test, spec_forest_test = sensitivity_specificity(Ytest, forest_pred_test)

    # ROC-кривые для дерева и леса на одном графике (п.5)
    # Ytest для ROC в формате 0/1 (roc_auc_score и roc_curve принимают и bool)
    Ytest_int = (np.asarray(Ytest) != 0).astype(int) if Ytest.dtype == bool else np.asarray(Ytest)

    fpr_tree, tpr_tree, _ = roc_curve(Ytest_int, tree_proba_test[:, 1])
    fpr_forest, tpr_forest, _ = roc_curve(Ytest_int, forest_proba_test[:, 1])
    auc_tree = roc_auc_score(Ytest_int, tree_proba_test[:, 1])
    auc_forest = roc_auc_score(Ytest_int, forest_proba_test[:, 1])

    plt.figure(figsize=(10, 10))
    plt.plot(fpr_tree, tpr_tree, label=f"Дерево (AUC = {auc_tree:.4f})")
    plt.plot(fpr_forest, tpr_forest, label=f"Лес (AUC = {auc_forest:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Доля ложноположительных")
    plt.ylabel("Доля истинно положительных")
    plt.title(f"ROC-кривые ({title_suffix})")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"lab3_roc_{suffix}.png")
    plt.close()

    if skplt is not None:
        # Дополнительный вариант через scikit-plot (одна кривая — лес)
        skplt.metrics.plot_roc(Ytest_int, forest_proba_test, figsize=(10, 10))
        plt.title(f"ROC — случайный лес ({title_suffix})")
        plt.savefig(FIGURES_DIR / f"lab3_roc_forest_only_{suffix}.png")
        plt.close()

    # Гистограммы распределения вероятностей для леса — тест и трейн (п.6)
    plt.figure()
    plt.hist(forest_proba_test[Ytest, 1], bins="auto", alpha=0.7, label="Класс 1 (истинный)", color="C0")
    plt.hist(forest_proba_test[~Ytest, 1], bins="auto", alpha=0.7, label="Класс 0 (истинный)", color="C1")
    plt.xlabel("Вероятность принадлежности классу 1")
    plt.ylabel("Число объектов")
    plt.title(f"Случайный лес: результаты классификации, тест ({title_suffix})")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"lab3_hist_forest_test_{suffix}.png")
    plt.close()

    plt.figure()
    plt.hist(forest_proba_train[Ytrain, 1], bins="auto", alpha=0.7, label="Класс 1 (истинный)", color="C0")
    plt.hist(forest_proba_train[~Ytrain, 1], bins="auto", alpha=0.7, label="Класс 0 (истинный)", color="C1")
    plt.xlabel("Вероятность принадлежности классу 1")
    plt.ylabel("Число объектов")
    plt.title(f"Случайный лес: результаты классификации, трейн ({title_suffix})")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"lab3_hist_forest_train_{suffix}.png")
    plt.close()

    print(f"[{title_suffix}] AUC дерево: {auc_tree:.4f}, AUC лес: {auc_forest:.4f}")

    return {
        "tree": {
            "train": (len(Ytrain), acc_tree_train, sens_tree_train, spec_tree_train),
            "test": (len(Ytest), acc_tree_test, sens_tree_test, spec_tree_test),
            "auc": auc_tree,
        },
        "forest": {
            "train": (len(Ytrain), acc_forest_train, sens_forest_train, spec_forest_train),
            "test": (len(Ytest), acc_forest_test, sens_forest_test, spec_forest_test),
            "auc": auc_forest,
        },
    }


def print_table(results, model_name, dataset_name):
    """Печать таблицы метрик для одной модели (дерево или лес) по одной выборке."""
    print(f"\n--- {dataset_name} — {model_name} ---")
    print(f"{'':8} {'Число объектов':>14} {'Точность, %':>12} {'Чувствительность, %':>20} {'Специфичность, %':>18}")
    print("-" * 78)
    for part in ("train", "test"):
        label = "Train" if part == "train" else "Test"
        n, acc, sens, spec = results[model_name][part]
        print(f"{label:8} {n:>14} {acc*100:>11.2f}% {sens*100:>19.2f}% {spec*100:>17.2f}%")


# ---------- Выборка А: хорошо разделимые нормальные данные ----------
mu0_A = [0, 2, 3]
mu1_A = [3, 5, 1]
sigma0_A = [2, 1, 2]
sigma1_A = [1, 2, 1]
X_A, Y_A, _, _ = dg.norm_dataset([mu0_A, mu1_A], [sigma0_A, sigma1_A], N)
Xtrain_A, Ytrain_A, Xtest_A, Ytest_A = split_70_30(X_A, Y_A)
results_A = run_dataset(Xtrain_A, Ytrain_A, Xtest_A, Ytest_A, "A", "выборка А")
print_table(results_A, "tree", "Выборка А")
print_table(results_A, "forest", "Выборка А")

# ---------- Выборка Б: плохо разделимые нормальные данные (средняя степень пересечения) ----------
mu0_B = [1, 3, 2]
mu1_B = [2, 4, 3]
sigma0_B = [2.5, 2, 2.5]
sigma1_B = [2, 2.5, 2]
X_B, Y_B, _, _ = dg.norm_dataset([mu0_B, mu1_B], [sigma0_B, sigma1_B], N)
Xtrain_B, Ytrain_B, Xtest_B, Ytest_B = split_70_30(X_B, Y_B)
results_B = run_dataset(Xtrain_B, Ytrain_B, Xtest_B, Ytest_B, "B", "выборка Б")
print_table(results_B, "tree", "Выборка Б")
print_table(results_B, "forest", "Выборка Б")

# ---------- Выборка В: нелинейно разделимые данные ----------
X_C, Y_C, _, _ = dg.nonlinear_dataset_5(N)
Xtrain_C, Ytrain_C, Xtest_C, Ytest_C = split_70_30(X_C, Y_C)
results_C = run_dataset(Xtrain_C, Ytrain_C, Xtest_C, Ytest_C, "C", "выборка В")
print_table(results_C, "tree", "Выборка В")
print_table(results_C, "forest", "Выборка В")

# ---------- П.5 самостоятельной: подбор гиперпараметров дерева (глубина) для снижения переобучения ----------
print("\n--- Подбор max_depth дерева (выборка Б) ---")
best_tree = None
best_test_acc = -1
best_depth = None
for max_d in [3, 5, 10, 15, None]:
    tree_tune = DecisionTreeClassifier(max_depth=max_d, random_state=0).fit(Xtrain_B, Ytrain_B)
    acc_t = tree_tune.score(Xtrain_B, Ytrain_B)
    acc_v = tree_tune.score(Xtest_B, Ytest_B)
    depth_str = str(max_d) if max_d is not None else "None"
    print(f"  max_depth={depth_str}: train={acc_t*100:.2f}%, test={acc_v*100:.2f}%")
    if acc_v > best_test_acc:
        best_test_acc = acc_v
        best_depth = max_d
print(f"  Лучший max_depth по тестовой точности: {best_depth} (test accuracy = {best_test_acc*100:.2f}%)")

# ---------- П.6 самостоятельной: подбор n_estimators для леса (максимум AUC на тесте) ----------
# Диапазон от 1 до 300 с шагом 10: 1, 11, 21, ..., 291
print("\n--- Подбор n_estimators леса (от 1 до 300, шаг 10), выборка А ---")
best_n = 1
best_auc = 0.0
for n_est in range(1, 301, 10):
    rf = RandomForestClassifier(n_estimators=n_est, random_state=0).fit(Xtrain_B, Ytrain_B)
    proba = rf.predict_proba(Xtest_B)[:, 1]
    Ytest_B_int = (np.asarray(Ytest_B) != 0).astype(int)
    auc = roc_auc_score(Ytest_B_int, proba)
    print(f"  n_estimators={n_est}: AUC на тесте={best_auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_n = n_est
print(f"  Наилучшее n_estimators: {best_n}, AUC на тесте: {best_auc:.4f}")

print(f"\nГрафики сохранены в папке: {FIGURES_DIR}")
