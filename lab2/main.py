"""
Лабораторная работа 2. Классификатор на основе логистической регрессии.
Запуск из корня leti: python -m lab2.main
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Папка для сохранения графиков
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

try:
    import lab1.DataGenerator as dg
except ImportError:
    import DataGenerator as dg

Nvar = 5  # номер варианта (для random_state)
N = 1000


def sensitivity_specificity(Y_true, Y_pred):
    """
    Класс 0 = отсутствие признака, класс 1 = наличие признака.
    Чувствительность = TP / (TP + FN). Специфичность = TN / (TN + FP).
    Y_pred — метки 0/1 (результат predict).
    """
    Y_true = np.asarray(Y_true).ravel()
    Y_pred = np.asarray(Y_pred).ravel()
    # Приводим к int для надёжного сравнения
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


def run_experiment(Xtrain, Ytrain, Xtest, Ytest, suffix, title_suffix):
    """
    Обучает логистическую регрессию (SAGA), считает метрики,
    строит гистограммы вероятностей для train и test.
    """
    clf = LogisticRegression(random_state=Nvar, solver="saga").fit(Xtrain, Ytrain)

    Pred_train = clf.predict(Xtrain)
    Pred_test = clf.predict(Xtest)
    Pred_train_proba = clf.predict_proba(Xtrain)
    Pred_test_proba = clf.predict_proba(Xtest)

    acc_train = clf.score(Xtrain, Ytrain)
    acc_test = clf.score(Xtest, Ytest)
    acc_test_manual = np.sum(Pred_test == Ytest) / len(Ytest)

    sens_train, spec_train = sensitivity_specificity(Ytrain, Pred_train)
    sens_test, spec_test = sensitivity_specificity(Ytest, Pred_test)

    # Гистограммы: вероятность класса 1 для объектов с истинной меткой 0 и 1
    # Тест
    plt.figure()
    plt.hist(Pred_test_proba[Ytest, 1], bins="auto", alpha=0.7, label="Класс 1 (истинный)", color="C0")
    plt.hist(Pred_test_proba[~Ytest, 1], bins="auto", alpha=0.7, label="Класс 0 (истинный)", color="C1")
    plt.xlabel("Вероятность принадлежности классу 1")
    plt.ylabel("Число объектов")
    plt.title(f"Результаты классификации, тест ({title_suffix})")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"lab2_hist_test_{suffix}.png")
    plt.close()

    # Трейн
    plt.figure()
    plt.hist(Pred_train_proba[Ytrain, 1], bins="auto", alpha=0.7, label="Класс 1 (истинный)", color="C0")
    plt.hist(Pred_train_proba[~Ytrain, 1], bins="auto", alpha=0.7, label="Класс 0 (истинный)", color="C1")
    plt.xlabel("Вероятность принадлежности классу 1")
    plt.ylabel("Число объектов")
    plt.title(f"Результаты классификации, трейн ({title_suffix})")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"lab2_hist_train_{suffix}.png")
    plt.close()

    return {
        "train": (len(Ytrain), acc_train, sens_train, spec_train),
        "test": (len(Ytest), acc_test, sens_test, spec_test),
    }


def print_table(results, dataset_name):
    """Выводит таблицу метрик для одной выборки (Train/Test)."""
    print(f"\n--- {dataset_name} ---")
    print(f"{'':8} {'Число объектов':>14} {'Точность, %':>12} {'Чувствительность, %':>20} {'Специфичность, %':>18}")
    print("-" * 78)
    for part in ("train", "test"):
        label = "Train" if part == "train" else "Test"
        n, acc, sens, spec = results[part]
        print(f"{label:8} {n:>14} {acc*100:>11.2f}% {sens*100:>19.2f}% {spec*100:>17.2f}%")


# ---------- Выборка А: хорошо разделимые нормальные данные ----------
mu0_A = [0, 2, 3]
mu1_A = [3, 5, 1]
sigma0_A = [2, 1, 2]
sigma1_A = [1, 2, 1]
X_A, Y_A, _, _ = dg.norm_dataset([mu0_A, mu1_A], [sigma0_A, sigma1_A], N)
Xtrain_A, Ytrain_A, Xtest_A, Ytest_A = split_70_30(X_A, Y_A)
results_A = run_experiment(Xtrain_A, Ytrain_A, Xtest_A, Ytest_A, "A", "выборка А")
print_table(results_A, "Выборка А (хорошо разделимые)")

# ---------- Выборка Б: плохо разделимые нормальные данные ----------
mu0_B = [1, 3, 2]
mu1_B = [2, 4, 3]
sigma0_B = [2.5, 2, 2.5]
sigma1_B = [2, 2.5, 2]
X_B, Y_B, _, _ = dg.norm_dataset([mu0_B, mu1_B], [sigma0_B, sigma1_B], N)
Xtrain_B, Ytrain_B, Xtest_B, Ytest_B = split_70_30(X_B, Y_B)
results_B = run_experiment(Xtrain_B, Ytrain_B, Xtest_B, Ytest_B, "B", "выборка Б")
print_table(results_B, "Выборка Б (плохо разделимые)")

# ---------- Выборка В: нелинейно разделимые данные ----------
X_C, Y_C, _, _ = dg.nonlinear_dataset_5(N)
Xtrain_C, Ytrain_C, Xtest_C, Ytest_C = split_70_30(X_C, Y_C)
results_C = run_experiment(Xtrain_C, Ytrain_C, Xtest_C, Ytest_C, "C", "выборка В")
print_table(results_C, "Выборка В (нелинейно разделимые)")

print(f"\nГистограммы сохранены в папке: {FIGURES_DIR}")
