"""
Лабораторная работа 1. Генерация модельных наборов данных.
Запуск: из папки lab1 — python main.py; из корня leti — python -m lab1.main
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Папка для сохранения графиков (рядом со скриптом)
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

try:
    import DataGenerator as dg
except ImportError:
    import lab1.DataGenerator as dg

# --- Параметры по методичке (3 признака) ---
N = 1000
mu0 = [0, 2, 3]
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]
mu = [mu0, mu1]
sigma = [sigma0, sigma1]

# Генерация нормального датасета
X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
col = len(mu0)

# Разбиение на обучающую и тестовую подвыборки 70/30
train_count = round(0.7 * 2 * N)
Xtrain = X[0:train_count]
Xtest = X[train_count : 2 * N]
Ytrain = Y[0:train_count]
Ytest = Y[train_count : 2 * N]

# --- Визуализация нормального датасета ---
# Гистограммы по каждому признаку
for i in range(col):
    plt.figure()
    plt.hist(class0[:, i], bins="auto", alpha=0.7, label="Класс 0")
    plt.hist(class1[:, i], bins="auto", alpha=0.7, label="Класс 1")
    plt.xlabel("Значение признака")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма признака {i + 1}")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"hist_{i + 1}.png")
    plt.show()

# Скатерограмма (признаки 0 и 2)
plt.figure()
plt.scatter(class0[:, 0], class0[:, 2], marker=".", alpha=0.7, label="Класс 0")
plt.scatter(class1[:, 0], class1[:, 2], marker=".", alpha=0.7, label="Класс 1")
plt.xlabel("Признак 1")
plt.ylabel("Признак 3")
plt.title("Скатерограмма: признаки 1 и 3 (нормальный датасет)")
plt.legend()
plt.savefig(FIGURES_DIR / "scatter_norm.png")
plt.show()

# --- Нелинейный датасет (вариант 5) ---
Xn, Yn, class0_nl, class1_nl = dg.nonlinear_dataset_5(N)

plt.figure()
plt.scatter(class0_nl[:, 0], class0_nl[:, 1], marker=".", alpha=0.7, label="Класс 0")
plt.scatter(class1_nl[:, 0], class1_nl[:, 1], marker=".", alpha=0.7, label="Класс 1")
plt.xlabel("Признак X")
plt.ylabel("Признак Y")
plt.title("Нелинейный датасет (вариант 5): вложенные области")
plt.legend()
plt.savefig(FIGURES_DIR / "scatter_nonlinear_5.png")
plt.show()
