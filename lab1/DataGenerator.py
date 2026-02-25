"""
Генерация модельных наборов данных для лабораторной работы 1.
"""
import numpy as np


def norm_dataset(mu, sigma, N):
    """
    Генерирует данные двух классов с нормальным распределением по каждому признаку.

    Параметры:
        mu: список [mu0, mu1] — средние для класса 0 и класса 1 (каждый — список по признакам)
        sigma: список [sigma0, sigma1] — стандартные отклонения для каждого класса
        N: число объектов в каждом классе

    Возвращает:
        X: массив признаков (2*N, col), уже перемешанный
        Y: метки классов (2*N,) — 0 и 1
        class0, class1: массивы по классам для визуализации
    """
    mu0 = mu[0]
    mu1 = mu[1]
    sigma0 = sigma[0]
    sigma1 = sigma[1]
    col = len(mu0)

    # Первый столбец (признак 0)
    class0 = np.random.normal(mu0[0], sigma0[0], (N, 1))
    class1 = np.random.normal(mu1[0], sigma1[0], (N, 1))
    for i in range(1, col):
        v0 = np.random.normal(mu0[i], sigma0[i], (N, 1))
        class0 = np.hstack((class0, v0))
        v1 = np.random.normal(mu1[i], sigma1[i], (N, 1))
        class1 = np.hstack((class1, v1))

    Y0 = np.zeros((N, 1), dtype=bool)
    Y1 = np.ones((N, 1), dtype=bool)

    X = np.vstack((class0, class1))
    Y = np.vstack((Y0, Y1)).ravel()

    rng = np.random.default_rng()
    arr = np.arange(2 * N)
    rng.shuffle(arr)
    X = X[arr]
    Y = Y[arr]

    return X, Y, class0, class1


def nonlinear_dataset_5(N=1000):
    """
    Генерирует двумерные данные варианта 5: вложенные эллипсы (внешний и внутренний класс).

    Класс 0 — точки в большой эллиптической области.
    Класс 1 — точки в меньшей эллиптической области внутри первой (миндалевидная форма).

    Параметры:
        N: число объектов в каждом классе

    Возвращает:
        X, Y, class0, class1 — аналогично norm_dataset
    """
    # Параметры эллипсов (полуоси). Внешний больше внутреннего.
    a_outer, b_outer = 2.5, 1.8
    a_inner, b_inner = 1.0, 0.6
    center = (0.0, 0.0)
    noise = 0.08

    rng = np.random.default_rng()

    # Класс 0: отбор точек внутри внешнего эллипса (rejection sampling)
    class0_list = []
    while len(class0_list) < N:
        x = rng.uniform(-a_outer * 1.2, a_outer * 1.2)
        y = rng.uniform(-b_outer * 1.2, b_outer * 1.2)
        if (x - center[0]) ** 2 / a_outer**2 + (y - center[1]) ** 2 / b_outer**2 < 1:
            x += rng.normal(0, noise)
            y += rng.normal(0, noise)
            class0_list.append([x, y])
    class0 = np.array(class0_list)

    # Класс 1: точки внутри внутреннего эллипса
    class1_list = []
    while len(class1_list) < N:
        x = rng.uniform(-a_inner, a_inner)
        y = rng.uniform(-b_inner, b_inner)
        if (x - center[0]) ** 2 / a_inner**2 + (y - center[1]) ** 2 / b_inner**2 < 1:
            x += rng.normal(0, noise)
            y += rng.normal(0, noise)
            class1_list.append([x, y])
    class1 = np.array(class1_list)

    Y0 = np.zeros((N,), dtype=bool)
    Y1 = np.ones((N,), dtype=bool)

    X = np.vstack((class0, class1))
    Y = np.concatenate((Y0, Y1))

    arr = np.arange(2 * N)
    rng.shuffle(arr)
    X = X[arr]
    Y = Y[arr]

    return X, Y, class0, class1
