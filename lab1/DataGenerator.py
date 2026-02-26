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
    Генерирует двумерные данные варианта 5: класс 0 обволакивает класс 1.

    Класс 1 — вытянутый овал в нижнем левом углу (большая ось снизу-слева вверх-вправо).

    Класс 0 — U-образная оболочка вокруг класса 1: внешний эллипс с «вырезом»
    в нижнем левом углу (открытие U), так что класс 0 огибает овал сверху, справа и снизу.

    Параметры:
        N: число объектов в каждом классе

    Возвращает:
        X, Y, class0, class1 — аналогично norm_dataset
    """
    rng = np.random.default_rng()
    # Шум при генерации: добавляется к каждой точке. Больше — размытее границы.
    noise = 0.04

    # ========== Класс 1 (внутренний овал) ==========
    # cx, cy — центр овала класса 1 на плоскости (X, Y). Сдвиг меняет положение овала.
    cx, cy = 0.25, -0.3
    # angle_deg — угол поворота овала в градусах (0 = горизонтально, 90 = вертикально).
    # Оба класса используют один и тот же угол.
    angle_deg = 28
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    def to_local(x, y, cx_=None, cy_=None):
        """Перевод (x,y) в локальные координаты эллипса: u вдоль большой оси, v вдоль малой."""
        if cx_ is None:
            cx_, cy_ = cx, cy
        u = (x - cx_) * cos_a + (y - cy_) * sin_a
        v = -(x - cx_) * sin_a + (y - cy_) * cos_a
        return u, v

    def inside_ellipse(x, y, a, b, cx_=None, cy_=None):
        if cx_ is None:
            cx_, cy_ = cx, cy
        u, v = to_local(x, y, cx_, cy_)
        return (u**2 / a**2 + v**2 / b**2) < 1

    # a1 — большая полуось овала класса 1 (длина). b1 — малая полуось (толщина).
    # Меньше b1 при том же a1 — овал более сплюснутый (тоньше).
    a1, b1 = 0.5, 0.14
    class1_list = []
    while len(class1_list) < N:
        x = rng.uniform(cx - a1 - 0.2, cx + a1 + 0.2)
        y = rng.uniform(cy - a1 - 0.2, cy + a1 + 0.2)
        if inside_ellipse(x, y, a1, b1):
            x += rng.normal(0, noise)
            y += rng.normal(0, noise)
            class1_list.append([x, y])
    class1 = np.array(class1_list)

    # ========== Класс 0 (U-образная оболочка) ==========
    # cx0, cy0 — центр оболочки. Задаётся сдвигом от центра класса 1 (cx, cy).
    #   cx0 = cx + ... — сдвиг вправо (больше = оболочка правее).
    #   cy0 = cy + ... — сдвиг вверх (больше = оболочка выше). Отрицательное — ниже.
    cx0, cy0 = cx + 0.2, cy + 0.12
    # a2, b2 — полуоси внешнего эллипса оболочки (её внешняя граница).
    #   a2 — вдоль «длины» U, b2 — вдоль «высоты». Меньше b2 — U более сплюснутая.
    a2, b2 = 1.2, 0.50
    # a_inner_gap, b_inner_gap — полуоси внутренней границы оболочки (где заканчивается класс 0).
    #   Больше этих значений — больший зазор между классом 0 и классом 1 (оболочка дальше от овала).
    #   Меньше — оболочка ближе к овалу, зазор тоньше.
    a_inner_gap, b_inner_gap = 0.5, 0.21
    # u_cut — граница «выреза» U в локальных координатах оболочки (ось u вдоль большой оси).
    #   Точки с u < u_cut отбрасываются → получается открытие U. Больше u_cut (например 0) — вырез шире,
    #   оболочка короче слева. Меньше u_cut (например -0.5) — вырез уже, оболочка длиннее слева.
    u_cut = -0.25

    class0_list = []
    while len(class0_list) < N:
        x = rng.uniform(cx0 - a2 - 0.15, cx0 + a2 + 0.15)
        y = rng.uniform(cy0 - b2 - 0.15, cy0 + b2 + 0.15)
        u, v = to_local(x, y, cx0, cy0)
        # Внутри внешнего эллипса, снаружи внутренней границы оболочки, и не в «вырезе» (u >= u_cut)
        in_outer = inside_ellipse(x, y, a2, b2, cx0, cy0)
        in_inner = inside_ellipse(x, y, a_inner_gap, b_inner_gap, cx0, cy0)
        not_in_cut = u >= u_cut
        if in_outer and not in_inner and not_in_cut:
            x += rng.normal(0, noise)
            y += rng.normal(0, noise)
            class0_list.append([x, y])
    class0 = np.array(class0_list)

    Y0 = np.zeros((N,), dtype=bool)
    Y1 = np.ones((N,), dtype=bool)

    X = np.vstack((class0, class1))
    Y = np.concatenate((Y0, Y1))

    arr = np.arange(2 * N)
    rng.shuffle(arr)
    X = X[arr]
    Y = Y[arr]

    return X, Y, class0, class1
