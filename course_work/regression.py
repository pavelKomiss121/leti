"""
Регрессионный анализ (Задание 4).
Один файл — одна задача: простая линейная регрессия n от m и m от n, метрики, значимость.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COL_N = "Humidity3pm"
COL_M = "Rainfall"
COL_TARGET = "RainTomorrow"
TEST_SIZE = 0.3  # 70% train / 30% test, как в лабораторных
RANDOM_STATE = 42


def _split(X: pd.DataFrame, y: pd.Series, stratify: pd.Series):
    """Train/test с сохранением доли классов (stratify по RainTomorrow)."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )


def _ols_summary(X: np.ndarray, y: np.ndarray, feature_names: list) -> object:
    """Статистическая значимость коэффициентов через statsmodels OLS."""
    try:
        import statsmodels.api as sm

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        return model
    except ImportError:
        return None


def _fit_simple(X_train, y_train, X_test, y_test, x_name: str, y_name: str) -> dict:
    """Одна простая линейная регрессия: метрики и OLS-сводка."""
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    result = {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "mse_test": mean_squared_error(y_test, y_pred_test),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "coef": lr.coef_.ravel(),
        "intercept": lr.intercept_,
        "ols": None,
    }

    ols = _ols_summary(
        X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train,
        y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train,
        [x_name],
    )
    result["ols"] = ols
    return result


def run_simple_regressions(df: pd.DataFrame) -> None:
    """Простая линейная регрессия: (1) n от m, (2) m от n. Метрики и значимость."""
    stratify = df[COL_TARGET]

    # 1) n (Humidity3pm) от m (Rainfall)
    X_n = df[[COL_M]]
    y_n = df[COL_N]
    X_tr, X_te, y_tr, y_te = _split(X_n, y_n, stratify)
    res_n = _fit_simple(X_tr, y_tr, X_te, y_te, COL_M, COL_N)

    print("\n--- Регрессия 1: n (Humidity3pm) от m (Rainfall) ---")
    print(f"  R² train = {res_n['r2_train']:.4f}, R² test = {res_n['r2_test']:.4f}")
    print(f"  MSE test = {res_n['mse_test']:.4f}, RMSE test = {res_n['rmse_test']:.4f}")
    print(f"  Коэффициенты: intercept = {res_n['intercept']:.4f}, {COL_M} = {res_n['coef'][0]:.4f}")
    if res_n["ols"] is not None:
        print(res_n["ols"].summary().tables[1].as_text())

    # 2) m (Rainfall) от n (Humidity3pm)
    X_m = df[[COL_N]]
    y_m = df[COL_M]
    X_tr, X_te, y_tr, y_te = _split(X_m, y_m, stratify)
    res_m = _fit_simple(X_tr, y_tr, X_te, y_te, COL_N, COL_M)

    print("\n--- Регрессия 2: m (Rainfall) от n (Humidity3pm) ---")
    print(f"  R² train = {res_m['r2_train']:.4f}, R² test = {res_m['r2_test']:.4f}")
    print(f"  MSE test = {res_m['mse_test']:.4f}, RMSE test = {res_m['rmse_test']:.4f}")
    print(f"  Коэффициенты: intercept = {res_m['intercept']:.4f}, {COL_N} = {res_m['coef'][0]:.4f}")
    if res_m["ols"] is not None:
        print(res_m["ols"].summary().tables[1].as_text())

    print("\n(Ожидаемо низкий R² ~0.05–0.15: связь Rainfall и Humidity не строго линейна.)")


def plot_residuals(df: pd.DataFrame) -> None:
    """Опционально: график остатков (предсказания vs остатки) для обеих регрессий."""
    stratify = df[COL_TARGET]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (y_name, x_name) in enumerate([(COL_N, COL_M), (COL_M, COL_N)]):
        X = df[[x_name]]
        y = df[y_name]
        X_tr, X_te, y_tr, y_te = _split(X, y, stratify)
        lr = LinearRegression().fit(X_tr, y_tr)
        y_pred = lr.predict(X_te)
        residuals = y_te.to_numpy() - y_pred

        axes[idx].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[idx].axhline(0, color="k", linestyle="--")
        axes[idx].set_xlabel("Предсказание")
        axes[idx].set_ylabel("Остаток")
        axes[idx].set_title(f"Остатки: {y_name} от {x_name}")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task4_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task4_residuals.png")


def run_task4(df: pd.DataFrame) -> None:
    """Задание 4: регрессионный анализ."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 4: регрессионный анализ")
    print("=" * 60)

    run_simple_regressions(df)
    plot_residuals(df)

    print("\nЗадание 4 завершено.")
