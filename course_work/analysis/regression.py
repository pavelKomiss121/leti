"""
Регрессионный анализ (Задание 4).
Для каждой пары выбранных признаков: простая линейная регрессия A от B и B от A.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from course_work.config import FIGURES_DIR, COL_TARGET, TEST_SIZE, RANDOM_STATE


def _split(X: pd.DataFrame, y: pd.Series, stratify: pd.Series):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )


def _ols_summary(X: np.ndarray, y: np.ndarray) -> object:
    try:
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        return sm.OLS(y, X_const).fit()
    except ImportError:
        return None


def _fit_and_print(df: pd.DataFrame, x_col: str, y_col: str, idx: int) -> dict:
    """Одна регрессия y_col от x_col: метрики и OLS-сводка."""
    stratify = df[COL_TARGET]
    X = df[[x_col]]
    y = df[y_col]
    X_tr, X_te, y_tr, y_te = _split(X, y, stratify)

    lr = LinearRegression().fit(X_tr, y_tr)
    result = {
        "r2_train": r2_score(y_tr, lr.predict(X_tr)),
        "r2_test": r2_score(y_te, lr.predict(X_te)),
        "mse_test": mean_squared_error(y_te, lr.predict(X_te)),
        "rmse_test": np.sqrt(mean_squared_error(y_te, lr.predict(X_te))),
        "coef": lr.coef_.ravel(),
        "intercept": lr.intercept_,
    }
    ols = _ols_summary(
        X_tr.to_numpy() if hasattr(X_tr, "to_numpy") else X_tr,
        y_tr.to_numpy() if hasattr(y_tr, "to_numpy") else y_tr,
    )

    print(f"\n--- Регрессия {idx}: {y_col} от {x_col} ---")
    print(f"  R² train = {result['r2_train']:.4f}, R² test = {result['r2_test']:.4f}")
    print(f"  MSE test = {result['mse_test']:.4f}, RMSE test = {result['rmse_test']:.4f}")
    print(f"  Коэффициенты: intercept = {result['intercept']:.4f}, {x_col} = {result['coef'][0]:.4f}")
    if ols is not None:
        print(ols.summary().tables[1].as_text())
    return result


def plot_residuals(df: pd.DataFrame, features: list) -> None:
    """График остатков для каждой пары выбранных признаков."""
    pairs = list(combinations(features, 2))
    n_plots = len(pairs) * 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    stratify = df[COL_TARGET]
    idx = 0
    for a, b in pairs:
        for y_col, x_col in [(a, b), (b, a)]:
            X = df[[x_col]]
            y = df[y_col]
            X_tr, X_te, y_tr, y_te = _split(X, y, stratify)
            lr = LinearRegression().fit(X_tr, y_tr)
            y_pred = lr.predict(X_te)
            residuals = y_te.to_numpy() - y_pred
            axes[idx].scatter(y_pred, residuals, alpha=0.3, s=10)
            axes[idx].axhline(0, color="k", linestyle="--")
            axes[idx].set_xlabel("Предсказание")
            axes[idx].set_ylabel("Остаток")
            axes[idx].set_title(f"Остатки: {y_col} от {x_col}")
            axes[idx].grid(True, alpha=0.3)
            idx += 1
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task4_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task4_residuals.png")


def run_task4(df: pd.DataFrame, features: list) -> None:
    """Задание 4: регрессионный анализ для выбранных признаков (попарно)."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 4: регрессионный анализ")
    print("=" * 60)
    print(f"  Анализируемые признаки: {features}")

    pairs = list(combinations(features, 2))
    reg_idx = 1
    for a, b in pairs:
        _fit_and_print(df, b, a, reg_idx)
        reg_idx += 1
        _fit_and_print(df, a, b, reg_idx)
        reg_idx += 1

    plot_residuals(df, features)
    print("\nЗадание 4 завершено.")
