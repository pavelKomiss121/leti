"""
Корреляционный анализ (Задание 3).
Один файл — одна задача: тепловая карта, Пирсон/Спирмен, мультиколлинеарность.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COL_N = "Humidity3pm"
COL_M = "Rainfall"
COL_TARGET = "RainTomorrow"


def _get_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Числовые столбцы (без даты). Целевую при необходимости бинаризуем для корреляции."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    if COL_TARGET in df.columns and COL_TARGET not in numeric.columns:
        numeric[COL_TARGET] = (df[COL_TARGET] == "Yes").astype(int)
    return numeric


def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> None:
    """Тепловая карта корреляций числовых признаков."""
    num_df = _get_numeric_df(df)
    corr = num_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    ax.set_title(f"Тепловая карта корреляций ({method.capitalize()})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"task3_heatmap_{method}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: figures/task3_heatmap_{method}.png")


def print_correlation_coefficients(df: pd.DataFrame) -> None:
    """Коэффициенты Пирсона и Спирмена: Пирсон для близких к нормальным, Спирмен для скошенных (Rainfall)."""
    num_df = _get_numeric_df(df)

    print("\n--- Корреляция Пирсона (числовые признаки) ---")
    pearson = num_df.corr(method="pearson")
    print(pearson.round(3).to_string())

    print("\n--- Корреляция Спирмена (устойчива к скошенности, например Rainfall) ---")
    spearman = num_df.corr(method="spearman")
    print(spearman.round(3).to_string())

    # Ключевые пары: n, m и целевая
    if COL_TARGET in num_df.columns:
        print(f"\n--- Связь с целевой {COL_TARGET} (1=Yes) ---")
        for col in [COL_N, COL_M]:
            if col in num_df.columns:
                p = pearson.loc[col, COL_TARGET]
                s = spearman.loc[col, COL_TARGET]
                print(f"  {col}: Пирсон = {p:.3f}, Спирмен = {s:.3f}")


def print_multicollinearity(df: pd.DataFrame, threshold: float = 0.8) -> None:
    """Пары признаков с |r| > threshold (мультиколлинеарность)."""
    num_df = _get_numeric_df(df)
    corr = num_df.corr(method="pearson")

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                pairs.append((corr.columns[i], corr.columns[j], r))

    print(f"\n--- Мультиколлинеарность (|r| > {threshold}) ---")
    if not pairs:
        print("  Сильных пар не найдено.")
    else:
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"  {a} — {b}: r = {r:.3f}")
    print("  (При множественной регрессии в Задании 4 — учесть, при необходимости VIF.)")


def run_task3(df: pd.DataFrame) -> None:
    """Задание 3: корреляционный анализ."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 3: корреляционный анализ")
    print("=" * 60)

    plot_correlation_heatmap(df, method="pearson")
    plot_correlation_heatmap(df, method="spearman")
    print_correlation_coefficients(df)
    print_multicollinearity(df, threshold=0.8)

    print("\nЗадание 3 завершено. Графики в каталоге figures/.")
