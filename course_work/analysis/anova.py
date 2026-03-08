"""
Дисперсионный анализ ANOVA (Задание 6).
Для каждого выбранного признака: различаются ли средние между классами RainTomorrow.
"""
import numpy as np
import pandas as pd
from scipy import stats

from course_work.config import COL_TARGET


def _get_groups(df: pd.DataFrame, col: str) -> list:
    """Группы по целевой переменной для заданного признака."""
    return [group[col].dropna().values for _, group in df.groupby(COL_TARGET)]


def _check_assumptions(df: pd.DataFrame, col: str) -> None:
    groups = _get_groups(df, col)
    if len(groups) < 2:
        print("  Недостаточно групп для проверки.")
        return
    stat, p = stats.levene(*groups)
    print(f"  Тест Левена ({col}): статистика = {stat:.4f}, p = {p:.4f}")
    if p > 0.05:
        print("    Предпосылка однородности дисперсий не отвергается (p > 0.05).")
    else:
        print("    Внимание: дисперсии могут различаться (p <= 0.05).")


def _run_anova_single(df: pd.DataFrame, col: str) -> None:
    groups = _get_groups(df, col)
    if len(groups) < 2:
        print(f"  Недостаточно групп для ANOVA ({col}).")
        return

    print(f"\n  H₀: средние значения {col} не различаются между классами {COL_TARGET}.")
    print(f"  H₁: хотя бы одна пара групп различается по среднему.")

    F, p = stats.f_oneway(*groups)
    print(f"  F-статистика = {F:.4f}, p-value = {p:.4f}")

    if p < 0.05:
        print(f"  Вывод: H₀ отвергается (p < 0.05). Средние {col} значимо различаются между классами.")
    else:
        print(f"  Вывод: H₀ не отвергается. Существенных различий средних не выявлено.")

    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    print(f"  Размер эффекта (eta-squared) = {eta_sq:.4f}.")


def run_task6(df: pd.DataFrame, features: list) -> None:
    """Задание 6: дисперсионный анализ для каждого выбранного признака."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 6: дисперсионный анализ (ANOVA)")
    print("=" * 60)
    print(f"  Анализируемые признаки: {features}")

    for col in features:
        print(f"\n--- {col} ---")
        print("  Предпосылки:")
        _check_assumptions(df, col)
        _run_anova_single(df, col)

    print("\n  Примечание: при двух группах (Yes/No) ANOVA эквивалентна t-test Стьюдента.")
    print("\nЗадание 6 завершено.")
