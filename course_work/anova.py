"""
Дисперсионный анализ ANOVA (Задание 6).
Один файл — одна задача: H₀, предпосылки, F и p-value, размер эффекта (eta-squared).
"""
import numpy as np
import pandas as pd
from scipy import stats

COL_N = "Humidity3pm"
COL_TARGET = "RainTomorrow"


def _get_groups(df: pd.DataFrame) -> list:
    """Группы по целевой переменной для параметра n (Humidity3pm)."""
    return [group[COL_N].dropna().values for _, group in df.groupby(COL_TARGET)]


def check_assumptions(df: pd.DataFrame) -> None:
    """Краткая проверка предпосылок: однородность дисперсий (Левен)."""
    groups = _get_groups(df)
    if len(groups) < 2:
        print("  Недостаточно групп для проверки.")
        return
    stat, p = stats.levene(*groups)
    print(f"  Тест Левена (однородность дисперсий): статистика = {stat:.4f}, p = {p:.4f}")
    if p > 0.05:
        print("  Предпосылка однородности дисперсий не отвергается (p > 0.05).")
    else:
        print("  Внимание: дисперсии могут различаться (p <= 0.05).")


def run_anova(df: pd.DataFrame) -> None:
    """Однофакторный ANOVA: различаются ли средние Humidity3pm между классами RainTomorrow."""
    groups = _get_groups(df)
    if len(groups) < 2:
        print("Недостаточно групп для ANOVA.")
        return

    # H₀: средние значения параметра n не различаются между классами RainTomorrow
    print("\n  H₀: средние значения Humidity3pm (n) не различаются между классами RainTomorrow.")
    print("  H₁: хотя бы одна пара групп различается по среднему.")

    F, p = stats.f_oneway(*groups)
    print(f"\n  F-статистика = {F:.4f}, p-value = {p:.4f}")

    if p < 0.05:
        print("  Вывод: H₀ отвергается (p < 0.05). Средние Humidity3pm значимо различаются между классами.")
    else:
        print("  Вывод: H₀ не отвергается. Существенных различий средних не выявлено.")

    # Размер эффекта eta-squared: SS_between / SS_total
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    print(f"  Размер эффекта (eta-squared) = {eta_sq:.4f}.")

    print("\n  Примечание: при двух группах (RainTomorrow Yes/No) ANOVA эквивалентна t-test Стьюдента.")


def run_task6(df: pd.DataFrame) -> None:
    """Задание 6: дисперсионный анализ."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 6: дисперсионный анализ (ANOVA)")
    print("=" * 60)

    print("\n--- Предпосылки ---")
    check_assumptions(df)

    print("\n--- ANOVA: Humidity3pm по классам RainTomorrow ---")
    run_anova(df)

    print("\nЗадание 6 завершено.")
