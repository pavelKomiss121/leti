"""
Визуализация и графики (Задание 2 и далее).
Один файл — одна задача: только построение графиков и сохранение в figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Параметры курсовой: n, m, целевая переменная
COL_N = "Humidity3pm"
COL_M = "Rainfall"
COL_TARGET = "RainTomorrow"


def _setup_plot(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Подписи осей, сетка, легенда (если есть)."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_distributions_by_class(df: pd.DataFrame) -> None:
    """Распределение n (Humidity3pm) и m (Rainfall) по классам RainTomorrow."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(data=df, x=COL_N, hue=COL_TARGET, kde=True, ax=axes[0], alpha=0.6)
    _setup_plot(axes[0], "Распределение Humidity3pm по классам", "Humidity3pm (%)", "Плотность / число")
    if axes[0].get_legend():
        axes[0].get_legend().set_title(COL_TARGET)

    sns.histplot(data=df, x=COL_M, hue=COL_TARGET, kde=True, ax=axes[1], alpha=0.6)
    _setup_plot(axes[1], "Распределение Rainfall по классам", "Rainfall (мм)", "Плотность / число")
    if axes[1].get_legend():
        axes[1].get_legend().set_title(COL_TARGET)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_distributions_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task2_distributions_by_class.png")


def print_descriptive_stats(df: pd.DataFrame) -> None:
    """Медиана, среднее, СКО для n и m: вся выборка и по классам."""
    print("\n--- Описательная статистика (n = Humidity3pm, m = Rainfall) ---")
    for col, name in [(COL_N, "Humidity3pm (n)"), (COL_M, "Rainfall (m)")]:
        stats_all = df[col].agg(["mean", "median", "std"])
        print(f"\n{name} — вся выборка: mean={stats_all['mean']:.4f}, median={stats_all['median']:.4f}, std={stats_all['std']:.4f}")
        by_class = df.groupby(COL_TARGET)[col].agg(["mean", "median", "std"])
        print(by_class.to_string())


def plot_above_below_median(df: pd.DataFrame) -> None:
    """Разбивка выше/ниже медианы по n и по m: гистограмма, scatter, boxplot; оценка СКО."""
    for col, name, unit in [(COL_N, "Humidity3pm", "%"), (COL_M, "Rainfall", "мм")]:
        median_val = df[col].median()
        above = df[df[col] > median_val]
        below = df[df[col] <= median_val]
        std_above, std_below = above[col].std(), below[col].std()
        print(f"\n{name}: медиана = {median_val:.4f}. СКО выше медианы: {std_above:.4f}, ниже: {std_below:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        sns.histplot(above[col], label="Выше медианы", ax=axes[0], alpha=0.7, color="C1")
        sns.histplot(below[col], label="Ниже медианы", ax=axes[0], alpha=0.7, color="C0")
        _setup_plot(axes[0], f"Гистограмма {name}", name, "Число")
        axes[0].legend()

        # Scatter: n vs m, цвет по выше/ниже медианы (выборка для скорости)
        sample = df.sample(n=min(5000, len(df)), random_state=42)
        above_s = sample[sample[col] > median_val]
        below_s = sample[sample[col] <= median_val]
        axes[1].scatter(below_s[COL_N], below_s[COL_M], alpha=0.3, s=10, label="Ниже медианы", c="C0")
        axes[1].scatter(above_s[COL_N], above_s[COL_M], alpha=0.3, s=10, label="Выше медианы", c="C1")
        _setup_plot(axes[1], f"Scatter (n vs m), разбивка по медиане {name}", COL_N, COL_M)
        axes[1].legend()

        sns.boxplot(data=df, x=COL_TARGET, y=col, ax=axes[2])
        _setup_plot(axes[2], f"Boxplot {name} по классам", COL_TARGET, f"{name} ({unit})")

        plt.suptitle(f"Выше/ниже медианы {name}")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(FIGURES_DIR / f"task2_above_below_median_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранено: figures/task2_above_below_median_{safe_name}.png")


def plot_class_balance(df: pd.DataFrame) -> None:
    """График сбалансированности классов (pie или barplot)."""
    counts = df[COL_TARGET].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts.plot(kind="bar", ax=axes[0], color=["C0", "C1"])
    _setup_plot(axes[0], "Баланс классов RainTomorrow", COL_TARGET, "Число наблюдений")

    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["C0", "C1"])
    axes[1].set_title("Доля классов (%)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task2_class_balance.png")


def run_task2(df: pd.DataFrame) -> None:
    """Задание 2: все графики и вывод описательной статистики."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 2: визуализация и описательная статистика")
    print("=" * 60)

    plot_distributions_by_class(df)
    print_descriptive_stats(df)
    plot_above_below_median(df)
    plot_class_balance(df)

    print("\nЗадание 2 завершено. Графики в каталоге figures/.")
