"""
Визуализация и графики (Задание 2).
Работает с выбранными признаками (передаются как параметр features).
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from course_work.config import FIGURES_DIR, COL_TARGET


def _setup_plot(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_distributions_by_class(df: pd.DataFrame, features: list) -> None:
    """Распределение каждого выбранного признака по классам RainTomorrow."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, features):
        sns.histplot(data=df, x=col, hue=COL_TARGET, kde=True, ax=ax, alpha=0.6)
        _setup_plot(ax, f"Распределение {col} по классам", col, "Плотность / число")
        # Для Rainfall: пик в нуле доминирует — ограничиваем X и лог по Y, чтобы видеть хвост
        if col == "Rainfall":
            ax.set_xlim(0, min(25, df[col].quantile(0.99) * 1.1))
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)
            ax.set_ylabel("Число (лог. шкала)")
        if ax.get_legend():
            ax.get_legend().set_title(COL_TARGET)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_distributions_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: figures/task2_distributions_by_class.png")


def print_descriptive_stats(df: pd.DataFrame, features: list) -> None:
    """Медиана, среднее, СКО для выбранных признаков: вся выборка и по классам."""
    print("\n--- Описательная статистика выбранных признаков ---")
    for col in features:
        stats_all = df[col].agg(["mean", "median", "std"])
        print(f"\n{col} — вся выборка: mean={stats_all['mean']:.4f}, median={stats_all['median']:.4f}, std={stats_all['std']:.4f}")
        by_class = df.groupby(COL_TARGET)[col].agg(["mean", "median", "std"])
        print(by_class.to_string())


def plot_above_below_median(df: pd.DataFrame, features: list) -> None:
    """Разбивка выше/ниже медианы: гистограмма, boxplot; оценка СКО."""
    for col in features:
        median_val = df[col].median()
        above = df[df[col] > median_val]
        below = df[df[col] <= median_val]
        std_above, std_below = above[col].std(), below[col].std()
        print(f"\n{col}: медиана = {median_val:.4f}. СКО выше медианы: {std_above:.4f}, ниже: {std_below:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        sns.histplot(above[col], label="Выше медианы", ax=axes[0], alpha=0.7, color="C1")
        sns.histplot(below[col], label="Ниже медианы", ax=axes[0], alpha=0.7, color="C0")
        _setup_plot(axes[0], f"Гистограмма {col}", col, "Число")
        axes[0].legend()

        sns.boxplot(data=df, x=COL_TARGET, y=col, ax=axes[1])
        _setup_plot(axes[1], f"Boxplot {col} по классам", COL_TARGET, col)

        plt.suptitle(f"Выше/ниже медианы {col}")
        plt.tight_layout()
        safe = col.replace(" ", "_").lower()
        plt.savefig(FIGURES_DIR / f"task2_above_below_median_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранено: figures/task2_above_below_median_{safe}.png")


def plot_class_balance(df: pd.DataFrame) -> None:
    """График сбалансированности классов (pie + barplot)."""
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


def run_task2(df: pd.DataFrame, features: list) -> None:
    """Задание 2: все графики и описательная статистика для выбранных признаков."""
    print("\n" + "=" * 60)
    print("Курсовая. Задание 2: визуализация и описательная статистика")
    print("=" * 60)
    print(f"  Анализируемые признаки: {features}")

    plot_distributions_by_class(df, features)
    print_descriptive_stats(df, features)
    plot_above_below_median(df, features)
    plot_class_balance(df)

    print("\nЗадание 2 завершено. Графики в каталоге figures/.")
