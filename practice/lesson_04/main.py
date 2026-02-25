"""
Занятие 4. Работа с табличными данными в библиотеке Pandas.
Запуск: python main.py из папки lesson_04. N — номер в списке группы (заменить в константе N_GROUP).
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CSV_PATHS = [
    os.path.join(SCRIPT_DIR, "diabetes_data_upload.csv"),
    os.path.join(DATA_DIR, "diabetes_data_upload.csv"),
]
CSV_PATH = next((p for p in CSV_PATHS if os.path.isfile(p)), CSV_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR

# Номер в списке группы — замените на свой
N_GROUP = 5


def get_csv():
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Файл не найден: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


# --- П.1: Загрузка CSV ---
# df = pd.read_csv('diabetes_data_upload.csv')


# --- Функция: описание DataFrame (индексы, типы, describe, первые 5 строк для первых N+2 столбцов) ---
def describe_dataframe(df: pd.DataFrame, n: int):
    """Выводит в консоль: индекс, dtypes, describe(), head(5) для первых N+2 столбцов."""
    k = min(n + 2, len(df.columns))
    cols = df.columns[:k].tolist()
    sub = df[cols]
    print("--- Индекс (метки строк) ---")
    print(sub.index)
    print("\n--- Типы данных (первые N+2 столбцов) ---")
    print(sub.dtypes)
    print("\n--- Описательная статистика (describe) ---")
    print(sub.describe(include="all"))
    print("\n--- Первые 5 строк (первые N+2 столбцов) ---")
    print(sub.head(5))


# --- П.3: Два DataFrame — строки со значением Yes и No в колонке N+1 (1-based: 6-я колонка = index 5) ---
def split_by_column_n1(df: pd.DataFrame, n: int):
    col_idx = n  # колонка N+1 в 1-based = index N
    if col_idx >= len(df.columns):
        col_idx = 5
    col_name = df.columns[col_idx]
    df_yes = df[df[col_name] == "Yes"].copy()
    df_no = df[df[col_name] == "No"].copy()
    print(f"\nКолонка N+1 (индекс {col_idx}): '{col_name}'")
    print(f"  Yes: {len(df_yes)} строк, No: {len(df_no)} строк")
    return df_yes, df_no, col_name


# --- Сортировка по колонкам N+1, N+2, Age; сохранение в новый DataFrame ---
def sort_and_save(df: pd.DataFrame, n: int):
    col1 = df.columns[n]
    col2 = df.columns[min(n + 1, len(df.columns) - 1)]
    by_cols = [col1, col2, "Age"]
    by_cols = [c for c in by_cols if c in df.columns]
    df_sorted = df.sort_values(by=by_cols).copy()
    path = os.path.join(OUTPUT_DIR, "diabetes_sorted.csv")
    df_sorted.to_csv(path, index=False)
    print(f"\nСортировка по {by_cols}. Результат сохранён: {path}")
    return df_sorted


# --- Пропуски: проверка и удаление строк с хотя бы одним пропуском ---
def drop_missing(df: pd.DataFrame):
    missing = pd.isna(df).any(axis=1).sum()
    print(f"\nСтрок с хотя бы одним пропуском: {missing}")
    if missing > 0:
        df_clean = df.dropna(how="any")
        print(f"После dropna(how='any'): {len(df_clean)} строк (удалено {len(df) - len(df_clean)})")
        return df_clean
    return df.copy()


# --- Гистограммы Age для двух таблиц (subplot) ---
def plot_hist_age_two(df_yes: pd.DataFrame, df_no: pd.DataFrame, col_label: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(df_yes["Age"], bins=20, edgecolor="black", alpha=0.7)
    ax1.set_title(f"Age — где {col_label} = Yes")
    ax1.set_xlabel("Age")
    ax2.hist(df_no["Age"], bins=20, edgecolor="black", alpha=0.7)
    ax2.set_title(f"Age — где {col_label} = No")
    ax2.set_xlabel("Age")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lesson04_hist_age_two.png"))
    plt.close()
    print("\nГистограммы Age сохранены: lesson04_hist_age_two.png")


# --- Boxplot Age для двух таблиц (два ящика рядом) ---
def plot_boxplot_age_two(df_yes: pd.DataFrame, df_no: pd.DataFrame, col_label: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    data = [df_yes["Age"].dropna(), df_no["Age"].dropna()]
    ax.boxplot(data, labels=[f"{col_label}=Yes", f"{col_label}=No"])
    ax.set_ylabel("Age")
    ax.set_title("Boxplot: Age по группам")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lesson04_boxplot_age_two.png"))
    plt.close()
    print("Boxplot Age сохранён: lesson04_boxplot_age_two.png")


# --- Scatter matrix для Age, колонки N+1, N+2 с цветом по class (Positive — красный, Negative — синий) ---
def plot_scatter_matrix_class(df: pd.DataFrame, n: int):
    cols = ["Age", df.columns[n], df.columns[min(n + 1, len(df.columns) - 1)]]
    cols = [c for c in cols if c in df.columns]
    if "class" not in df.columns:
        return
    # Кодируем класс для цвета: Positive=1 (красный), Negative=0 (синий)
    color = df["class"].map({"Positive": "red", "Negative": "blue"})
    # Числовое представление для осей (категории в код)
    plot_df = df[cols + ["class"]].copy()
    for c in cols:
        if plot_df[c].dtype == object or plot_df[c].dtype.name == "object":
            plot_df[c] = pd.Categorical(plot_df[c]).codes
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x_cols = [cols[0], cols[0], cols[1]]
    y_cols = [cols[1], cols[2], cols[2]]
    for ax, xc, yc in zip(axes, x_cols, y_cols):
        for label, clr in [("Positive", "red"), ("Negative", "blue")]:
            mask = df["class"] == label
            ax.scatter(plot_df.loc[mask, xc], plot_df.loc[mask, yc], c=clr, label=label, alpha=0.6)
        ax.set_xlabel(xc)
        ax.set_ylabel(yc)
        ax.legend()
    plt.suptitle("Попарные scatter: Age, N+1, N+2 (цвет = class)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lesson04_scatter_matrix_class.png"))
    plt.close()
    print("Scatter matrix (цвет по class) сохранена: lesson04_scatter_matrix_class.png")


if __name__ == "__main__":
    df = get_csv()
    n = N_GROUP
    describe_dataframe(df, n)
    df_yes, df_no, col_label = split_by_column_n1(df, n)
    df_sorted = sort_and_save(df, n)
    df_clean = drop_missing(df)
    plot_hist_age_two(df_yes, df_no, col_label)
    plot_boxplot_age_two(df_yes, df_no, col_label)
    plot_scatter_matrix_class(df, n)
