"""
Загрузка данных и проверка качества (пропуски, дубликаты, аномалии).
Аномалии проверяются по всем числовым столбцам с естественным диапазоном (до отбора признаков).
"""
import pandas as pd

from course_work.config import FILE_CSV


def load_raw_data() -> pd.DataFrame:
    """Загрузка сырых данных из CSV (в файле пропуски обозначены как NA)."""
    return pd.read_csv(FILE_CSV, na_values=["NA"], keep_default_na=True)


# Проверки аномалий по столбцам: (имя, минимально допустимое, максимально допустимое), None = не проверять верх/низ
_ANOMALY_CHECKS = [
    ("Rainfall", 0, 300),
    ("Humidity3pm", 0, 100),
    ("Humidity9am", 0, 100),
    ("WindGustSpeed", 0, None),
    ("WindSpeed9am", 0, None),
    ("WindSpeed3pm", 0, None),
    ("Cloud9am", 0, 9),
    ("Cloud3pm", 0, 9),
]


def check_missing_duplicates_anomalies(df: pd.DataFrame) -> pd.Series:
    """Проверка пропусков, дубликатов и аномалий. Возвращает series с % пропусков по столбцам."""
    print("--- Проверка пропусков ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({"пропусков": missing, "%": missing_pct})
    with_missing = missing_df[missing_df["пропусков"] > 0].sort_values("%", ascending=False)
    print(with_missing.to_string())
    print()

    print("--- Дубликаты ---")
    n_dup = df.duplicated().sum()
    print("Число полных дубликатов строк:", n_dup)
    print()

    print("--- Аномалии (по числовым столбцам с естественным диапазоном, до отбора признаков) ---")
    for col, lo, hi in _ANOMALY_CHECKS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        n_bad = 0
        if lo is not None:
            n_bad += (s < lo).sum()
        if hi is not None:
            n_bad += (s > hi).sum()
        if n_bad > 0:
            range_str = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
            print(f"  {col} вне {range_str}: {n_bad} строк")
    print()

    return missing_pct
