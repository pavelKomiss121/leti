"""
Загрузка и первичная обработка данных (Задание 1).
Один файл — одна задача: только чтение, проверка качества и очистка данных.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from course_work.utils import winsorize_iqr, winsorize_rainfall

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILE_CSV = DATA_DIR / "weatherAUS.csv"


def load_raw_data() -> pd.DataFrame:
    """Загрузка сырых данных из CSV (в файле пропуски обозначены как NA)."""
    return pd.read_csv(FILE_CSV, na_values=["NA"], keep_default_na=True)


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

    print("--- Аномалии ---")
    anom_rain = (df["Rainfall"] < 0) | (df["Rainfall"] > 300)
    anom_hum = df["Humidity3pm"].notna() & (
        (df["Humidity3pm"] < 0) | (df["Humidity3pm"] > 100)
    )
    print("Строк с аномалией Rainfall (отриц. или >300 мм):", anom_rain.sum())
    print("Строк с Humidity3pm вне [0, 100]:", anom_hum.sum())

    # Отрицательные скорости ветра — аномалия (MinTemp < 0 допустимо)
    for col in ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                print(f"Строк с {col} < 0:", neg)
    print()

    return missing_pct


def clean_data(df: pd.DataFrame, missing_pct: pd.Series) -> pd.DataFrame:
    """
    Стратегия обработки: дата → datetime; дроп столбцов >40% пропусков;
    дроп строк без RainTomorrow; импутация; исправление аномалий (WindSpeed >= 0);
    выбросы: Rainfall — winsorize по 99-му перцентилю (без удаления строк),
    Humidity3pm — winsorize по 1.5*IQR (методичка: «обработка выбросов методом IQR» без удаления).
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    threshold = 40.0
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        print("Удалены столбцы (>40% пропусков):", drop_cols)

    df = df.dropna(subset=["RainTomorrow"])
    print("После удаления строк с пропуском RainTomorrow:", df.shape[0], "строк")

    # Удалить только экстремальные аномалии (без массового IQR-удаления)
    before_anom = len(df)
    df = df[(df["Rainfall"] >= 0) & (df["Rainfall"] <= 300)].copy()
    df = df[
        df["Humidity3pm"].isna()
        | ((df["Humidity3pm"] >= 0) & (df["Humidity3pm"] <= 100))
    ].copy()
    if len(df) < before_anom:
        print("Удалены строки с экстремальными аномалиями (Rainfall <0 или >300 мм, Humidity3pm вне [0,100]):", before_anom - len(df), "строк")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Мода для категориальных
    cat_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    for c in cat_cols:
        if c in df.columns and df[c].isnull().any():
            mode_val = df[c].mode()
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
    if "RainToday" in df.columns and df["RainToday"].isnull().any():
        mode_val = df["RainToday"].mode()
        df["RainToday"] = df["RainToday"].fillna(
            mode_val.iloc[0] if not mode_val.empty else "No"
        )

    # Отрицательные скорости ветра — аномалия, заменить на 0
    for col in ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = 0

    print("Пропуски после импутации:", df.isnull().sum().sum())

    # Выбросы: без удаления строк (сохраняем распределение целевой и признака m)
    # Rainfall: дождливые дни — не выбросы, а реальные события; ограничиваем верхний хвост 99-м перцентилем
    df["Rainfall"] = winsorize_rainfall(df["Rainfall"], upper_percentile=99.0)
    print("Rainfall: winsorize по 99-му перцентилю (строки не удаляются)")

    # Humidity3pm: winsorize по 1.5*IQR (замена на границы вместо удаления)
    df["Humidity3pm"] = winsorize_iqr(df["Humidity3pm"], factor=1.5)
    print("Humidity3pm: winsorize по 1.5*IQR (строки не удаляются)")

    return df


def print_class_balance(df: pd.DataFrame, target: str = "RainTomorrow") -> None:
    """Вывод баланса классов после очистки (для Задания 1 и связи с Заданием 2)."""
    if target not in df.columns:
        return
    print("\n--- Баланс классов (после очистки) ---")
    counts = df[target].value_counts(normalize=True)
    print(counts.to_string())
    print("(доля Yes ≈ 22–23% в оригинале; сильное отклонение — пересмотреть стратегию выбросов)")
