"""
Очистка данных: дроп столбцов/строк, импутация, исправление аномалий, winsorize.
"""
import numpy as np
import pandas as pd

from course_work.config import COL_TARGET
from course_work.preprocessing.utils import winsorize_iqr, winsorize_rainfall


def clean_data(df: pd.DataFrame, missing_pct: pd.Series) -> pd.DataFrame:
    """
    Стратегия обработки: дата → datetime; дроп столбцов >40% пропусков;
    дроп строк без RainTomorrow; импутация; исправление аномалий (WindSpeed >= 0);
    выбросы: Rainfall — winsorize по 99-му перцентилю (без удаления строк),
    Humidity3pm — winsorize по 1.5*IQR (без удаления).
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    threshold = 40.0
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        print("Удалены столбцы (>40% пропусков):", drop_cols)

    df = df.dropna(subset=[COL_TARGET])
    print(f"После удаления строк с пропуском {COL_TARGET}:", df.shape[0], "строк")

    before_anom = len(df)
    if "Rainfall" in df.columns:
        df = df[(df["Rainfall"] >= 0) & (df["Rainfall"] <= 300)].copy()
    if "Humidity3pm" in df.columns:
        df = df[
            df["Humidity3pm"].isna()
            | ((df["Humidity3pm"] >= 0) & (df["Humidity3pm"] <= 100))
        ].copy()
    if len(df) < before_anom:
        print("Удалены строки с экстремальными аномалиями:", before_anom - len(df), "строк")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

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

    for col in ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = 0

    print("Пропуски после импутации:", df.isnull().sum().sum())

    if "Rainfall" in df.columns:
        df["Rainfall"] = winsorize_rainfall(df["Rainfall"], upper_percentile=99.0)
        print("Rainfall: winsorize по 99-му перцентилю (строки не удаляются)")

    if "Humidity3pm" in df.columns:
        df["Humidity3pm"] = winsorize_iqr(df["Humidity3pm"], factor=1.5)
        print("Humidity3pm: winsorize по 1.5*IQR (строки не удаляются)")

    return df


def print_class_balance(df: pd.DataFrame, target: str = COL_TARGET) -> None:
    """Вывод баланса классов после очистки."""
    if target not in df.columns:
        return
    print("\n--- Баланс классов (после очистки) ---")
    counts = df[target].value_counts(normalize=True)
    print(counts.to_string())
    print("(доля Yes ≈ 22–23% в оригинале; сильное отклонение — пересмотреть стратегию выбросов)")
