"""
Вспомогательные функции для курсовой работы (Rain in Australia).
"""
import pandas as pd


def filter_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Маска значений в пределах [Q1 - factor*IQR, Q3 + factor*IQR] (для диагностики)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    return (series >= low) & (series <= high)


def winsorize_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Winsorize: заменить значения за пределами factor*IQR на граничные (строки не удаляются)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower=low, upper=high)


def winsorize_rainfall(series: pd.Series, upper_percentile: float = 99.0) -> pd.Series:
    """
    Ограничить Rainfall сверху перцентилем (нижняя граница 0).
    Сохраняет дождливые дни, убирает только экстремальные ливни.
    """
    return series.clip(lower=0, upper=series.quantile(upper_percentile / 100))
