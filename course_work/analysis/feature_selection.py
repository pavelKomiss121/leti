"""
Отбор признаков по корреляции с целевой переменной.
Вызывается ПОСЛЕ корреляционного анализа (Задание 3), ДО визуализации, регрессии, ANOVA, классификации.
"""
import numpy as np
import pandas as pd

from course_work.config import COL_TARGET, CLASSIFIER_FEATURES, CLASSIFIER_FEATURES_TOP_N


def select_features(df: pd.DataFrame) -> list:
    """
    Выбрать 2–3 самых важных признака для предсказания RainTomorrow.
    Если в config задан CLASSIFIER_FEATURES — берутся они.
    Иначе — топ по модулю корреляции Пирсона с целевой переменной.
    Выводит обоснование в консоль. Возвращает список имён признаков.
    """
    print("\n" + "=" * 60)
    print("Отбор признаков для анализа и классификации")
    print("=" * 60)

    if CLASSIFIER_FEATURES:
        available = [c for c in CLASSIFIER_FEATURES if c in df.columns]
        if not available:
            raise ValueError(f"Ни один из признаков {CLASSIFIER_FEATURES} не найден в данных.")
        print(f"  Признаки заданы вручную в config: {available}")
        return available

    numeric = df.select_dtypes(include=[np.number])
    if COL_TARGET in df.columns and COL_TARGET not in numeric.columns:
        y = (df[COL_TARGET] == "Yes").astype(int)
    else:
        y = numeric[COL_TARGET] if COL_TARGET in numeric.columns else (df[COL_TARGET] == "Yes").astype(int)
    cand = numeric.drop(columns=[COL_TARGET], errors="ignore")
    for c in ["Date", "Unnamed: 0"]:
        if c in cand.columns:
            cand = cand.drop(columns=[c])
    if cand.empty:
        raise ValueError("Нет числовых признаков для отбора.")
    cand = cand.fillna(cand.median())
    corr = cand.corrwith(y).abs().sort_values(ascending=False)

    print(f"\n  Модуль корреляции Пирсона каждого числового признака с {COL_TARGET}:")
    for feat, r in corr.items():
        print(f"    {feat:20s}  |r| = {r:.3f}")

    top = corr.head(CLASSIFIER_FEATURES_TOP_N)
    features = top.index.tolist()
    parts = [f"{c} (|r|={top[c]:.3f})" for c in features]
    print(f"\n  Выбраны топ-{CLASSIFIER_FEATURES_TOP_N} признака: {', '.join(parts)}.")
    print(f"  Эти признаки будут использоваться в заданиях 2, 4, 6, 7.")
    return features
