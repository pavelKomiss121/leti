"""
Курсовая работа: статистический анализ данных «Rain in Australia».
Целевая: RainTomorrow.

Пайплайн:
  1) Задание 1 — загрузка, проверка, очистка данных.
  2) Задание 3 — корреляционный анализ (все числовые признаки).
  3) Отбор признаков — топ по |корреляции| с целевой (или из config).
  4) Задание 2 — визуализация и описательная статистика выбранных признаков.
  5) Задание 4 — регрессионный анализ выбранных признаков.
  6) Задание 6 — ANOVA для выбранных признаков.
  7) Задание 7 — классификаторы (дерево, лес, MLP, логистика) на выбранных признаках.

Запуск из корня leti: python -m course_work.main
"""
from course_work.preprocessing import (
    load_raw_data,
    check_missing_duplicates_anomalies,
    clean_data,
    print_class_balance,
)
from course_work.analysis.correlation import run_task3
from course_work.analysis.feature_selection import select_features
from course_work.analysis.visualization import run_task2
from course_work.analysis.regression import run_task4
from course_work.analysis.anova import run_task6
from course_work.classifiers import run_task7


def main():
    # --- Задание 1: первичная обработка ---
    print("=" * 60)
    print("Курсовая. Задание 1: первичная обработка данных")
    print("=" * 60)

    df = load_raw_data()
    print("Размер до обработки:", df.shape)
    print()

    missing_pct = check_missing_duplicates_anomalies(df)

    print("--- Применение стратегии обработки ---")
    df_clean = clean_data(df, missing_pct)

    print_class_balance(df_clean)
    print("\nИтоговый размер выборки:", df_clean.shape)
    print("\nОбзор числовых признаков (describe):")
    print(df_clean.describe().to_string())

    # --- Задание 3: корреляционный анализ (все числовые) ---
    run_task3(df_clean)

    # --- Отбор признаков на основе корреляции ---
    features = select_features(df_clean)

    # --- Задание 2: визуализация выбранных признаков ---
    run_task2(df_clean, features)

    # --- Задание 4: регрессия выбранных признаков ---
    run_task4(df_clean, features)

    # --- Задание 6: ANOVA для выбранных признаков ---
    run_task6(df_clean, features)

    # --- Задание 7: классификаторы на выбранных признаках ---
    run_task7(df_clean, features)

    return df_clean


if __name__ == "__main__":
    main()
