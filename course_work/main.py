"""
Курсовая работа: статистический анализ данных «Rain in Australia».
Параметры: n = Humidity3pm, m = Rainfall. Целевая: RainTomorrow.

Точка входа — только запуск пайплайна (данные → проверка → очистка → вывод).
Вся логика обработки в data.py, отрисовка и анализ — в отдельных модулях.

Запуск из корня leti: python -m course_work.main
"""
from course_work.data import (
    load_raw_data,
    check_missing_duplicates_anomalies,
    clean_data,
    print_class_balance,
)
from course_work.viz import run_task2
from course_work.correlation import run_task3
from course_work.regression import run_task4
from course_work.anova import run_task6
from course_work.classifier import run_task7


def main():
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

    run_task2(df_clean)
    run_task3(df_clean)
    run_task4(df_clean)
    run_task6(df_clean)
    run_task7(df_clean)

    return df_clean


if __name__ == "__main__":
    main()
