"""
Занятие 2. Загрузка и настройка модулей NumPy, Pandas, scikit-learn, TensorFlow, Keras.
Запуск: python main.py из папки lesson_02. Файл diabetes_data_upload.csv — в practice/data/ или lesson_02/.
"""
import os
import time

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

# Keras в новых версиях идёт из tensorflow
try:
    import keras
except ImportError:
    keras = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CSV_PATHS = [
    os.path.join(SCRIPT_DIR, "diabetes_data_upload.csv"),
    os.path.join(DATA_DIR, "diabetes_data_upload.csv"),
]
CSV_PATH = next((p for p in CSV_PATHS if os.path.isfile(p)), CSV_PATHS[0])


# --- П.1–2: Импорт библиотек (уже в начале файла) ---
# import numpy as np
# import pandas as pd
# import sklearn
# import tensorflow as tf
# import keras


# --- П.3: Вывод версий каждой библиотеки из скрипта ---
def task3_print_versions():
    print("Версии библиотек (из кода):")
    print("  numpy:", np.__version__)
    print("  pandas:", pd.__version__)
    print("  sklearn:", sklearn.__version__)
    print("  tensorflow:", tf.__version__)
    if keras is not None:
        print("  keras:", keras.__version__)
    else:
        print("  keras: (не установлен или встроен в tf)")


# --- П.4: Открытие CSV через Pandas ---
def task4_read_csv_pandas():
    table = pd.read_csv(CSV_PATH)
    print("\nП.4: Загрузка через pd.read_csv. Форма таблицы:", table.shape)
    print("Первые 2 строки:\n", table.head(2))
    return table


# --- П.5: Среднее по Age для всех и отдельно для мужчин/женщин (NumPy/Pandas), сравнение с занятием 1 ---
def task5_mean_pandas(table):
    age_all = table["Age"].mean()
    male = table[table["Gender"] == "Male"]
    female = table[table["Gender"] == "Female"]
    age_male = male["Age"].mean()
    age_female = female["Age"].mean()
    print("\nП.5: Среднее по колонке Age (Pandas):")
    print("  Все:", age_all)
    print("  Мужчины:", age_male)
    print("  Женщины:", age_female)
    print("(Сравните с расчётами из занятия 1 — цикл for по вложенному списку; результаты должны совпадать.)")
    return age_all, age_male, age_female


# --- П.6: Время выполнения расчёта из п.5 (Pandas) и аналогичного расчёта из занятия 1 (цикл по списку) ---
def task6_time_compare(table):
    import csv as csv_module
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        data = list(csv_module.reader(f))
    header = data[0]
    col_age = header.index("Age")
    col_gender = header.index("Gender")

    # Время: расчёт среднего циклом (как в занятии 1)
    start = time.time()
    sum_m, cnt_m = 0, 0
    sum_f, cnt_f = 0, 0
    for i in range(1, len(data)):
        if not data[i][col_age].isdigit():
            continue
        age = int(data[i][col_age])
        g = data[i][col_gender].strip()
        if g == "Male":
            sum_m += age
            cnt_m += 1
        elif g == "Female":
            sum_f += age
            cnt_f += 1
    _ = (sum_m / cnt_m if cnt_m else 0), (sum_f / cnt_f if cnt_f else 0)
    end = time.time()
    time_loop = end - start

    # Время: расчёт через Pandas (как в п.5)
    start = time.time()
    _ = table["Age"].mean()
    _ = table[table["Gender"] == "Male"]["Age"].mean()
    _ = table[table["Gender"] == "Female"]["Age"].mean()
    end = time.time()
    time_pandas = end - start

    print("\nП.6: Время выполнения:")
    print("  Цикл for (как в занятии 1):", f"{time_loop:.6f} с")
    print("  Pandas (п.5):", f"{time_pandas:.6f} с")
    print("(Pandas обычно быстрее за счёт векторных операций.)")


if __name__ == "__main__":
    task3_print_versions()
    table = task4_read_csv_pandas()
    task5_mean_pandas(table)
    task6_time_compare(table)
