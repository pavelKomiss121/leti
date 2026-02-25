"""
Занятие 3. NumPy arrays and functions.
Запуск: python main.py из папки lesson_03.
Для заданий с CSV нужен diabetes_data_upload.csv в practice/data/ или lesson_03/.
"""
import os

import numpy as np
import pandas as pd

# Размер массива: длина = номер в списке группы, ширина = номер первой буквы фамилии (в алфавите).
# Замените на свои значения (например, 5 и 3 → массив 5×3).
NUM_ROW = 5   # номер в списке группы
NUM_COL = 3   # номер первой буквы фамилии (А=1, Б=2, ..., Я=33 и т.д.)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CSV_PATHS = [
    os.path.join(SCRIPT_DIR, "diabetes_data_upload.csv"),
    os.path.join(DATA_DIR, "diabetes_data_upload.csv"),
]
CSV_PATH = next((p for p in CSV_PATHS if os.path.isfile(p)), CSV_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR


# --- П.1: Двумерный массив NumPy заданного размера ---
def task1_create_array():
    a = np.zeros((NUM_ROW, NUM_COL))
    print("П.1: Двумерный массив shape = (номер в списке, буква фамилии):")
    print(a)
    print("shape:", a.shape)
    return a


# --- П.2: Массивы того же размера с равномерным и нормальным распределением ---
def task2_random_arrays():
    uniform = np.random.rand(NUM_ROW, NUM_COL)
    normal = np.random.randn(NUM_ROW, NUM_COL)
    print("\nП.2: Равномерное [0, 1) — np.random.rand:")
    print(uniform)
    print("Нормальное — np.random.randn:")
    print(normal)
    return uniform, normal


# --- П.3: Изучение массива (shape, ndim, dtype, itemsize, size) ---
def task3_array_info_direct(a):
    print("\nП.3: Атрибуты массива (на примере случайного):")
    print("  a.shape:", a.shape)
    print("  a.ndim:", a.ndim)
    print("  a.dtype.name:", a.dtype.name)
    print("  a.itemsize:", a.itemsize)
    print("  a.size:", a.size)


# --- Анализ данных, п.1: Функция, выводящая информацию о массиве ---
def print_array_info(a):
    """Принимает NumPy-массив, выводит в консоль shape, ndim, dtype.name, itemsize, size."""
    a = np.asarray(a)
    print("  shape:", a.shape)
    print("  ndim:", a.ndim)
    print("  dtype.name:", a.dtype.name)
    print("  itemsize:", a.itemsize)
    print("  size:", a.size)


# --- Анализ данных, п.2: Типы колонок таблицы средствами NumPy, запись в CSV ---
def task_analysis2_column_types_to_csv():
    if not os.path.isfile(CSV_PATH):
        print("\nФайл diabetes_data_upload.csv не найден. Пропуск заданий с таблицей.")
        return
    table = pd.read_csv(CSV_PATH)
    rows = []
    for col in table.columns:
        arr = table[col].to_numpy()
        dtype_name = np.asarray(arr).dtype.name
        rows.append([col, dtype_name])
    out_path = os.path.join(OUTPUT_DIR, "column_types.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("column,dtype\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]}\n")
    print("\nАнализ п.2: Типы колонок записаны в", out_path)
    for r in rows:
        print(" ", r[0], "->", r[1])


# --- Анализ данных, п.3: Перекодирование категорий в числа, сохранение CSV, повтор п.2 ---
def task_analysis3_encode_and_save():
    if not os.path.isfile(CSV_PATH):
        return
    table = pd.read_csv(CSV_PATH)
    encoded = table.copy()
    for col in encoded.columns:
        if encoded[col].dtype == object or encoded[col].dtype.name == "object":
            uniq = encoded[col].astype(str).unique()
            mapping = {v: i for i, v in enumerate(sorted(uniq))}
            encoded[col] = encoded[col].astype(str).map(mapping)
    out_path = os.path.join(OUTPUT_DIR, "diabetes_encoded.csv")
    encoded.to_csv(out_path, index=False)
    print("\nАнализ п.3: Таблица без буквенных обозначений сохранена в", out_path)

    # Повтор п.2 для новой таблицы: типы колонок в CSV
    rows = []
    for col in encoded.columns:
        arr = encoded[col].to_numpy()
        dtype_name = np.asarray(arr).dtype.name
        rows.append([col, dtype_name])
    out_types2 = os.path.join(OUTPUT_DIR, "column_types_after_encode.csv")
    with open(out_types2, "w", encoding="utf-8") as f:
        f.write("column,dtype\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]}\n")
    print("Типы колонок после кодирования записаны в", out_types2)


if __name__ == "__main__":
    a = task1_create_array()
    uniform, normal = task2_random_arrays()
    task3_array_info_direct(normal)
    print("\n--- Функция print_array_info(a) ---")
    print_array_info(normal)
    task_analysis2_column_types_to_csv()
    task_analysis3_encode_and_save()
