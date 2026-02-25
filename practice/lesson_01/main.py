"""
Занятие 1. Базовые операции и синтаксис Python.
Запуск: из папки lesson_01 выполнить python main.py.
Файл данных: diabetes_data_upload.csv — положить в practice/data/ или в lesson_01/.
"""
import csv
import os

# Путь к CSV: рядом со скриптом или в practice/data/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CSV_PATHS = [
    os.path.join(SCRIPT_DIR, "diabetes_data_upload.csv"),
    os.path.join(DATA_DIR, "diabetes_data_upload.csv"),
]
CSV_PATH = next((p for p in CSV_PATHS if os.path.isfile(p)), CSV_PATHS[0])


# --- П.4: Открытие CSV в Python (вложенный список) ---
def task4_read_csv():
    with open(CSV_PATH, newline="", encoding="utf-8") as csvfile:
        data = list(csv.reader(csvfile))
    print("П.4: первые 3 строки загруженных данных:")
    for row in data[:3]:
        print(row)
    return data


# --- П.5: Среднее по колонке Age с помощью цикла for ---
def task5_mean_age(data):
    header = data[0]
    col_age = header.index("Age")  # индекс колонки Age
    sum_num = 0
    count = 0
    for i in range(1, len(data)):
        val = data[i][col_age]
        if val.isdigit():
            sum_num += int(val)
            count += 1
    avg = sum_num / count if count else 0
    print(f"\nП.5: Среднее по Age (цикл for): {avg:.2f}, число записей: {count}")
    return avg


# --- П.6: Среднее по Age отдельно для мужчин и женщин (if-else) ---
def task6_mean_by_gender(data):
    header = data[0]
    col_age = header.index("Age")
    col_gender = header.index("Gender")
    sum_m, cnt_m = 0, 0
    sum_f, cnt_f = 0, 0
    for i in range(1, len(data)):
        age_val = data[i][col_age]
        if not age_val.isdigit():
            continue
        age = int(age_val)
        g = data[i][col_gender].strip()
        if g == "Male":
            sum_m += age
            cnt_m += 1
        elif g == "Female":
            sum_f += age
            cnt_f += 1
    avg_m = sum_m / cnt_m if cnt_m else 0
    avg_f = sum_f / cnt_f if cnt_f else 0
    print(f"\nП.6: Среднее Age мужчины: {avg_m:.2f} (n={cnt_m}), женщины: {avg_f:.2f} (n={cnt_f})")
    return avg_m, avg_f


# --- П.7: Таблица связь диабет × ожирение (четыре массива номеров пациентов) ---
def task7_diabetes_obesity_table(data):
    header = data[0]
    col_class = header.index("class")    # Positive / Negative — диабет (в файле UCI колонка "class")
    col_obesity = header.index("Obesity") # Yes / No — ожирение
    # Индексы строк (1..len-1 — это «номера пациентов» в смысле строки таблицы)
    diabetes_pos_obesity_pos = []
    diabetes_pos_obesity_neg = []
    diabetes_neg_obesity_pos = []
    diabetes_neg_obesity_neg = []
    for i in range(1, len(data)):
        d = data[i][col_class].strip()
        o = data[i][col_obesity].strip()
        # Приводим к единому виду: Positive/1 и Negative/0 для диабета; для ожирения Yes=1, No=0
        is_d = d == "Positive"
        is_o = o == "Yes"
        if is_d and is_o:
            diabetes_pos_obesity_pos.append(i)
        elif is_d and not is_o:
            diabetes_pos_obesity_neg.append(i)
        elif not is_d and is_o:
            diabetes_neg_obesity_pos.append(i)
        else:
            diabetes_neg_obesity_neg.append(i)
    # Вывод таблицы в консоль
    print("\nП.7: Таблица (связь диабет × ожирение) — количество пациентов:")
    print("                    Diabetes Positive (1)   Diabetes Negative (0)")
    print("Obesity Positive (1)      {:5d}                    {:5d}".format(
        len(diabetes_pos_obesity_pos), len(diabetes_neg_obesity_pos)))
    print("Obesity Negative (0)      {:5d}                    {:5d}".format(
        len(diabetes_pos_obesity_neg), len(diabetes_neg_obesity_neg)))
    print("\nЧетыре массива с номерами строк пациентов (примеры первых 5):")
    print("  Диабет+, Ожирение+:", diabetes_pos_obesity_pos[:5])
    print("  Диабет+, Ожирение-:", diabetes_pos_obesity_neg[:5])
    print("  Диабет-, Ожирение+:", diabetes_neg_obesity_pos[:5])
    print("  Диабет-, Ожирение-:", diabetes_neg_obesity_neg[:5])
    return diabetes_pos_obesity_pos, diabetes_pos_obesity_neg, diabetes_neg_obesity_pos, diabetes_neg_obesity_neg


if __name__ == "__main__":
    data = task4_read_csv()
    task5_mean_age(data)
    task6_mean_by_gender(data)
    task7_diabetes_obesity_table(data)
