"""
Упражнение 5. Машинное обучение с библиотекой scikit-learn.
Классификатор: наличие диабета по признакам. Предобработка, train/test 80/20, логистическая регрессия, метрики, подбор C.
Запуск: python main.py из папки lesson_05. N_GROUP — номер в списке группы.
"""
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CSV_PATHS = [
    os.path.join(SCRIPT_DIR, "diabetes_data_upload.csv"),
    os.path.join(DATA_DIR, "diabetes_data_upload.csv"),
]
CSV_PATH = next((p for p in CSV_PATHS if os.path.isfile(p)), CSV_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR

N_GROUP = 5


def get_df():
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Файл не найден: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


# --- 1. Загрузка; 2. Разделение на X (все признаки кроме class) и Y (class) ---
def prepare_xy(df: pd.DataFrame):
    target_col = "class"
    if target_col not in df.columns:
        target_col = df.columns[-1]
    Y = df[target_col].values
    feature_cols = [c for c in df.columns if c != target_col]
    X_df = df[feature_cols]
    return X_df, Y, feature_cols


# --- 3. Стандартизация числовых переменных (Age и др.); 4. OneHotEncoder для категориальных ---
def preprocess(X_df: pd.DataFrame):
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    parts = []
    if numeric_cols:
        X_num = X_df[numeric_cols].values.astype(float)
        X_num = preprocessing.scale(X_num)
        parts.append(X_num)
    if cat_cols:
        enc = preprocessing.OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(X_df[cat_cols])
        parts.append(X_cat)
    if not parts:
        return np.array([]), numeric_cols, cat_cols
    X = np.hstack(parts)
    return X, numeric_cols, cat_cols


# --- Кодирование Y в 0/1 для метрик ---
def encode_y(Y):
    uniq = np.unique(Y)
    mapping = {v: i for i, v in enumerate(uniq)}
    return np.array([mapping[y] for y in Y])


def main():
    df = get_df()
    X_df, Y_raw, feature_cols = prepare_xy(df)
    X, num_cols, cat_cols = preprocess(X_df)
    if X.size == 0:
        print("Нет числовых/категориальных столбцов для X.")
        return
    Y = encode_y(Y_raw)

    # Сохранение X, Y в CSV (после предобработки)
    np.savetxt(os.path.join(OUTPUT_DIR, "X.csv"), X, delimiter=",")
    np.savetxt(os.path.join(OUTPUT_DIR, "Y.csv"), Y, delimiter=",", fmt="%d")
    print("Сохранены X.csv, Y.csv")

    # 80% / 20%, random_state = номер в списке группы
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=N_GROUP
    )
    print("Размерности: X_train", X_train.shape, "Y_train", Y_train.shape, "X_test", X_test.shape, "Y_test", Y_test.shape)

    # Обучение логистической регрессии
    model = LogisticRegression(random_state=0).fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Метрики: accuracy и одна дополнительная (F1, подходит для бинарной классификации)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average="binary")
    try:
        Y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test, Y_proba)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
    except Exception:
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # Оценка: достаточность метрики
    print("Одна метрика (только accuracy) может быть недостаточной при дисбалансе классов или разной стоимости ошибок; F1 и AUC дополняют картину.")

    # Улучшение: подбор C из заданного интервала
    param_range = [100, 10, 1, 0.1, 0.01, 0.001]
    best_c, best_acc = None, -1
    for c in param_range:
        m = LogisticRegression(C=c, random_state=0).fit(X_train, Y_train)
        a = accuracy_score(Y_test, m.predict(X_test))
        if a > best_acc:
            best_acc = a
            best_c = c
    print(f"Наилучшая точность при C из [100,10,1,0.1,0.01,0.001]: C={best_c}, accuracy={best_acc:.4f}")


if __name__ == "__main__":
    main()
