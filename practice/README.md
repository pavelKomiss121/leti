# Практические занятия

В этой папке собраны практические задания по курсу: базовый Python, работа с CSV, установка библиотек, NumPy/Pandas, табличные данные, scikit-learn, PyTorch. Код — только там, где требуется программа; остальное — краткие сводки для понимания (могут спросить на защите лабораторных).

- **lesson_01** — Занятие 1: базовые операции, синтаксис Python, чтение CSV, циклы и условия.
- **lesson_02** — Занятие 2: установка и импорт numpy, pandas, sklearn, tensorflow, keras; версии; замер времени.
- **lesson_03** — Занятие 3: массивы NumPy, случайные массивы, атрибуты ndarray, типы колонок, кодирование категорий.
- **lesson_04** — Занятие 4: Pandas — структуры, исследование DataFrame, фильтрация, сортировка, пропуски, гистограммы, boxplot, scatter matrix.
- **lesson_05** — Упражнение 5: scikit-learn — предобработка (стандартизация, one-hot), train/test 80/20, логистическая регрессия, метрики, подбор C.
- **lesson_06** — Упражнение 6: PyTorch — классификатор на make_moons (Dataset/DataLoader, BCEWithLogitsLoss, цикл обучения, оценка accuracy). Рекомендуется запуск в Google Colab.
- **lesson_07** — Упражнение 7: сверточные сети PyTorch для MNIST — базовая модель (Flatten + Linear), цикл обучения, ReLU и улучшения, transfer learning (ResNet18, заморозка первых слоёв). Рекомендуется Colab.

**Данные:** для заданий с диабетом нужен файл `diabetes_data_upload.csv`. Скачать: https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv и положить в `practice/data/` или в папку занятия (`lesson_01`–`lesson_05`).

**Зависимости:** в корне проекта выполнить `pip install -r requirements.txt`. Для занятий 1–5 нужны numpy, pandas, matplotlib, scikit-learn; для занятия 2 опционально — tensorflow, keras; для занятий 6–7 — torch, torchvision (или выполнять в Colab).
