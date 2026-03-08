"""
Общие константы курсовой работы (Rain in Australia).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FILE_CSV = DATA_DIR / "weatherAUS.csv"

# Целевая переменная
COL_TARGET = "RainTomorrow"

# Отбор признаков для анализа и классификации.
# None = выбрать автоматически по |корреляции| с целью (топ CLASSIFIER_FEATURES_TOP_N).
# Иначе список вручную, напр. ["Humidity3pm", "Rainfall"] или ["Humidity3pm", "Rainfall", "Pressure3pm"].
CLASSIFIER_FEATURES = None
CLASSIFIER_FEATURES_TOP_N = 3

# Разбиение и воспроизводимость
TEST_SIZE = 0.3  # 70% train / 30% test
CV_FOLDS = 5
RANDOM_STATE = 42
