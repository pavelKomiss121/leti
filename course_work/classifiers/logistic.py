"""
Логистическая регрессия.
"""
from sklearn.linear_model import LogisticRegression

from course_work.config import RANDOM_STATE


def build_logistic():
    """Создать LogisticRegression с сбалансированными весами классов."""
    return LogisticRegression(
        max_iter=500, random_state=RANDOM_STATE, class_weight="balanced"
    )
