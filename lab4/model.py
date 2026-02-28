"""
Модель нейронной сети: активационные функции и класс NeuralNetwork.
"""
import numpy as np


def sigmoid(Z):
    """Сигмоида: выход от 0 до 1. Устойчиво к переполнению при больших |Z|."""
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(p):
    """Производная сигмоиды в точке p: sigma'(p) = p * (1 - p)."""
    return p * (1 - p)


class NeuralNetwork:
    """
    Однослойная нейронная сеть: входной слой -> скрытый слой (n_neuro нейронов) -> выход (1 нейрон).
    Обучение: прямое распространение (feedforward) + обратное распространение (backprop).
    """

    def __init__(self, x, y, n_neuro=4, learning_rate=0.5, seed=None):
        """
        Инициализация весов случайными значениями.
        x: (n_samples, n_features), y: (n_samples, 1).
        """
        if seed is not None:
            np.random.seed(seed)
        self.input = x
        n_inp = self.input.shape[1]
        self.n_neuro = n_neuro
        self.lr = learning_rate
        self.weights1 = np.random.rand(n_inp, n_neuro)
        self.weights2 = np.random.rand(n_neuro, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self, X=None):
        """
        Прямое распространение: выходы слоёв по сигмоиде.
        Если X задан — используется он (для предсказания на тестовых данных), иначе self.input.
        """
        inp = X if X is not None else self.input
        self.layer1 = sigmoid(np.dot(inp, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        """
        Обратное распространение: коррекция весов по градиенту MSE.
        .T — транспонирование для согласования размерностей при градиенте по весам.
        """
        d_weights2 = np.dot(
            self.layer1.T,
            2 * (self.y - self.output) * sigmoid_derivative(self.output),
        )
        d_weights1 = np.dot(
            self.input.T,
            np.dot(
                2 * (self.y - self.output) * sigmoid_derivative(self.output),
                self.weights2.T,
            )
            * sigmoid_derivative(self.layer1),
        )
        self.weights1 += self.lr * d_weights1
        self.weights2 += self.lr * d_weights2

    def train_step(self, X, y):
        """Один шаг обучения: прямой проход и обновление весов."""
        self.input = X
        self.y = y
        self.output = self.feedforward()
        self.backprop()

    def test(self, X):
        """
        Предсказание на новых данных X без изменения весов.
        Возвращает вероятности (0..1); для меток класса: (pred > 0.5).astype(int).
        """
        return self.feedforward(X)
