"""
Лабораторная работа 4. Нейронные сети.
Простейшая нейронная сеть на Python без специализированных библиотек (только numpy).
Запуск из корня leti: python -m lab4.main
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lab4.model import NeuralNetwork
from lab4.utils import accuracy, mse_loss, split_70_30

# Папка для сохранения графиков
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

try:
    import lab1.DataGenerator as dg
except ImportError:
    import sys

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    import lab1.DataGenerator as dg


# ---------- Параметры данных (как в методичке, можно менять) ----------
N = 1000
mu0 = [0, 2, 3]
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]
col = len(mu0)
mu = [mu0, mu1]
sigma = [sigma0, sigma1]

# Генерация выборки и приведение Y к форме (n_samples, 1)
X, Y_flat, class0, class1 = dg.norm_dataset(mu, sigma, N)
n_samples = X.shape[0]
Y = np.reshape(Y_flat.astype(np.float64), (n_samples, 1))

# Параметры сети и обучения
N_NEURONS = 4
N_EPOCH = 50
LEARNING_RATE = 0.5
SEED = 42

# Инициализация сети
NN = NeuralNetwork(X, Y, n_neuro=N_NEURONS, learning_rate=LEARNING_RATE, seed=SEED)

# История потерь и точности по эпохам
history_loss = []
history_acc = []

print("Обучение нейронной сети...")
for i in range(N_EPOCH):
    pred = NN.feedforward()
    loss = mse_loss(Y, pred)
    acc = accuracy(Y, pred)
    history_loss.append(loss)
    history_acc.append(acc)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Эпоха {i + 1:3d}  Loss: {loss:.6f}  Точность: {acc*100:.2f}%")
    NN.train_step(X, Y)

# Итоговая точность
pred_final = NN.feedforward()
final_accuracy = accuracy(Y, pred_final)
final_loss = mse_loss(Y, pred_final)
print(f"\nИтоговая точность на обучающей выборке: {final_accuracy*100:.2f}%")
print(f"Итоговая MSE: {final_loss:.6f}")

# Вынесение весов в отдельные переменные (п.5 самостоятельной)
weights_layer1 = NN.weights1.copy()
weights_layer2 = NN.weights2.copy()
print(f"\nВеса первого слоя (вход -> скрытый), shape {weights_layer1.shape}:")
print(weights_layer1)
print(f"\nВеса второго слоя (скрытый -> выход), shape {weights_layer2.shape}:")
print(weights_layer2)

# ---------- Графики зависимости потерь и точности от номера эпохи ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, N_EPOCH + 1), history_loss, "b-")
ax1.set_xlabel("Номер эпохи")
ax1.set_ylabel("Среднеквадратичная ошибка (MSE)")
ax1.set_title("Функция потерь на обучающей выборке")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, N_EPOCH + 1), [a * 100 for a in history_acc], "g-")
ax2.set_xlabel("Номер эпохи")
ax2.set_ylabel("Точность, %")
ax2.set_title("Точность на обучающей выборке")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "lab4_loss_accuracy.png")
plt.close()
print(f"\nГрафики сохранены в папке: {FIGURES_DIR}")

# ---------- Доп. задание: оценка на тестовой выборке (метод test) ----------
print("\n--- Оценка на тестовой выборке (train/test 70/30) ---")
Xtrain, Ytrain, Xtest, Ytest = split_70_30(X, Y)
NN_split = NeuralNetwork(Xtrain, Ytrain, n_neuro=N_NEURONS, learning_rate=LEARNING_RATE, seed=SEED)
for _ in range(N_EPOCH):
    NN_split.train_step(Xtrain, Ytrain)
pred_test = NN_split.test(Xtest)
acc_test = accuracy(Ytest, pred_test)
acc_train_split = accuracy(Ytrain, NN_split.feedforward())
print(f"  Точность на обучающей выборке: {acc_train_split*100:.2f}%")
print(f"  Точность на тестовой выборке (метод test): {acc_test*100:.2f}%")

# ---------- Подбор числа нейронов и эпох (оценка по тестовой выборке) ----------
print("\n--- Подбор числа нейронов и эпох (обучение на 70%, оценка на 30% теста) ---")
best_test_acc = -1
best_nn, best_ep = None, None
for n_neur in [2, 4, 8, 16]:
    for n_ep in [30, 50, 100]:
        nn = NeuralNetwork(Xtrain, Ytrain, n_neuro=n_neur, learning_rate=LEARNING_RATE, seed=SEED)
        for _ in range(n_ep):
            nn.train_step(Xtrain, Ytrain)
        acc_train = accuracy(Ytrain, nn.feedforward())
        acc_test_val = accuracy(Ytest, nn.test(Xtest))
        print(f"  нейронов={n_neur:2d}, эпох={n_ep:3d} -> train {acc_train*100:.2f}%, test {acc_test_val*100:.2f}%")
        if acc_test_val > best_test_acc:
            best_test_acc = acc_test_val
            best_nn, best_ep = n_neur, n_ep
print(f"  Оптимально по тестовой точности: нейронов={best_nn}, эпох={best_ep}, test={best_test_acc*100:.2f}%")
