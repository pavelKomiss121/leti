"""
Упражнение 6. Классификатор на PyTorch (логистическая регрессия на make_moons).
Рекомендуется запуск в Google Colab. Локально: pip install torch matplotlib scikit-learn.
N_GROUP — номер в списке группы (влияет на random_state, lr, max_epochs).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_GROUP = 5
BATCH_SIZE = 128
TOL = 1e-5  # порог сходимости по изменению весов


def check_cuda():
    if not TORCH_AVAILABLE:
        print("PyTorch не установлен. Установите: pip install torch")
        return False
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print("CUDA недоступна. Обучение на CPU ...")
    else:
        print("CUDA доступна! Обучение на GPU ...")
    return True


# --- 3. Генерация данных make_moons и отображение ---
def get_data(n_group: int):
    N = max(n_group, 1)
    X, y = make_moons(n_samples=5000, random_state=1, noise=0.1)
    plt.figure(figsize=(8, 5))
    plt.title("Dataset make_moons")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
    plt.savefig("lesson06_moons.png")
    plt.close()
    print("Датасет сохранён: lesson06_moons.png")
    return X, y


# --- 4. Разделение на train/val ---
def split_data(X, y, n_group: int):
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=n_group)
    return X_train, X_val, y_train, y_val


# --- 6. Модель: линейный слой (логистическая регрессия) ---
class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weights)
        self.bias = bias
        if bias:
            self.bias_term = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = x @ self.weights
        if self.bias:
            x = x + self.bias_term
        return x


# --- 10. Предсказание на выборке ---
def predict(dataloader, model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            outp = model(x_batch)
            probs = torch.sigmoid(outp)
            preds = (probs > 0.5).long()
            predictions.append(preds.numpy().flatten())
    return np.concatenate(predictions)


def main():
    if not check_cuda():
        return
    N = N_GROUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = get_data(N)
    X_train, X_val, y_train, y_val = split_data(X, y, N)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    linear_regression = LinearRegression(2, 1).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(linear_regression.parameters(), lr=0.01 * N)
    max_epochs = 5 * N
    losses = []
    prev_weights = torch.zeros_like(linear_regression.weights)
    stop_it = False

    for epoch in range(max_epochs):
        for it, (X_batch, y_batch) in enumerate(train_dataloader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outp = linear_regression.forward(X_batch).squeeze(1)
            loss = loss_function(outp, y_batch.squeeze(1))
            loss.backward()
            losses.append(loss.detach().item())
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(linear_regression.forward(X_batch))
                preds = (probs > 0.5).long().squeeze(1)
                batch_acc = (preds == y_batch.squeeze(1).long()).float().mean().item()
            if it % 5 == 0:
                print(f"Iteration: {it + epoch * len(train_dataloader)}, Batch accuracy: {batch_acc:.4f}")

            current_weights = linear_regression.weights.detach().clone()
            if (prev_weights - current_weights).abs().max() < TOL:
                print(f"Convergence. Stopping at iteration {it + epoch * len(train_dataloader)}.")
                stop_it = True
                break
            prev_weights = current_weights
        if stop_it:
            break

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.savefig("lesson06_losses.png")
    plt.close()
    print("График потерь сохранён: lesson06_losses.png")

    # Точность на тестовой (валидационной) выборке
    linear_regression.to("cpu")
    acc = accuracy_score(predict(val_dataloader, linear_regression), y_val)
    print(f"Accuracy на тестовой выборке: {acc:.4f}")

    print("Улучшение: попробуйте изменить lr, max_epochs или заменить модель на нейросеть с активациями (ReLU, несколько слоёв).")


if __name__ == "__main__":
    main()
