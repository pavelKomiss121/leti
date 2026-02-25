"""
Упражнение 7. Сверточные нейронные сети PyTorch для классификации изображений (MNIST).
Рекомендуется Google Colab. Локально: pip install torch torchvision.
N_GROUP — номер в списке группы (seed, max_epochs).
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Номер в списке группы
N_GROUP = 5
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA недоступна. Обучение на CPU ...")
else:
    print("CUDA доступна! Обучение на GPU ...")


# --- Классы из методички: Identical, Flatten ---
class Identical(nn.Module):
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


# --- П.3: Загрузка MNIST (torchvision вместо catalyst) ---
def get_mnist_loaders(root_dir=None):
    root_dir = root_dir or os.getcwd()
    train_data = datasets.MNIST(root=root_dir, train=True, download=True, transform=transforms.ToTensor())
    val_data = datasets.MNIST(root=root_dir, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    return train_loader, val_loader


# --- П.4: Простейшая сеть (Flatten + Linear; активацию можно заменить на nn.ReLU()) ---
def make_baseline_model(activation=None):
    activation = activation or Identical
    return nn.Sequential(
        Flatten(),
        nn.Linear(28 * 28, 128),
        activation(),
        nn.Linear(128, 128),
        activation(),
        nn.Linear(128, 10),
    )


# --- П.5–6: Цикл обучения, точность на первой и последней итерации ---
def train_and_evaluate(model, train_loader, val_loader, max_epochs, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loaders = {"train": train_loader, "valid": val_loader}
    accuracy = {"train": [], "valid": []}
    first_train_acc, first_valid_acc = None, None
    last_train_acc, last_valid_acc = None, None

    for epoch in range(max_epochs):
        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                # MNIST torchvision: (B, 1, 28, 28); при необходимости добавляем канал
                inp = x_batch.float()
                if inp.dim() == 3:
                    inp = inp.unsqueeze(1)  # (B, 28, 28) -> (B, 1, 28, 28)
                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(inp)
                    loss = criterion(outp, y_batch)
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(inp)
                preds = outp.argmax(-1)
                correct = (preds == y_batch).sum().item()
                epoch_correct += correct
                epoch_all += len(y_batch)
            acc = epoch_correct / epoch_all
            accuracy[k].append(acc)
            if k == "train":
                print(f"Epoch: {epoch + 1}")
            print(f"  Loader: {k}, Accuracy: {acc:.4f}")
            if first_train_acc is None and k == "train":
                first_train_acc = acc
            if first_valid_acc is None and k == "valid":
                first_valid_acc = acc
            last_train_acc = acc if k == "train" else last_train_acc
            last_valid_acc = acc if k == "valid" else last_valid_acc

    return accuracy, (first_train_acc, first_valid_acc), (last_train_acc, last_valid_acc)


def run_baseline():
    """Базовая модель с Identical (без нелинейной активации)."""
    torch.manual_seed(N_GROUP)
    train_loader, val_loader = get_mnist_loaders()
    model = make_baseline_model(Identical)
    max_epochs = max(2, N_GROUP)
    print("--- Базовая модель (Identical activation), max_epochs =", max_epochs, "---")
    acc_hist, first, last = train_and_evaluate(model, train_loader, val_loader, max_epochs)
    print("\nТочность на первой итерации (train, valid):", first)
    print("Точность на последней итерации (train, valid):", last)
    return acc_hist


def run_with_relu():
    """Улучшение: ReLU и больше эпох (п.7)."""
    torch.manual_seed(N_GROUP)
    train_loader, val_loader = get_mnist_loaders()
    model = make_baseline_model(nn.ReLU)
    max_epochs = max(5, N_GROUP)
    print("--- Модель с ReLU, max_epochs =", max_epochs, "---")
    acc_hist, first, last = train_and_evaluate(model, train_loader, val_loader, max_epochs, lr=1e-3)
    print("\nТочность (последняя) train:", last[0], "valid:", last[1])
    return acc_hist


# --- П.8: Transfer learning — ResNet18, заморозка первых 3 слоёв, адаптация под MNIST ---
def get_mnist_224_loaders(root_dir=None):
    """MNIST с resize до 224x224 и 3 канала (для ResNet)."""
    root_dir = root_dir or os.getcwd()
    tr = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    train_data = datasets.MNIST(root=root_dir, train=True, download=True, transform=tr)
    val_data = datasets.MNIST(root=root_dir, train=False, download=True, transform=tr)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    return train_loader, val_loader


def make_resnet18_mnist(pretrained=True, freeze_first=3):
    import torchvision.models as models
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 4:  # freeze first 3 children (conv1, bn1, relu)
            for param in child.parameters():
                param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def run_transfer_learning():
    """Transfer learning: ResNet18, заморозка первых 3 слоёв (п.8)."""
    torch.manual_seed(N_GROUP)
    train_loader, val_loader = get_mnist_224_loaders()
    model = make_resnet18_mnist(pretrained=True, freeze_first=3)
    max_epochs = max(3, N_GROUP)
    print("--- Transfer learning (ResNet18), max_epochs =", max_epochs, "---")
    acc_hist, first, last = train_and_evaluate(model, train_loader, val_loader, max_epochs, lr=1e-3)
    print("\nТочность (последняя) valid:", last[1])
    return acc_hist


if __name__ == "__main__":
    run_baseline()
    print("\n" + "=" * 50 + "\n")
    run_with_relu()
    print("\n" + "=" * 50 + "\n")
    try:
        run_transfer_learning()
    except Exception as e:
        print("Transfer learning (ResNet18) пропущен:", e)
        print("Убедитесь, что установлены torch и torchvision с поддержкой pretrained моделей.")
