import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import models, datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import random

# Set seed and device
torch.manual_seed(43)
np.random.seed(43)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- Utilities ----------------------------
def log_device():
    print(f"Using device: {DEVICE}")

def get_transforms(train=True):
    base = [transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip()] + base)
    return transforms.Compose(base)

def compute_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return (correct / total) * 100

def forward_pass(model, criterion, data_loader, is_train=False, optimizer=None):
    phase = 'train' if is_train else 'val/test'
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds, all_labels = [], []

    data_iter = iter(data_loader)
    idx = 0
    while idx < len(data_loader):
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            if is_train:
                loss.backward()
                optimizer.step()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        idx += 1

    avg_loss = total_loss / len(data_loader)
    acc = (total_correct / total_samples) * 100

    return avg_loss, acc, all_preds, all_labels

def plot_accuracy(train_hist, val_hist):
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label="Train Acc", marker="o", color="blue")
    plt.plot(val_hist, label="Val Acc", marker="s", color="red")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_conf_matrix(y_true, y_pred, classes):
    matrix = confusion_matrix(y_true, y_pred)
    norm = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ---------------------------- Data Loading ----------------------------
def prepare_dataloaders(data_dir="/kaggle/input/nature-922/inaturalist_12K"):
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=get_transforms(True))
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=get_transforms(False))

    labels = train_data.classes
    train_subset, val_subset = random_split(train_data, [8000, 1999])

    def make_loader(ds): return DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

    return labels, make_loader(train_subset), make_loader(val_subset), make_loader(test_data)

# ---------------------------- Training Loop ----------------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, epochs, class_names):
    train_accs, val_accs = [], []

    for ep in range(epochs):
        tr_loss, tr_acc, _, _ = forward_pass(model, criterion, train_loader, is_train=True, optimizer=optimizer)
        val_loss, val_acc, _, _ = forward_pass(model, criterion, val_loader)

        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {ep+1}/{epochs} - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    plot_accuracy(train_accs, val_accs)

    # Final test performance
    tst_loss, tst_acc, y_pred, y_true = forward_pass(model, criterion, test_loader)
    print(f"Test Loss: {tst_loss:.4f}, Test Accuracy: {tst_acc:.2f}%")
    plot_conf_matrix(y_true, y_pred, class_names)
    
    return model

# ---------------------------- Layer Freezing ----------------------------
def freeze_layers(model, k, strategy):
    children = list(model.named_children())

    match strategy:
        case "first":
            freeze_range = range(k)
        case "middle":
            total = len(children)
            freeze_range = range(total // 3, (2 * total) // 3)
        case "last":
            freeze_range = range(len(children) - k, len(children))
        case _:
            raise ValueError("Strategy must be 'first', 'middle', or 'last'.")

    for idx, (name, layer) in enumerate(children):
        if idx in freeze_range:
            for param in layer.parameters():
                param.requires_grad = False
    print(f"Strategy '{strategy}' applied: frozen layers {list(freeze_range)}")

# ---------------------------- Fine-tune Driver ----------------------------
def fine_tune_model(epoch, k, strategy, optim_type):
    log_device()
    class_labels, train_loader, val_loader, test_loader = prepare_dataloaders()
    
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(DEVICE)

    freeze_layers(model, k, strategy)

    optimizer = {
        "adam": lambda: optim.Adam(model.parameters(), lr=0.001),
        "sgd": lambda: optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    }.get(optim_type, lambda: (_ for _ in ()).throw(ValueError("Invalid optimizer")))()

    criterion = nn.CrossEntropyLoss()

    start = time.time()
    model = train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, epoch, class_labels)
    print(f"Total training time: {time.time() - start:.2f}s")
    
    return model

