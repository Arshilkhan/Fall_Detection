import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    y_true, y_pred = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

    return accuracy_score(y_true, y_pred)

def evaluate(model, loader, return_preds=False, return_probs=False):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            probs = torch.softmax(out, dim=1)[:, 1]

            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    if return_preds and return_probs:
        return acc, prec, rec, np.array(y_true), np.array(y_pred), np.array(y_probs)
    elif return_preds:
        return acc, prec, rec, np.array(y_true), np.array(y_pred)
    else:
        return acc, prec, rec

def save_confusion_matrix(y_true, y_pred, model_name, train_size, results_dir):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Fall", "Fall"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} (Train {train_size})")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_cm_{train_size}.png")
    plt.close()

def save_roc_curve(y_true, y_probs, model_name, train_size, results_dir):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve (Train {train_size})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/{model_name}_roc_{train_size}.png")
    plt.close()