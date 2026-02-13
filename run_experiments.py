# run_experiments.py
import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
from multiprocessing import freeze_support

from dataset import FallDataset
from transforms import train_transform, eval_transform
from models import (
    SimpleCNN,
    ComplexCNN,
    resnet18_model,
    efficientnet_b0_model,
    densenet121_model
)
from train_utils import (
    train_epoch,
    evaluate,
    save_confusion_matrix,
    save_roc_curve,
    device
)

# -------------------------
# Configuration
# -------------------------
ROOT = "archive"
RESULTS_DIR = "results"
MODEL_SAVE_DIR = "models"

BATCH_SIZE = 16          # safer on Windows
EPOCHS = 30              # as discussed
NUM_WORKERS = 0          # REQUIRED for stability
PIN_MEMORY = False

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Clean training sets only
TRAIN_SETS = {
    "500": f"{ROOT}/train-500",
    "1000": f"{ROOT}/train-1000",
}

MODELS = {
    "SimpleCNN": lambda: SimpleCNN(),
    "ComplexCNN": lambda: ComplexCNN(),
    "ResNet18": lambda: resnet18_model(),
    "EfficientNetB0": lambda: efficientnet_b0_model(),
    "DenseNet121": lambda: densenet121_model(),
}

# -------------------------
# Storage for plots
# -------------------------
results = {}

# -------------------------
# Main (Windows safe)
# -------------------------
if __name__ == "__main__":
    freeze_support()

    for model_name, model_fn in MODELS.items():
        print(f"\n==============================")
        print(f"MODEL: {model_name}")
        print(f"==============================")

        results[model_name] = {
            "train_sizes": [],
            "val_acc": [],
            "val_prec": [],
            "val_rec": [],
            "test_acc": [],
            "test_prec": [],
            "test_rec": [],
        }

        for size, path in TRAIN_SETS.items():
            print(f"\nTraining with {size} images")

            # -------------------------
            # Datasets
            # -------------------------
            train_ds = FallDataset(
                f"{path}/images",
                f"{path}/labels",
                train_transform
            )
            val_ds = FallDataset(
                f"{ROOT}/valid/images",
                f"{ROOT}/valid/labels",
                eval_transform
            )
            test_ds = FallDataset(
                f"{ROOT}/test/images",
                f"{ROOT}/test/labels",
                eval_transform
            )

            print("Train class distribution:",
                  Counter([l for _, l in train_ds.samples]))

            # -------------------------
            # DataLoaders
            # -------------------------
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )

            # -------------------------
            # Model setup
            # -------------------------
            model = model_fn().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3
            )

            # -------------------------
            # Training
            # -------------------------
            for epoch in range(EPOCHS):
                acc = train_epoch(model, train_loader, criterion, optimizer)
                print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {acc:.3f}")

            # -------------------------
            # Evaluation
            # -------------------------
            val_acc, val_prec, val_rec = evaluate(model, val_loader)

            test_acc, test_prec, test_rec, y_true, y_pred, y_probs = evaluate(
                model,
                test_loader,
                return_preds=True,
                return_probs=True
            )

            print(f"[VALID] Acc:{val_acc:.3f} Prec:{val_prec:.3f} Rec:{val_rec:.3f}")
            print(f"[TEST ] Acc:{test_acc:.3f} Prec:{test_prec:.3f} Rec:{test_rec:.3f}")

            # -------------------------
            # Confusion Matrix
            # -------------------------
            save_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                train_size=size,
                results_dir=RESULTS_DIR
            )

            save_roc_curve(
                y_true=y_true,
                y_probs=y_probs,
                model_name=model_name,
                train_size=size,
                results_dir=RESULTS_DIR
            )

            # -------------------------
            # Save trained model
            # -------------------------
            model_path = f"{MODEL_SAVE_DIR}/{model_name}_{size}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

            # -------------------------
            # Store results for plots
            # -------------------------
            results[model_name]["train_sizes"].append(int(size))
            results[model_name]["val_acc"].append(val_acc)
            results[model_name]["val_prec"].append(val_prec)
            results[model_name]["val_rec"].append(val_rec)
            results[model_name]["test_acc"].append(test_acc)
            results[model_name]["test_prec"].append(test_prec)
            results[model_name]["test_rec"].append(test_rec)

    # -------------------------
    # Plot metrics
    # -------------------------
    print("\nGenerating metric plots...")

    for model_name, data in results.items():
        sizes = data["train_sizes"]

        # Accuracy
        plt.figure()
        plt.plot(sizes, data["test_acc"], marker="o", label="Test Accuracy")
        plt.plot(sizes, data["val_acc"], marker="o", linestyle="--", label="Val Accuracy")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} - Accuracy vs Training Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTS_DIR}/{model_name}_accuracy.png")
        plt.close()

        # Precision
        plt.figure()
        plt.plot(sizes, data["test_prec"], marker="o", label="Test Precision")
        plt.plot(sizes, data["val_prec"], marker="o", linestyle="--", label="Val Precision")
        plt.xlabel("Training Set Size")
        plt.ylabel("Precision")
        plt.title(f"{model_name} - Precision vs Training Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTS_DIR}/{model_name}_precision.png")
        plt.close()

        # Recall
        plt.figure()
        plt.plot(sizes, data["test_rec"], marker="o", label="Test Recall")
        plt.plot(sizes, data["val_rec"], marker="o", linestyle="--", label="Val Recall")
        plt.xlabel("Training Set Size")
        plt.ylabel("Recall")
        plt.title(f"{model_name} - Recall vs Training Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTS_DIR}/{model_name}_recall.png")
        plt.close()

    print("✅ Training complete.")
    print("✅ Models saved in 'models_saved/' folder.")
    print("✅ Confusion matrices and plots saved in 'results/' folder.")
