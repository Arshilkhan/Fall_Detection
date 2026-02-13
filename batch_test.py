import os
import csv
import torch
from PIL import Image

from models import (
    SimpleCNN,
    ComplexCNN,
    resnet18_model,
    efficientnet_b0_model,
    densenet121_model
)
from transforms import eval_transform

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_saved")
IMAGE_DIR = os.path.join(BASE_DIR, "test")   # unlabeled images
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "all_models_predictions.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "SimpleCNN": SimpleCNN,
    "ComplexCNN": ComplexCNN,
    "ResNet18": resnet18_model,
    "EfficientNetB0": efficientnet_b0_model,
    "DenseNet121": densenet121_model,
}

def load_model(model_class, weight_path):
    model = model_class()
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

if __name__ == "__main__":

    print("\nRunning batch inference for ALL models...")
    print("Saving results to ONE CSV file\n")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image",
            "Model",
            "Train_Size",
            "Prediction",
            "Confidence"
        ])

        for model_name, model_class in MODELS.items():

            weight_files = sorted(
                f for f in os.listdir(MODEL_DIR)
                if f.startswith(model_name) and f.endswith(".pth")
            )

            if not weight_files:
                print(f"⚠ No saved models found for {model_name}")
                continue

            for weight_file in weight_files:
                weight_path = os.path.join(MODEL_DIR, weight_file)
                train_size = weight_file.replace(".pth", "").split("_")[-1]

                print(f"\nModel: {model_name} | Train size: {train_size}")

                model = load_model(model_class, weight_path)

                for img_name in sorted(os.listdir(IMAGE_DIR)):
                    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue

                    img_path = os.path.join(IMAGE_DIR, img_name)

                    image = Image.open(img_path).convert("RGB")
                    image_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(image_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred = probs.argmax(1).item()
                        conf = probs.max().item()

                    label = "Fall" if pred == 1 else "No Fall"

                    writer.writerow([
                        img_name,
                        model_name,
                        train_size,
                        label,
                        f"{conf:.4f}"
                    ])

                    print(
                        f"{img_name:25s} → {label} "
                        f"({conf*100:.2f}%)"
                    )

    print("\n==============================")
    print("Batch inference completed")
    print("==============================")
    print(f"Results saved to:\n{RESULTS_FILE}")
