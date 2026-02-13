# inference_select_model.py
import os
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

# -------------------------
# Configuration
# -------------------------
MODEL_DIR = "models_saved"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Available models mapping
# -------------------------
MODEL_FACTORY = {
    "1": ("SimpleCNN", SimpleCNN),
    "2": ("ComplexCNN", ComplexCNN),
    "3": ("ResNet18", resnet18_model),
    "4": ("EfficientNetB0", efficientnet_b0_model),
    "5": ("DenseNet121", densenet121_model),
}

# -------------------------
# List saved models
# -------------------------
def list_saved_models():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    return sorted(files)

# -------------------------
# Load selected model
# -------------------------
def load_model(model_name, model_class, model_path):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------
# Predict single image
# -------------------------
def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = eval_transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(1).item()
        confidence = probs.max().item()

    label = "Fall" if pred == 1 else "No Fall"
    return label, confidence

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("\nAvailable model architectures:")
    for key, (name, _) in MODEL_FACTORY.items():
        print(f"{key}. {name}")

    model_choice = input("\nSelect model architecture (1–5): ").strip()

    if model_choice not in MODEL_FACTORY:
        print("❌ Invalid model selection.")
        exit()

    model_name, model_class = MODEL_FACTORY[model_choice]

    print(f"\nSelected architecture: {model_name}")

    saved_models = list_saved_models()
    matching_models = [f for f in saved_models if f.startswith(model_name)]

    if not matching_models:
        print(f"❌ No saved weights found for {model_name}")
        exit()

    print("\nAvailable trained models:")
    for idx, fname in enumerate(matching_models, 1):
        print(f"{idx}. {fname}")

    weight_choice = input("\nSelect trained model: ").strip()

    try:
        weight_choice = int(weight_choice) - 1
        model_file = matching_models[weight_choice]
    except:
        print("❌ Invalid selection.")
        exit()

    model_path = os.path.join(MODEL_DIR, model_file)
    model = load_model(model_name, model_class, model_path)

    print(f"\nLoaded model: {model_file}")

    image_path = input("\nEnter path to test image: ").strip()    #test\image_name.extension

    if not os.path.exists(image_path):
        print("❌ Image not found.")
        exit()

    label, confidence = predict_image(model, image_path)

    print("\n==============================")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print("==============================")
