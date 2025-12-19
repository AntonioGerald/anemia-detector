from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torchvision.transforms as T

from torchvision import models


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "anemia_resnet18_binary.pt"
LABELS_PATH = MODELS_DIR / "anemia_resnet18_binary_labels.npy"  


def build_resnet18_binary(num_classes: int = 2) -> nn.Module:
    
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model_and_labels(device: torch.device) -> Tuple[nn.Module, list[str]]:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model tidak ditemukan: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise SystemExit(f"File label tidak ditemukan: {LABELS_PATH}")

    labels = np.load(LABELS_PATH, allow_pickle=True).tolist()

    model = build_resnet18_binary(num_classes=len(labels))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, labels


def preprocess_image(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise SystemExit(f"File gambar tidak ditemukan: {image_path}")

    img = Image.open(image_path)
    
    img = ImageOps.exif_transpose(img).convert("RGB")

    
    w, h = img.size
    left = 0
    right = max(int(0.75 * w), 1)
    top = int(0.2 * h)
    bottom = max(int(0.8 * h), top + 1)
    img = img.crop((left, top, right, bottom))

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transforms(img)
    return tensor.unsqueeze(0)


def predict_image(image_path_str: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Menggunakan device:", device)

    image_path = Path(image_path_str)

    model, labels = load_model_and_labels(device)
    input_tensor = preprocess_image(image_path).to(device)

    with torch.inference_mode():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]

    print(f"Gambar : {image_path}")
    print(f"Prediksi : {pred_label}")
    print("Probabilitas per kelas:")
    for label, p in zip(labels, probs):
        print(f"  {label:12s} : {p:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prediksi Anemia vs Tidak Anemia dari satu gambar kuku menggunakan ResNet18 biner."
    )
    parser.add_argument("image_path", type=str, help="Path ke file gambar kuku (JPEG/PNG)")

    args = parser.parse_args()

    predict_image(args.image_path)
