from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as T

from step5_train_cnn_pytorch import SimpleCNN  


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "anemia_cnn_pytorch.pt"
LABELS_PATH = MODELS_DIR / "anemia_cnn_labels.npy"


def load_model_and_labels(device: torch.device) -> Tuple[nn.Module, list[str]]:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model tidak ditemukan: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise SystemExit(f"File label tidak ditemukan: {LABELS_PATH}")

    labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
    num_classes = len(labels)

    model = SimpleCNN(num_classes=num_classes)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, labels


def preprocess_image(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise SystemExit(f"File gambar tidak ditemukan: {image_path}")

    img = Image.open(image_path).convert("RGB")

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
    print(f"Prediksi Tingkat Anemia : {pred_label}")
    print("Probabilitas per kelas:")
    for label, p in zip(labels, probs):
        print(f"  {label:14s} : {p:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prediksi tingkat anemia dari satu gambar kuku menggunakan model CNN PyTorch.")
    parser.add_argument("image_path", type=str, help="Path ke file gambar kuku (JPEG/PNG)")

    args = parser.parse_args()

    predict_image(args.image_path)
