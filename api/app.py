from pathlib import Path
from typing import Tuple
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import torch
from torch import nn
import torchvision.transforms as T
from torchvision import models

MODELS_DIR = Path("models")
MODEL_PATH_BINARY = MODELS_DIR / "anemia_resnet18_binary.pt"
LABELS_PATH_BINARY = MODELS_DIR / "anemia_resnet18_binary_labels.npy"
MODEL_PATH_4CLASS = MODELS_DIR / "anemia_resnet18_4class.pt"
LABELS_PATH_4CLASS = MODELS_DIR / "anemia_resnet18_4class_labels.npy"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_resnet18_binary(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
def build_resnet18_4class(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_binary_model_and_labels(device: torch.device) -> Tuple[nn.Module, list[str]]:
    if not MODEL_PATH_BINARY.exists() or not LABELS_PATH_BINARY.exists():
        raise RuntimeError("Binary model or label file not found. Train the binary model first.")

    labels = np.load(LABELS_PATH_BINARY, allow_pickle=True).tolist()

    model = build_resnet18_binary(num_classes=len(labels))
    state_dict = torch.load(MODEL_PATH_BINARY, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, labels


def load_4class_model_and_labels(device: torch.device) -> Tuple[nn.Module, list[str]]:
    if not MODEL_PATH_4CLASS.exists() or not LABELS_PATH_4CLASS.exists():
        raise RuntimeError("4-class model or label file not found. Train the 4-class model first.")

    labels = np.load(LABELS_PATH_4CLASS, allow_pickle=True).tolist()

    model = build_resnet18_4class(num_classes=len(labels))
    state_dict = torch.load(MODEL_PATH_4CLASS, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, labels


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_BINARY, LABELS_BINARY = load_binary_model_and_labels(DEVICE)
MODEL_4CLASS, LABELS_4CLASS = load_4class_model_and_labels(DEVICE)


def preprocess_pil_image(img: Image.Image) -> torch.Tensor:
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


@app.post("/predict")
async def predict_binary(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG or PNG.")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    input_tensor = preprocess_pil_image(img).to(DEVICE)

    with torch.inference_mode():
        outputs = MODEL_BINARY(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = LABELS_BINARY[pred_idx]

    return {
        "mode": "binary",
        "label": pred_label,
        "probabilities": {label: float(p) for label, p in zip(LABELS_BINARY, probs)},
    }


@app.post("/predict-multiclass")
async def predict_multiclass(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG or PNG.")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    input_tensor = preprocess_pil_image(img).to(DEVICE)

    with torch.inference_mode():
        outputs = MODEL_4CLASS(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = LABELS_4CLASS[pred_idx]

    return {
        "mode": "multiclass",
        "label": pred_label,
        "probabilities": {label: float(p) for label, p in zip(LABELS_4CLASS, probs)},
    }
