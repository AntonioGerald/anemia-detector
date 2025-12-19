from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision import models

DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "anemia_resnet18_4class.pt"
LABELS_PATH = MODELS_DIR / "anemia_resnet18_4class_labels.npy"
HISTORY_PATH = MODELS_DIR / "anemia_resnet18_4class_history.npz"


class FingernailDataset4Class(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: T.Compose | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:  
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["class_idx"])  

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        w, h = img.size
        left = 0
        right = max(int(0.75 * w), 1)
        top = int(0.2 * h)
        bottom = max(int(0.8 * h), top + 1)
        img = img.crop((left, top, right, bottom))

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, label


def prepare_dataframe_4class() -> tuple[pd.DataFrame, list[str]]:
    if not DATASET_CSV.exists():
        raise SystemExit(f"CSV dataset not found: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)

    df = df[df["Tingkat_Anemia"] != "Tidak Diketahui"].copy()
    df = df[~df["HB"].isna()].copy()

    valid_indices: list[int] = []
    for idx, row in df.iterrows():
        img_path = row["image_path"]
        try:
            with Image.open(img_path) as im:  
                im.verify()
            valid_indices.append(idx)
        except Exception:
            continue

    df = df.loc[valid_indices].reset_index(drop=True)

    labels = sorted(df["Tingkat_Anemia"].unique())
    label_to_idx = {name: i for i, name in enumerate(labels)}
    df["class_idx"] = df["Tingkat_Anemia"].map(label_to_idx)

    print("Number of samples after filtering:", len(df))
    print("Class distribution (4-class):")
    print(df["Tingkat_Anemia"].value_counts())

    return df, labels


def build_resnet18_4class(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_resnet_4class(num_epochs: int = 30, batch_size: int = 32, lr: float = 1e-4) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    df, class_names = prepare_dataframe_4class()

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = FingernailDataset4Class(df, transforms=transforms)

    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_resnet18_4class(num_classes=len(class_names)).to(device)

    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0

    history_epochs: list[int] = []
    history_train_loss: list[float] = []
    history_val_loss: list[float] = []
    history_train_acc: list[float] = []
    history_val_acc: list[float] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss_val = criterion(outputs, labels)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
                val_running_loss += loss_val.item() * images.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_running_loss / val_total if val_total > 0 else 0.0

        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            np.save(LABELS_PATH, np.array(class_names))
            print(f"  -> Best model updated (val_acc={val_acc:.4f}), saved to {MODEL_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        HISTORY_PATH,
        epochs=np.array(history_epochs, dtype=int),
        train_loss=np.array(history_train_loss, dtype=float),
        val_loss=np.array(history_val_loss, dtype=float),
        train_acc=np.array(history_train_acc, dtype=float),
        val_acc=np.array(history_val_acc, dtype=float),
    )

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"4-class ResNet18 model saved to: {MODEL_PATH.resolve()}")
    print(f"4-class labels saved to: {LABELS_PATH.resolve()}")
    print(f"4-class training history saved to: {HISTORY_PATH.resolve()}")


if __name__ == "__main__":
    train_resnet_4class(num_epochs=30, batch_size=32, lr=1e-4)
