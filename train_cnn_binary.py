from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from step5_train_cnn_pytorch import SimpleCNN


DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "anemia_cnn_binary.pt"
LABELS_PATH = MODELS_DIR / "anemia_cnn_binary_labels.npy"  


class BinaryFingernailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: T.Compose | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:  
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["is_anemia"])  

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, label


def prepare_binary_dataframe() -> pd.DataFrame:
    if not DATASET_CSV.exists():
        raise SystemExit(f"CSV dataset tidak ditemukan: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)

    
    df = df[df["Tingkat_Anemia"] != "Tidak Diketahui"].copy()

    
    df = df[~df["HB"].isna()].copy()

    
    valid_indices: list[int] = []
    skipped = 0
    for idx, row in df.iterrows():
        img_path = row["image_path"]
        try:
            with Image.open(img_path) as im:  
                im.verify()
            valid_indices.append(idx)
        except Exception:
            skipped += 1

    df = df.loc[valid_indices].reset_index(drop=True)

    
    anemia_labels = {"Anemia Berat", "Anemia Ringan", "Anemia Sedang"}
    df["is_anemia"] = df["Tingkat_Anemia"].apply(lambda x: 1 if x in anemia_labels else 0)

    print("Jumlah sampel setelah filter & gambar valid:", len(df))
    print("Distribusi label biner (0=Tidak Anemia, 1=Anemia):")
    print(df["is_anemia"].value_counts())

    return df


def train_binary_model(num_epochs: int = 10, batch_size: int = 32, lr: float = 1e-3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training di device:", device)

    df = prepare_binary_dataframe()

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = BinaryFingernailDataset(df, transforms=transforms)

    
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = 2
    model = SimpleCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

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

        train_loss = running_loss / total
        train_acc = correct / total

        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            np.save(LABELS_PATH, np.array(["Tidak Anemia", "Anemia"]))
            print(f"  -> Model terbaik diperbarui (val_acc={val_acc:.4f}), disimpan ke {MODEL_PATH}")

    print("Training selesai.")
    print(f"Akurasi validasi terbaik: {best_val_acc:.4f}")
    print(f"Model biner tersimpan di: {MODEL_PATH.resolve()}")
    print(f"Label biner tersimpan di: {LABELS_PATH.resolve()}")


if __name__ == "__main__":
    train_binary_model(num_epochs=10, batch_size=32, lr=1e-3)
