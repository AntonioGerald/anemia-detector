from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision import models
import matplotlib.pyplot as plt

DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "anemia_resnet18_4class.pt"
LABELS_PATH = MODELS_DIR / "anemia_resnet18_4class_labels.npy"
HISTORY_PATH = MODELS_DIR / "anemia_resnet18_4class_history.npz"


class EvalDataset4Class(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: T.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:  
        return len(self.df)

    def __getitem__(self, idx: int):  
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        label = int(row["class_idx"])

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        w, h = img.size
        left = 0
        right = max(int(0.75 * w), 1)
        top = int(0.2 * h)
        bottom = max(int(0.8 * h), top + 1)
        img = img.crop((left, top, right, bottom))

        img = self.transforms(img)
        return img, label


def load_dataframe_4class() -> tuple[pd.DataFrame, list[str]]:
    if not DATASET_CSV.exists():
        raise SystemExit(f"CSV dataset not found: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)
    df = df[df["Tingkat_Anemia"] != "Tidak Diketahui"].copy()
    df = df[~df["HB"].isna()].copy()

    labels = sorted(df["Tingkat_Anemia"].unique())
    label_to_idx = {name: i for i, name in enumerate(labels)}
    df["class_idx"] = df["Tingkat_Anemia"].map(label_to_idx)

    return df, labels


def build_resnet18_4class(num_classes: int):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(device: torch.device):
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise SystemExit("4-class model or label file not found. Train the model first.")

    class_names = np.load(LABELS_PATH, allow_pickle=True).tolist()
    model = build_resnet18_4class(num_classes=len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation device:", device)

    df, class_names = load_dataframe_4class()
    print("Total samples for evaluation:", len(df))

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = EvalDataset4Class(df, transforms=transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model, class_names = load_model(device)

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(list(probs))
            all_preds.extend(list(preds))
            all_targets.extend(targets.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs)

    print("\n=== Confusion Matrix (4-class) ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n=== Classification Report (4-class) ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax_cm)

    tick_marks = np.arange(len(class_names))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax_cm.set_ylabel("True Class")
    ax_cm.set_xlabel("Predicted Class")
    ax_cm.set_title("4-Class Confusion Matrix")
    fig_cm.tight_layout()
    cm_path = reports_dir / "confusion_matrix_4class.png"
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)

    if HISTORY_PATH.exists():
        history = np.load(HISTORY_PATH)
        epochs = history["epochs"]
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        train_acc = history["train_acc"]
        val_acc = history["val_acc"]

        fig_hist, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

        ax_loss.plot(epochs, train_loss, label="Training Loss")
        ax_loss.plot(epochs, val_loss, label="Validation Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("4-Class Training and Validation Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, train_acc, label="Training Accuracy")
        ax_acc.plot(epochs, val_acc, label="Validation Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("4-Class Training and Validation Accuracy")
        ax_acc.legend()

        fig_hist.tight_layout()
        hist_path = reports_dir / "training_curves_4class.png"
        fig_hist.savefig(hist_path, dpi=300)
        plt.close(fig_hist)

    class_counts = np.bincount(y_true, minlength=len(class_names))
    fig_bar, ax_bar = plt.subplots(figsize=(4, 4))
    ax_bar.bar(
        np.arange(len(class_names)),
        class_counts,
        tick_label=class_names,
        color="#4C72B0",
    )
    ax_bar.set_ylabel("Number of Samples")
    ax_bar.set_title("4-Class Label Distribution on Evaluation Set")
    fig_bar.tight_layout()
    bar_path = reports_dir / "label_distribution_4class.png"
    fig_bar.savefig(bar_path, dpi=300)
    plt.close(fig_bar)

    max_probs = y_probs.max(axis=1)
    correct_mask = y_pred == y_true
    probs_correct = max_probs[correct_mask]
    probs_incorrect = max_probs[~correct_mask]

    fig_prob, ax_prob = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0.0, 1.0, 21)
    ax_prob.hist(
        probs_correct,
        bins=bins,
        alpha=0.6,
        label="Correct Predictions",
        color="#4C72B0",
        density=True,
    )
    ax_prob.hist(
        probs_incorrect,
        bins=bins,
        alpha=0.6,
        label="Incorrect Predictions",
        color="#DD8452",
        density=True,
    )
    ax_prob.set_xlabel("Predicted Probability (Chosen Class)")
    ax_prob.set_ylabel("Density")
    ax_prob.set_title("4-Class Prediction Confidence Distribution")
    ax_prob.legend()
    fig_prob.tight_layout()
    prob_path = reports_dir / "probability_distribution_4class.png"
    fig_prob.savefig(prob_path, dpi=300)
    plt.close(fig_prob)


if __name__ == "__main__":
    main()
