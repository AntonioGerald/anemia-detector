from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from torchvision import models

DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "anemia_resnet18_binary.pt"
LABELS_PATH = MODELS_DIR / "anemia_resnet18_binary_labels.npy"
HISTORY_PATH = MODELS_DIR / "anemia_resnet18_binary_history.npz"


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: T.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:  
        return len(self.df)

    def __getitem__(self, idx: int):  
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        label = int(row["is_anemia"])

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


def load_dataframe() -> pd.DataFrame:
    if not DATASET_CSV.exists():
        raise SystemExit(f"CSV dataset tidak ditemukan: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)
    df = df[df["Tingkat_Anemia"] != "Tidak Diketahui"].copy()
    df = df[~df["HB"].isna()].copy()

    anemia_labels = {"Anemia Berat", "Anemia Ringan", "Anemia Sedang"}
    df["is_anemia"] = df["Tingkat_Anemia"].apply(lambda x: 1 if x in anemia_labels else 0)

    return df


def build_resnet18_binary(num_classes: int = 2):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(device: torch.device):
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise SystemExit("Model atau file label belum ditemukan. Latih model terlebih dahulu.")

    labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
    model = build_resnet18_binary(num_classes=len(labels))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, labels


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluasi di device:", device)

    df = load_dataframe()
    print("Total sampel untuk evaluasi:", len(df))

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = EvalDataset(df, transforms=transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model, labels = load_model(device)

    all_preds: list[int] = []
    all_probs: list[float] = []
    all_targets: list[int] = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)

    print("\n=== Confusion Matrix (0=Non-Anemia, 1=Anemia) ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["Non-Anemia", "Anemia"]))

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax_cm)
    classes = ["Non-Anemia", "Anemia"]
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels(classes, rotation=45, ha="right")
    ax_cm.set_yticklabels(classes)
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
    ax_cm.set_title("Binary Confusion Matrix")
    fig_cm.tight_layout()
    cm_path = reports_dir / "confusion_matrix_binary.png"
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)

    try:
        auc = roc_auc_score(y_true, y_score)
        print("ROC-AUC:", round(auc, 4))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Binary ROC Curve")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        roc_path = reports_dir / "roc_curve_binary.png"
        fig_roc.savefig(roc_path, dpi=300)
        plt.close(fig_roc)
    except ValueError:
        print("ROC-AUC tidak dapat dihitung (mungkin hanya 1 kelas di y_true).")

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
        ax_loss.set_title("Training and Validation Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, train_acc, label="Training Accuracy")
        ax_acc.plot(epochs, val_acc, label="Validation Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Training and Validation Accuracy")
        ax_acc.legend()

        fig_hist.tight_layout()
        hist_path = reports_dir / "training_curves_binary.png"
        fig_hist.savefig(hist_path, dpi=300)
        plt.close(fig_hist)

    class_counts = np.bincount(y_true, minlength=2)
    fig_bar, ax_bar = plt.subplots(figsize=(4, 4))
    ax_bar.bar([0, 1], class_counts, tick_label=["Non-Anemia", "Anemia"], color=["#4C72B0", "#DD8452"])
    ax_bar.set_ylabel("Number of Samples")
    ax_bar.set_title("Label Distribution on Evaluation Set")
    fig_bar.tight_layout()
    bar_path = reports_dir / "label_distribution_binary.png"
    fig_bar.savefig(bar_path, dpi=300)
    plt.close(fig_bar)

    probs_non_anemia = y_score[y_true == 0]
    probs_anemia = y_score[y_true == 1]
    fig_hist_prob, ax_prob = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0.0, 1.0, 21)
    ax_prob.hist(
        probs_non_anemia,
        bins=bins,
        alpha=0.6,
        label="Non-Anemia (label 0)",
        color="#4C72B0",
        density=True,
    )
    ax_prob.hist(
        probs_anemia,
        bins=bins,
        alpha=0.6,
        label="Anemia (label 1)",
        color="#DD8452",
        density=True,
    )
    ax_prob.set_xlabel("Predicted Probability of Anemia")
    ax_prob.set_ylabel("Density")
    ax_prob.set_title("Distribution of Predicted Probabilities")
    ax_prob.legend()
    fig_hist_prob.tight_layout()
    prob_path = reports_dir / "probability_distribution_binary.png"
    fig_hist_prob.savefig(prob_path, dpi=300)
    plt.close(fig_hist_prob)


if __name__ == "__main__":
    main()
