from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "anemia_rf_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"


def extract_color_features(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    
    img = cv2.resize(img, (128, 128))

    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
    mean_h = hsv[:, :, 0].mean()
    mean_s = hsv[:, :, 1].mean()
    mean_v = hsv[:, :, 2].mean()

    return np.array([mean_h, mean_s, mean_v], dtype=np.float32)


def build_features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for _, row in df.iterrows():
        img_path = row["image_path"]
        label = row["Tingkat_Anemia"]

        try:
            feats = extract_color_features(img_path)
        except Exception as e:  
            print(f"Lewati {img_path} karena error: {e}")
            continue

        X_list.append(feats)
        y_list.append(label)

    if not X_list:
        raise SystemExit("Tidak ada fitur yang berhasil diekstrak dari gambar.")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def main() -> None:
    if not DATASET_CSV.exists():
        raise SystemExit(f"CSV dataset tidak ditemukan: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)

    print("=== Contoh 5 baris dataset ===")
    print(df.head())
    print()

    print("Menghitung fitur warna rata-rata untuk setiap gambar...")
    X, y = build_features_and_labels(df)

    print(f"Total sampel setelah ekstraksi fitur: {X.shape[0]}")
    print(f"Dimensi fitur: {X.shape[1]}")

    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Melatih model RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    print("Evaluasi di data test...")
    y_pred = clf.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print()
    print(f"Model disimpan di: {MODEL_PATH.resolve()}")
    print(f"Label encoder disimpan di: {LABEL_ENCODER_PATH.resolve()}")


if __name__ == "__main__":
    main()
