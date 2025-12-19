from pathlib import Path

import pandas as pd
from PIL import Image

try:
    import pillow_heif
except ImportError as e:  
    raise SystemExit(
        "Modul 'pillow_heif' belum terinstal. Jalankan 'pip install pillow-heif' dulu di environment ini."
    ) from e


pillow_heif.register_heif_opener()


DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"
BAD_IMAGES_CSV = Path("data") / "bad_images.csv"


def convert_heic_like_file(image_path: Path) -> bool:
    try:
        img = Image.open(image_path).convert("RGB")
        
        img.save(image_path, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def main() -> None:
    if not BAD_IMAGES_CSV.exists():
        raise SystemExit(f"File bad_images.csv tidak ditemukan: {BAD_IMAGES_CSV}")

    bad_df = pd.read_csv(BAD_IMAGES_CSV)

    total = len(bad_df)
    sukses = 0
    gagal = 0

    print(f"Total gambar bermasalah yang akan dicoba dikonversi: {total}")

    for idx, row in bad_df.iterrows():
        img_path = Path(str(row["image_path"]))
        if not img_path.exists():
            print(f"[{idx+1}/{total}] File tidak ditemukan, lewati: {img_path}")
            gagal += 1
            continue

        ok = convert_heic_like_file(img_path)
        if ok:
            sukses += 1
            print(f"[{idx+1}/{total}] Berhasil konversi ke JPEG: {img_path}")
        else:
            gagal += 1
            print(f"[{idx+1}/{total}] Gagal konversi: {img_path}")

    print()
    print("=== Ringkasan konversi HEIC -> JPEG ===")
    print(f"Berhasil : {sukses}")
    print(f"Gagal    : {gagal}")
    print("Setelah ini, jalankan lagi step7_check_corrupt_images.py untuk mengecek ulang.")


if __name__ == "__main__":
    main()
