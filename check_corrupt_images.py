from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"


BAD_IMAGES_CSV = Path("data") / "bad_images.csv"


def main() -> None:
    if not DATASET_CSV.exists():
        raise SystemExit(f"CSV dataset tidak ditemukan: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)

    bad_rows: list[dict] = []

    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        no = row.get("No")
        nama = row.get("Nama")

        if not img_path.exists():
            bad_rows.append(
                {
                    "reason": "file_not_found",
                    "image_path": str(img_path),
                    "No": no,
                    "Nama": nama,
                }
            )
            continue

        try:
            with Image.open(img_path) as im:  
                im.verify()
        except UnidentifiedImageError:
            bad_rows.append(
                {
                    "reason": "unidentified_image",
                    "image_path": str(img_path),
                    "No": no,
                    "Nama": nama,
                }
            )
        except Exception as e:  
            bad_rows.append(
                {
                    "reason": f"other_error: {e}",
                    "image_path": str(img_path),
                    "No": no,
                    "Nama": nama,
                }
            )

    if bad_rows:
        out_df = pd.DataFrame(bad_rows)
        BAD_IMAGES_CSV.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(BAD_IMAGES_CSV, index=False)
        print(f"Ditemukan {len(bad_rows)} gambar bermasalah.")
        print(f"Detail disimpan di: {BAD_IMAGES_CSV.resolve()}")
        print("Contoh 10 baris pertama:")
        print(out_df.head(10))
    else:
        print("Tidak ditemukan gambar bermasalah.")


if __name__ == "__main__":
    main()
