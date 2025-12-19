import os
from pathlib import Path

import pandas as pd

EXCEL_PATH = r"d:\project anemia\anemia-detector\Dataset Fingernails SMP 4.xlsx"


IMAGES_DIR = Path("data") / "images"


OUTPUT_CSV = Path("data") / "dataset_images_with_labels.csv"


def kategori_anemia(hb: float) -> str:
    if pd.isna(hb):
        return "Tidak Diketahui"
    if hb < 8:
        return "Anemia Berat"
    elif 8 <= hb <= 10.9:
        return "Anemia Sedang"
    elif 11 <= hb <= 11.9:
        return "Anemia Ringan"
    elif hb >= 12:
        return "Tidak Anemia"
    else:
        return "Tidak Diketahui"


def main() -> None:
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Folder gambar tidak ditemukan: {IMAGES_DIR}")

    
    df = pd.read_excel(EXCEL_PATH)

    
    df["HB_numeric"] = pd.to_numeric(df["HB"], errors="coerce")
    df["Tingkat_Anemia"] = df["HB_numeric"].apply(kategori_anemia)

    
    records: list[dict] = []

    for _, row in df.iterrows():
        no = row.get("No")
        nama = str(row.get("Nama", "unknown")).strip().replace(" ", "_")
        label = row.get("Tingkat_Anemia")
        hb = row.get("HB_numeric")

        if pd.isna(no):
            
            continue

        
        filename = f"{int(no):03d}_{nama}.jpg"
        img_path = IMAGES_DIR / filename

        if not img_path.exists():
            
            continue

        records.append(
            {
                "image_path": str(img_path.as_posix()),
                "No": int(no),
                "Nama": row.get("Nama"),
                "HB": hb,
                "Tingkat_Anemia": label,
            }
        )

    if not records:
        raise SystemExit("Tidak ada record yang memiliki gambar dan label.")

    out_df = pd.DataFrame.from_records(records)

    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total baris di Excel: {len(df)}")
    print(f"Total sampel dengan gambar & label: {len(out_df)}")
    print(f"Contoh 5 baris: \n{out_df.head()}\n")
    print(f"Dataset CSV disimpan di: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
