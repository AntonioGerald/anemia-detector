from pathlib import Path

import pandas as pd

DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"


def main() -> None:
    if not DATASET_CSV.exists():
        raise SystemExit(f"CSV dataset tidak ditemukan: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)

    print("=== Kondisi sebelum perbaikan ===")
    total = len(df)
    unknown_before = (df["Tingkat_Anemia"] == "Tidak Diketahui").sum()
    hb_na_before = df["HB"].isna().sum()
    print(f"Total baris           : {total}")
    print(f"Tidak Diketahui       : {unknown_before}")
    print(f"HB kosong (NaN)       : {hb_na_before}")

    
    mask_missing_hb = df["HB"].isna()

    
    df.loc[mask_missing_hb, "HB"] = 12.0

    
    mask_unknown_label = df["Tingkat_Anemia"] == "Tidak Diketahui"
    df.loc[mask_unknown_label & mask_missing_hb, "Tingkat_Anemia"] = "Tidak Anemia"

    
    df.to_csv(DATASET_CSV, index=False)

    
    print("\n=== Kondisi sesudah perbaikan ===")
    df2 = pd.read_csv(DATASET_CSV)
    unknown_after = (df2["Tingkat_Anemia"] == "Tidak Diketahui").sum()
    hb_na_after = df2["HB"].isna().sum()
    print(f"Total baris           : {len(df2)}")
    print(f"Tidak Diketahui       : {unknown_after}")
    print(f"HB kosong (NaN)       : {hb_na_after}")
    print("Perubahan disimpan ke:", DATASET_CSV.resolve())


if __name__ == "__main__":
    main()
