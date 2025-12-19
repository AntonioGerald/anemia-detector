import pandas as pd
import numpy as np

EXCEL_PATH = r"d:\project anemia\anemia-detector\Dataset Fingernails SMP 4.xlsx"


def kategori_anemia(hb: float) -> str:
    """Mengembalikan kategori anemia berdasarkan nilai HB (g/dL)."""
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
    df = pd.read_excel(EXCEL_PATH)

    print("=== Info tipe data kolom (sebelum konversi HB) ===")
    print(df.dtypes)
    print()

    df["HB_numeric"] = pd.to_numeric(df["HB"], errors="coerce")

    print("=== 5 nilai pertama kolom HB dan HB_numeric ===")
    print(df[["HB", "HB_numeric"]].head())
    print()

    df["Tingkat_Anemia"] = df["HB_numeric"].apply(kategori_anemia)

    print("=== 5 data pertama (dengan Tingkat_Anemia) ===")
    print(df.head())
    print()

    print("=== Distribusi Tingkat_Anemia ===")
    print(df["Tingkat_Anemia"].value_counts())


if __name__ == "__main__":
    main()
