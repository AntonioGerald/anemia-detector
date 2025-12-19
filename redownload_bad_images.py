from pathlib import Path
import time
import re

import pandas as pd
import requests


EXCEL_PATH = Path(r"d:\project anemia\anemia-detector\Dataset Fingernails SMP 4.xlsx")


DATASET_CSV = Path("data") / "dataset_images_with_labels.csv"


BAD_IMAGES_CSV = Path("data") / "bad_images.csv"


OUTPUT_DIR = Path("data") / "images"


def extract_drive_file_id(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    match = re.search(r"/d/([^/]+)/", url)
    if match:
        return match.group(1)
    return None


def build_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_image(url: str, out_path: Path) -> tuple[bool, str]:
    try:
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True, "ok"
    except Exception as e:  
        return False, str(e)


def main() -> None:
    if not BAD_IMAGES_CSV.exists():
        raise SystemExit(f"File bad_images.csv tidak ditemukan: {BAD_IMAGES_CSV}")
    if not EXCEL_PATH.exists():
        raise SystemExit(f"File Excel tidak ditemukan: {EXCEL_PATH}")

    bad_df = pd.read_csv(BAD_IMAGES_CSV)
    excel_df = pd.read_excel(EXCEL_PATH)

    
    bad_df["No"] = pd.to_numeric(bad_df["No"], errors="coerce").astype("Int64")
    excel_df["No"] = pd.to_numeric(excel_df["No"], errors="coerce").astype("Int64")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(bad_df)
    sukses = 0
    gagal = 0

    print(f"Total gambar bermasalah yang akan dicoba diunduh ulang: {total}")

    for idx, row in bad_df.iterrows():
        no = row["No"]
        img_path = Path(str(row["image_path"]))

        if pd.isna(no):
            print(f"[{idx+1}/{total}] No kosong untuk {img_path}, lewati")
            gagal += 1
            continue

        
        match = excel_df[excel_df["No"] == no]
        if match.empty:
            print(f"[{idx+1}/{total}] No {no} tidak ditemukan di Excel, lewati")
            gagal += 1
            continue

        link = match.iloc[0].get("Link Foto Kuku")
        file_id = extract_drive_file_id(link)
        if not file_id:
            print(f"[{idx+1}/{total}] Gagal ekstrak file ID dari link: {link}")
            gagal += 1
            continue

        url = build_direct_download_url(file_id)

        
        out_path = img_path
        if not out_path.is_absolute():
            out_path = OUTPUT_DIR / out_path.name

        ok, msg = download_image(url, out_path)
        if ok:
            sukses += 1
            print(f"[{idx+1}/{total}] Berhasil unduh ulang: {out_path}")
        else:
            gagal += 1
            print(f"[{idx+1}/{total}] Gagal unduh ulang {out_path}: {msg}")

        time.sleep(0.5)

    print()
    print("=== Ringkasan unduhan ulang ===")
    print(f"Berhasil : {sukses}")
    print(f"Gagal    : {gagal}")


if __name__ == "__main__":
    main()
