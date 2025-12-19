import os
from pathlib import Path
import time
import re

import pandas as pd
import requests

EXCEL_PATH = r"d:\project anemia\anemia-detector\Dataset Fingernails SMP 4.xlsx"

OUTPUT_DIR = Path("data") / "images"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_drive_file_id(url: str) -> str | None:
    """Ekstrak file ID dari URL Google Drive seperti
    https://drive.google.com/file/d/<ID>/view?usp=sharing
    """
    if not isinstance(url, str):
        return None
    match = re.search(r"/d/([^/]+)/", url)
    if match:
        return match.group(1)
    return None


def build_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_image(row: pd.Series) -> tuple[bool, str]:
    link = row.get("Link Foto Kuku")
    nama = str(row.get("Nama", "unknown")).strip().replace(" ", "_")
    no = row.get("No")

    file_id = extract_drive_file_id(link)
    if not file_id:
        return False, f"Gagal ekstrak file ID dari URL: {link}"

    url = build_direct_download_url(file_id)

    if pd.notna(no):
        filename = f"{int(no):03d}_{nama}.jpg"
    else:
        filename = f"{nama}.jpg"

    out_path = OUTPUT_DIR / filename

    if out_path.exists():
        return True, f"Sudah ada, skip: {out_path}"

    try:
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code} untuk {url}"

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True, f"Berhasil: {out_path}"
    except Exception as e:  # noqa: BLE001
        return False, f"Error saat download {url}: {e}"


def main() -> None:
    ensure_output_dir()

    df = pd.read_excel(EXCEL_PATH)

    total = len(df)
    sukses = 0
    gagal = 0

    print(f"Total baris data: {total}")
    print(f"Folder output: {OUTPUT_DIR.resolve()}")
    print()

    for idx, row in df.iterrows():
        ok, msg = download_image(row)
        if ok:
            sukses += 1
        else:
            gagal += 1
        print(f"[{idx+1}/{total}] {msg}")

        time.sleep(0.5)

    print()
    print("=== Ringkasan unduhan gambar ===")
    print(f"Berhasil : {sukses}")
    print(f"Gagal    : {gagal}")


if __name__ == "__main__":
    main()
