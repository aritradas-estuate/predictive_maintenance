from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import kagglehub
import requests

UCI_METRO_ZIP_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
KAGGLE_EV_HANDLE = "kanchana1990/ev-battery-qc-synthetic-defect-dataset"
SUPPORTED_SUFFIXES = (".csv", ".parquet", ".feather")

ROOT = Path(__file__).resolve().parents[1]
METRO_DIR = ROOT / "data" / "raw" / "metropt3"
EV_DIR = ROOT / "data" / "raw" / "ev_battery_qc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch MetroPT-3 from UCI and EV Battery QC from Kaggle."
    )
    parser.add_argument(
        "--metro-only",
        action="store_true",
        help="Download only MetroPT-3.",
    )
    parser.add_argument(
        "--ev-only",
        action="store_true",
        help="Download only EV Battery QC.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing raw dataset files.",
    )
    args = parser.parse_args()
    if args.metro_only and args.ev_only:
        parser.error("--metro-only and --ev-only cannot be used together.")
    return args


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def existing_supported_files(directory: Path) -> list[Path]:
    files = []
    for suffix in SUPPORTED_SUFFIXES:
        files.extend(sorted(directory.glob(f"*{suffix}")))
    return files


def clear_supported_files(directory: Path) -> None:
    for path in existing_supported_files(directory):
        path.unlink()


def copy_into_dir(source: Path, destination_dir: Path, destination_name: str | None = None) -> Path:
    ensure_dir(destination_dir)
    target = destination_dir / (destination_name or source.name)
    shutil.copy2(source, target)
    return target


def fetch_metro(force: bool = False) -> Path:
    ensure_dir(METRO_DIR)
    existing = existing_supported_files(METRO_DIR)
    if existing and not force:
        print(f"[skip] MetroPT-3 already present: {existing[0].name}")
        return existing[0]

    if force:
        clear_supported_files(METRO_DIR)

    print("[fetch] MetroPT-3 from official UCI archive")
    with tempfile.TemporaryDirectory(prefix="metropt3-") as temp_dir:
        temp_root = Path(temp_dir)
        zip_path = temp_root / "metropt3_dataset.zip"
        with requests.get(UCI_METRO_ZIP_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with zip_path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_obj.write(chunk)

        with zipfile.ZipFile(zip_path) as archive:
            csv_members = [
                member
                for member in archive.namelist()
                if member.lower().endswith(".csv")
            ]
            if not csv_members:
                raise RuntimeError("MetroPT-3 download did not contain a CSV file.")
            csv_member = csv_members[0]
            archive.extract(csv_member, path=temp_root)
            extracted_csv = temp_root / csv_member

        target = copy_into_dir(
            extracted_csv,
            METRO_DIR,
            destination_name="metropt3_air_compressor.csv",
        )
        print(f"[done] MetroPT-3 saved to {target}")
        return target


def require_kaggle_token() -> None:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing KAGGLE_API_TOKEN. Export it in your shell before fetching EV Battery QC."
        )


def fetch_ev(force: bool = False) -> Path:
    ensure_dir(EV_DIR)
    existing = existing_supported_files(EV_DIR)
    if existing and not force:
        print(f"[skip] EV Battery QC already present: {existing[0].name}")
        return existing[0]

    if force:
        clear_supported_files(EV_DIR)

    require_kaggle_token()
    print("[fetch] EV Battery QC from Kaggle")
    with tempfile.TemporaryDirectory(prefix="ev-battery-qc-") as temp_dir:
        download_root = Path(
            kagglehub.dataset_download(
                KAGGLE_EV_HANDLE,
                output_dir=temp_dir,
                force_download=force,
            )
        )
        dataset_files = [
            path
            for path in sorted(download_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not dataset_files:
            raise RuntimeError("EV Battery QC download did not contain a supported data file.")

        source = dataset_files[0]
        target = copy_into_dir(source, EV_DIR)
        print(f"[done] EV Battery QC saved to {target}")
        return target


def main() -> int:
    args = parse_args()
    run_metro = not args.ev_only
    run_ev = not args.metro_only

    try:
        if run_metro:
            fetch_metro(force=args.force)
        if run_ev:
            fetch_ev(force=args.force)
    except Exception as exc:  # pragma: no cover - command-line guardrail
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
