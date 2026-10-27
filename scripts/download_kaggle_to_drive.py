#!/usr/bin/env python3
"""
DeepFake-AI: Kaggle to Google Drive Automated Downloader
=======================================================
Directly streams and downloads massive Kaggle datasets (e.g., DFDC 472GB)
into mounted Google Drive storage from Google Colab Pro, bypassing local disk storage.

Key Capabilities:
1. Zero local storage usage (routes 100% through cloud RAM / Drive FUSE).
2. Resumable chunk-by-chunk download (skips already downloaded zip shards).
3. Google Drive free space inspection and quota safeguards.
4. Automatic Kaggle credentials configuration from Drive or environment.

Authors: Dar Rehan Rasool (Research Contributor), Dr. Riyaz, Dr. Javad
Institution: Islamic University of Science and Technology (IUST), Kashmir
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


def check_google_drive_mount(drive_dir: Path) -> bool:
    """Verifies that Google Drive is mounted and accessible."""
    print(f"[*] Checking Google Drive destination path: {drive_dir}")
    drive_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space on destination
    try:
        total, used, free = shutil.disk_usage(drive_dir)
        print(f"[✓] Google Drive Storage Audit:")
        print(f"    - Total Space:     {total / (1024**3):.2f} GB")
        print(f"    - Used Space:      {used / (1024**3):.2f} GB")
        print(f"    - Free Available:  {free / (1024**3):.2f} GB")
        if free / (1024**3) < 10:
            print("[!] WARNING: Less than 10GB free space detected on destination!")
    except Exception as e:
        print(f"[!] Note on disk usage query: {e}")
    return True


def setup_kaggle_credentials(
    drive_kaggle_json: Path = Path("/content/drive/MyDrive/kaggle.json"),
    token: Optional[str] = None,
    username: Optional[str] = None,
    key: Optional[str] = None
) -> bool:
    """Configures Kaggle credentials for API access (supports new KGAT tokens & legacy json)."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    dest_token = kaggle_dir / "access_token"
    dest_json = kaggle_dir / "kaggle.json"

    # Option 1: New KGAT Access Token
    active_token = token or os.environ.get("KAGGLE_API_TOKEN")
    if not active_token and dest_token.exists():
        active_token = dest_token.read_text().strip()

    drive_token_file = Path("/content/drive/MyDrive/kaggle_access_token")
    if not active_token and drive_token_file.exists():
        active_token = drive_token_file.read_text().strip()

    if active_token:
        with open(dest_token, "w") as f:
            f.write(active_token.strip())
        os.chmod(dest_token, 0o600)
        os.environ["KAGGLE_API_TOKEN"] = active_token.strip()
        print(f"[✓] Configured Kaggle API Access Token (~/.kaggle/access_token).")
        # Persist to Drive if mounted
        if Path("/content/drive/MyDrive").exists():
            try:
                drive_token_file.write_text(active_token.strip())
                print(f"[✓] Backed up Kaggle token to Google Drive: {drive_token_file}")
            except Exception:
                pass
        return True

    # Option 2: Explicit username & API key
    if username and key:
        import json
        with open(dest_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(dest_json, 0o600)
        print(f"[✓] Configured Kaggle credentials from parameters.")
        return True

    # Option 3: Pre-existing kaggle.json on Google Drive
    if drive_kaggle_json.exists():
        shutil.copy(drive_kaggle_json, dest_json)
        os.chmod(dest_json, 0o600)
        print(f"[✓] Copied Kaggle API token from {drive_kaggle_json} to {dest_json}")
        return True

    # Option 4: Already configured in ~/.kaggle/kaggle.json
    if dest_json.exists():
        os.chmod(dest_json, 0o600)
        print(f"[✓] Using existing ~/.kaggle/kaggle.json")
        return True

    # Option 5: Environment variables
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        print("[✓] Using KAGGLE_USERNAME and KAGGLE_KEY from environment.")
        return True

    print("[!] ERROR: Kaggle credentials not found!")
    return False


def get_dfdc_competition_file_list() -> List[str]:
    """Returns the ordered list of all DFDC competition chunks."""
    files = ["sample_submission.csv"]
    # 50 training chunks (~9.5GB to 10GB each)
    for i in range(50):
        files.append(f"dfdc_train_part_{i}.zip")
    return files


def download_chunk(
    competition: str,
    file_name: str,
    destination_dir: Path,
    max_retries: int = 3
) -> bool:
    """Downloads an individual chunk directly to Google Drive, resuming if needed."""
    target_file = destination_dir / file_name

    # Check if file already exists and is non-empty
    if target_file.exists() and target_file.stat().st_size > 1024 * 1024:
        size_gb = target_file.stat().st_size / (1024**3)
        print(f"[→] Skipping {file_name}: already downloaded ({size_gb:.2f} GB).")
        return True

    print(f"\n=================================================================")
    print(f"[*] Downloading {file_name} directly to Google Drive...")
    print(f"[*] Destination: {target_file}")
    print(f"=================================================================")

    cmd = [
        "kaggle", "competitions", "download",
        "-c", competition,
        "-f", file_name,
        "-p", str(destination_dir)
    ]

    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            res = subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            if target_file.exists():
                size_mb = target_file.stat().st_size / (1024**2)
                rate = size_mb / max(elapsed, 0.1)
                print(f"[✓] Successfully downloaded {file_name} ({size_mb:.1f} MB in {elapsed:.1f}s, ~{rate:.1f} MB/s).")
                return True
        except subprocess.CalledProcessError as e:
            print(f"[!] Attempt {attempt}/{max_retries} failed for {file_name}: {e}")
            time.sleep(5)

    return False


def run_download_pipeline(
    competition: str = "deepfake-detection-challenge",
    destination_dir: str = "/content/drive/MyDrive/DeepFake_Datasets/DFDC",
    parts: Optional[str] = "0-4",  # Default to first 5 parts (~50GB) for safety, or 'all'
    kaggle_json_path: str = "/content/drive/MyDrive/kaggle.json",
    token: Optional[str] = None,
    username: Optional[str] = None,
    key: Optional[str] = None
):
    dest_path = Path(destination_dir)
    check_google_drive_mount(dest_path)

    if not setup_kaggle_credentials(Path(kaggle_json_path), token=token, username=username, key=key):
        sys.exit(1)

    all_files = get_dfdc_competition_file_list()
    
    # Parse requested parts
    selected_files = ["sample_submission.csv"]
    if parts.lower() == "all":
        selected_files.extend([f"dfdc_train_part_{i}.zip" for i in range(50)])
    else:
        # e.g., "0-4" or "0,1,2"
        if "-" in parts:
            start, end = map(int, parts.split("-"))
            part_indices = list(range(start, end + 1))
        else:
            part_indices = [int(p.strip()) for p in parts.split(",")]
        
        for idx in part_indices:
            if 0 <= idx < 50:
                selected_files.append(f"dfdc_train_part_{idx}.zip")

    print(f"\n[*] Total files scheduled for download: {len(selected_files)}")
    print(f"[*] Files: {selected_files}\n")

    successful = 0
    failed = []

    for fn in selected_files:
        ok = download_chunk(competition, fn, dest_path)
        if ok:
            successful += 1
        else:
            failed.append(fn)

    print("\n=================================================================")
    print(f"[✓] Download Pipeline Completed!")
    print(f"    - Total Processed:  {len(selected_files)}")
    print(f"    - Successful:       {successful}")
    print(f"    - Failed / Pending: {len(failed)}")
    if failed:
        print(f"    - Failed Chunks:    {failed}")
    print(f"    - Target Directory: {dest_path}")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle to Google Drive Direct Stream Downloader")
    parser.add_argument("--competition", default="deepfake-detection-challenge", help="Kaggle competition name")
    parser.add_argument("--destination", default="/content/drive/MyDrive/DeepFake_Datasets/DFDC", help="Google Drive path")
    parser.add_argument("--parts", default="0-4", help="Parts to download: '0-4', '0,1,2', or 'all'")
    parser.add_argument("--token", default=None, help="Kaggle API Access Token (KGAT_...)")
    parser.add_argument("--kaggle-json", default="/content/drive/MyDrive/kaggle.json", help="Path to kaggle.json on Drive")
    parser.add_argument("--username", default=None, help="Kaggle Username")
    parser.add_argument("--key", default=None, help="Kaggle API Key")
    args = parser.parse_args()

    run_download_pipeline(
        competition=args.competition,
        destination_dir=args.destination,
        parts=args.parts,
        kaggle_json_path=args.kaggle_json,
        token=args.token,
        username=args.username,
        key=args.key
    )
