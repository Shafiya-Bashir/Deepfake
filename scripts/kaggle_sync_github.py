#!/usr/bin/env python3
"""
kaggle_sync_github.py
======================
Automated GitHub Synchronization for Kaggle GPU Training Workflows.
Syncs trained model checkpoints (.pt) and forensic logs from Kaggle directly to GitHub.

Research Team: Dar Rehan Rasool, Dr. Riyaz Ahmed, Dr. M. Abdul Jawad (M.Tech Students, Centre for Artificial Intelligence, IUST)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd_list, cwd=None, mask_secret=None):
    """Executes a command safely without leaking secrets into printed outputs (Python 3.6+ compatible)."""
    display_cmd = " ".join(cmd_list)
    if mask_secret:
        display_cmd = display_cmd.replace(mask_secret, "********")
    print(f"[*] Executing: {display_cmd}")
    
    result = subprocess.run(
        cmd_list,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if result.returncode != 0:
        err_msg = result.stderr.strip()
        if mask_secret:
            err_msg = err_msg.replace(mask_secret, "********")
        print(f"[!] Warning/Error during execution: {err_msg}", file=sys.stderr)
    return result


def sync_to_github(token: str, username: str = "darrehanrasool", repo: str = "Deepfake", branch: str = "main", message: str = None):
    if not token or not token.strip():
        print("[!] Error: GitHub Personal Access Token is required.", file=sys.stderr)
        sys.exit(1)

    token = token.strip()
    
    # 1. Determine Repository Root
    # In Kaggle, the repo is usually /kaggle/working/Deepfake or current directory
    cwd = Path.cwd()
    if (cwd / ".git").exists():
        repo_root = cwd
    elif (Path("/kaggle/working/Deepfake") / ".git").exists():
        repo_root = Path("/kaggle/working/Deepfake")
    elif (cwd / "Deepfake" / ".git").exists():
        repo_root = cwd / "Deepfake"
    else:
        # Search parent directories
        repo_root = None
        for p in [cwd] + list(cwd.parents):
            if (p / ".git").exists():
                repo_root = p
                break
        if repo_root is None:
            print("[!] Error: Could not locate a Git repository in or around current working directory.", file=sys.stderr)
            sys.exit(1)

    print(f"[✓] Detected Git Repository Root: {repo_root}")

    # 2. Collect artifacts from Kaggle working directory if distinct from repo
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        # Checkpoints
        kaggle_chk = kaggle_working / "checkpoints"
        repo_chk = repo_root / "checkpoints"
        if kaggle_chk.exists() and kaggle_chk != repo_chk:
            print(f"[*] Copying checkpoints from {kaggle_chk} to {repo_chk}...")
            repo_chk.mkdir(parents=True, exist_ok=True)
            for pt_file in kaggle_chk.rglob("*.pt"):
                rel = pt_file.relative_to(kaggle_chk)
                dest = repo_chk / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(pt_file), str(dest))
                print(f"    - Synced: {rel} ({dest.stat().st_size / (1024*1024):.2f} MB)")

        # Logs & CSV histories
        kaggle_logs = kaggle_working / "logs"
        repo_logs = repo_root / "logs"
        if kaggle_logs.exists() and kaggle_logs != repo_logs:
            print(f"[*] Copying logs from {kaggle_logs} to {repo_logs}...")
            repo_logs.mkdir(parents=True, exist_ok=True)
            for log_file in kaggle_logs.rglob("*"):
                if log_file.is_file():
                    rel = log_file.relative_to(kaggle_logs)
                    dest = repo_logs / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(log_file), str(dest))

        history_csv = kaggle_working / "kaggle_training_history.csv"
        if history_csv.exists():
            dest = repo_root / "logs" / "training" / "kaggle_training_history.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(history_csv), str(dest))

    # 3. Configure Git credentials locally for this repository
    run_cmd(["git", "config", "user.name", "Dar Rehan Rasool"], cwd=repo_root)
    run_cmd(["git", "config", "user.email", "darrehanrasool@users.noreply.github.com"], cwd=repo_root)

    # 4. Set remote URL with authentication
    remote_auth_url = f"https://{username}:{token}@github.com/{username}/{repo}.git"
    run_cmd(["git", "remote", "set-url", "origin", remote_auth_url], cwd=repo_root, mask_secret=token)

    # 5. Ensure branch
    run_cmd(["git", "checkout", "-B", branch], cwd=repo_root)

    # 6. Pull latest remote changes first to avoid non-fast-forward rejection
    print("[*] Fetching and integrating latest remote commits...")
    run_cmd(["git", "pull", "--rebase", "origin", branch], cwd=repo_root, mask_secret=token)

    # 7. Stage checkpoints, logs, and artifacts
    print("[*] Staging trained checkpoints, models, and training logs...")
    run_cmd(["git", "add", "checkpoints/", "logs/", "models/weights/"], cwd=repo_root)

    # 8. Check for changes
    status_res = run_cmd(["git", "status", "--porcelain"], cwd=repo_root)
    if not status_res.stdout.strip():
        print("[*] No new changes to commit. Everything is already up to date with GitHub!")
        return

    # 9. Commit
    commit_msg = message or "Auto-sync trained model checkpoints and training logs from Kaggle GPU"
    run_cmd(["git", "commit", "-m", commit_msg], cwd=repo_root)

    # 10. Push to remote
    print(f"[*] Pushing synchronized checkpoints to GitHub: {username}/{repo} ({branch})...")
    push_res = run_cmd(["git", "push", "origin", branch], cwd=repo_root, mask_secret=token)

    if push_res.returncode == 0:
        print("\n" + "=" * 80)
        print(f"[★] SUCCESS! Checkpoints and logs are now live on GitHub:")
        print(f"    https://github.com/{username}/{repo}")
        print("-" * 80)
        print("[★] TO SYNC ON YOUR LOCAL MAC / VS CODE:")
        print(f"    Run this command in your VS Code terminal:")
        print(f"        git pull origin {branch}")
        print("=" * 80 + "\n")
    else:
        print("[!] Push encountered an issue. See output above.", file=sys.stderr)


def test_token_permissions(token: str, username: str = "darrehanrasool", repo: str = "Deepfake", branch: str = "main") -> bool:
    """Pre-flight check: Verifies if the GitHub token has push permissions using git dry-run."""
    token = token.strip()
    cwd = Path.cwd()
    repo_root = cwd if (cwd / ".git").exists() else None
    if not repo_root:
        for p in [cwd] + list(cwd.parents):
            if (p / ".git").exists():
                repo_root = p
                break
    if not repo_root and (Path("/kaggle/working/Deepfake") / ".git").exists():
        repo_root = Path("/kaggle/working/Deepfake")
    if not repo_root and (cwd / "Deepfake" / ".git").exists():
        repo_root = cwd / "Deepfake"

    if not repo_root:
        print("[!] Error: No Git repository found to test token against.", file=sys.stderr)
        return False

    print("=" * 70)
    print("      GITHUB TOKEN PRE-FLIGHT WRITE PERMISSION TEST")
    print("=" * 70)
    remote_auth_url = f"https://{username}:{token}@github.com/{username}/{repo}.git"
    run_cmd(["git", "remote", "set-url", "origin", remote_auth_url], cwd=repo_root, mask_secret=token)
    
    print("[*] Performing live test push with --dry-run (zero changes made)...")
    res = run_cmd(["git", "push", "--dry-run", "origin", branch], cwd=repo_root, mask_secret=token)

    if res.returncode == 0:
        print("\n" + "=" * 70)
        print("[✓] SUCCESS! Your GitHub token has 100% verified WRITE permissions!")
        print(f"    Target Repo: https://github.com/{username}/{repo} ({branch})")
        print("    You can safely run overnight training and go to sleep.")
        print("    In the morning, just run 'git pull origin main' in VS Code!")
        print("=" * 70 + "\n")
        return True
    else:
        print("\n" + "=" * 70)
        print("[!] PERMISSION DENIED (403): Token lacks write permission.")
        print("-" * 70)
        print("HOW TO FIX IN 30 SECONDS ON GITHUB:")
        print("1. Open: https://github.com/settings/tokens/new")
        print("2. Note: 'Kaggle Overnight Training'")
        print("3. Scopes: CHECK THE 'repo' BOX (Full control of private repositories).")
        print("4. Click 'Generate token' at bottom and copy the new token.")
        print("=" * 70 + "\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kaggle training checkpoints and logs to GitHub")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""), help="GitHub Personal Access Token")
    parser.add_argument("--username", default="darrehanrasool", help="GitHub username")
    parser.add_argument("--repo", default="Deepfake", help="GitHub repository name")
    parser.add_argument("--branch", default="main", help="Git branch to push to")
    parser.add_argument("--message", default=None, help="Commit message")
    parser.add_argument("--test-token", default=None, help="Test token push permissions before training")
    args = parser.parse_args()

    if args.test_token:
        success = test_token_permissions(args.test_token, username=args.username, repo=args.repo, branch=args.branch)
        sys.exit(0 if success else 1)

    token_val = args.token or os.environ.get("GITHUB_TOKEN", "")
    sync_to_github(
        token=token_val,
        username=args.username,
        repo=args.repo,
        branch=args.branch,
        message=args.message
    )
