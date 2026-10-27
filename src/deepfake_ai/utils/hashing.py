"""Cryptographic checksums and integrity verification."""

import hashlib
from pathlib import Path


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_md5(text: str) -> str:
    """Computes MD5 hash of a string."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()
