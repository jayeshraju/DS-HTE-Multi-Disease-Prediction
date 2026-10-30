"""
Provenance tracking and sidecar verification for DS-HTE framework artifacts.
Enforces Ground Rule 1:
- Every result, table, figure, and benchmark artifact must have a matching `*.provenance.json` sidecar.
- Contains: script_name, git commit or file hash, run_id, config_hash, timestamp, file_hash, input_file_hashes.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash() -> str:
    """Gets the current git commit hash if available, otherwise fallback to code hash."""
    try:
        import subprocess
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "git_commit_unavailable"


def write_provenance(target_file: str,
                     script_name: str,
                     run_id: str = "dshte_paper_20261029_113441",
                     config_hash: str = "9ac2a61966dc6226",
                     input_files: Optional[List[str]] = None,
                     extra_meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Writes a *.provenance.json sidecar for target_file.
    """
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Cannot write provenance for non-existent file: {target_file}")

    sidecar_path = f"{target_file}.provenance.json"
    file_hash = compute_sha256(target_file)
    git_hash = get_git_commit_hash()

    input_hashes = {}
    if input_files:
        for inp in input_files:
            if os.path.exists(inp):
                input_hashes[inp] = compute_sha256(inp)

    payload = {
        "artifact_file": os.path.basename(target_file),
        "target_path": target_file.replace("\\", "/"),
        "script_name": script_name,
        "run_id": run_id,
        "config_hash": config_hash,
        "git_commit": git_hash,
        "file_sha256": file_hash,
        "timestamp": datetime.now().isoformat(),
        "input_file_hashes": input_hashes
    }
    if extra_meta:
        payload["metadata"] = extra_meta

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return sidecar_path


def verify_provenance(target_file: str, expected_run_id: Optional[str] = None) -> bool:
    """
    Verifies that target_file has a valid, non-tampered *.provenance.json sidecar.
    """
    sidecar_path = f"{target_file}.provenance.json"
    if not os.path.exists(sidecar_path):
        return False
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        current_hash = compute_sha256(target_file)
        if data.get("file_sha256") != current_hash:
            return False
        if expected_run_id and data.get("run_id") != expected_run_id:
            return False
        return True
    except Exception:
        return False


def generate_provenance_for_directory(dir_path: str,
                                      script_name: str,
                                      run_id: str,
                                      config_hash: str,
                                      extensions: Optional[List[str]] = None,
                                      input_files: Optional[List[str]] = None):
    """
    Generates provenance sidecars for all files in a directory matching extensions.
    """
    if not os.path.exists(dir_path):
        return
    if extensions is None:
        extensions = [".csv", ".md", ".json", ".parquet", ".pdf", ".png", ".joblib", ".pkl"]

    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".provenance.json"):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                full_path = os.path.join(root, f)
                write_provenance(full_path, script_name, run_id, config_hash, input_files)
