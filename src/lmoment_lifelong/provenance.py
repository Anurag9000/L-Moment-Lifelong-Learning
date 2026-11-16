"""Source identity helpers for reproducible experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


def source_commit() -> str:
    """Return the commit executed by a launcher, or resolve the current checkout."""
    supplied = os.environ.get("LMOMENT_SOURCE_COMMIT")
    if supplied:
        return supplied
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_fingerprint(config: dict) -> str:
    """Hash configuration and implementation commit to prevent stale skips."""
    payload = {"config": config, "source_commit": source_commit()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
