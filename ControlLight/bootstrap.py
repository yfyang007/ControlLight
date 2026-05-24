from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_local_paths() -> Path:
    """Ensure the vendored local package paths are importable.

    Returns the repository root so callers can reuse it when needed.
    """

    repo_root = Path(__file__).resolve().parent.parent
    local_diffusers_src = repo_root / "diffusers" / "src"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if local_diffusers_src.is_dir() and str(local_diffusers_src) not in sys.path:
        sys.path.insert(0, str(local_diffusers_src))

    return repo_root
