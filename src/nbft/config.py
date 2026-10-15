"""Resolve data paths from config/paths.yaml.

Lookup order for the data root:
1. ``NBFT_DATA_ROOT`` environment variable
2. ``data_root`` in ``config/paths.local.yaml`` (gitignored)
3. ``data_root`` in ``config/paths.yaml``
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@lru_cache(maxsize=1)
def load_config() -> dict:
    local = CONFIG_DIR / "paths.local.yaml"
    path = local if local.exists() else CONFIG_DIR / "paths.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    root = os.environ.get("NBFT_DATA_ROOT") or load_config()["data_root"]
    root = Path(root).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


def path(key: str, **fmt) -> Path:
    """Absolute data path for a dotted key.

    Example: ``path("statics.npr_ref", am_pm="am")``.
    """
    node = load_config()
    for part in key.split("."):
        node = node[part]
    return data_root() / node.format(**fmt)
