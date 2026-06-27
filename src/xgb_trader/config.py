from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("cache_dir", "dataset_dir"):
        Path(config["data"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["model"]["output_dir"]).mkdir(parents=True, exist_ok=True)
