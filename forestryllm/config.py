from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/project_config.json")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ((base_dir or project_root()) / candidate).resolve()


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root())
    return config
