"""Load ``EngineConfig`` from YAML (plan §8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from rag_engine.config.schema import EngineConfig


def load_engine_config(path: str | Path | None = None) -> EngineConfig:
    root = Path(__file__).resolve().parents[3]
    p = Path(path) if path is not None else root / "config" / "config.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data: dict[str, Any] = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")
    return EngineConfig.model_validate(data)
