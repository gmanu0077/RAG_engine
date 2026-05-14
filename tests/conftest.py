"""Minimal Vertex namespace so ``unittest.mock.patch(..., create=True)`` resolves."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def pytest_configure(config: object) -> None:
    if "vertexai" not in sys.modules:
        vertexai = types.ModuleType("vertexai")
        lm = types.ModuleType("vertexai.language_models")
        vertexai.language_models = lm  # type: ignore[attr-defined]
        sys.modules["vertexai"] = vertexai
        sys.modules["vertexai.language_models"] = lm


@pytest.fixture()
def data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data"
