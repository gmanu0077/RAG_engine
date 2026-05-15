"""Pytest hooks: Vertex stub modules for SDK-shaped imports."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.vertex_stubs import ensure_vertexai_stub_modules


def pytest_configure(config: object) -> None:
    ensure_vertexai_stub_modules()


@pytest.fixture()
def data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data"
