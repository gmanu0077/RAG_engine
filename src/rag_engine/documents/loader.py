"""Load JSON list records into ``Document`` objects."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rag_engine.documents.models import Document


def load_documents_plaintext(path: str | Path, *, document_id: str | None = None) -> list[Document]:
    """Load a single ``Document`` from a UTF-8 text or markdown file (whole file = one doc).

    Use this for long-form prose, logs, or books. IDs default to the file stem
    (sanitized); metadata records ``source`` and ``path``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Document not found: {p}")
    text = p.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"Plaintext file is empty: {p}")
    stem = p.stem
    safe = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_") or "document"
    did = document_id.strip() if document_id and document_id.strip() else safe
    meta: dict[str, Any] = {"source": "plaintext_file", "path": str(p.resolve())}
    return [Document(document_id=did, text=stripped, metadata=meta)]


def load_documents_json(path: str | Path, text_field: str, id_field: str) -> list[Document]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")
    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError("Dataset file is empty.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {p}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects.")
    out: list[Document] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be an object, got {type(item).__name__}.")
        text = item.get(text_field)
        if text is None:
            raise ValueError(f"Item {i} is missing field {text_field!r}.")
        if not isinstance(text, str):
            raise ValueError(f"Item {i} field {text_field!r} must be a string.")
        stripped = text.strip()
        if not stripped:
            continue
        did = item.get(id_field)
        if did is None or str(did).strip() == "":
            document_id = f"doc_{i:03d}"
        else:
            document_id = str(did).strip()
        meta = {k: v for k, v in item.items() if k not in (text_field, id_field)}
        out.append(Document(document_id=document_id, text=stripped, metadata=meta))
    if not out:
        raise ValueError("No non-empty documents after filtering.")
    return out
