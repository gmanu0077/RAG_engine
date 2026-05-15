"""Query expansion (plan §4.3) via ``vertexai.language_models.GenerativeModel`` (stub or SDK)."""

from __future__ import annotations

from rag_engine.config.schema import QueryExpansionConfig
from rag_engine.vertex_stubs import ensure_vertexai_stub_modules

ensure_vertexai_stub_modules()
from vertexai.language_models import GenerativeModel, GenerativeResponse  # noqa: E402

# Back-compat for callers/tests expecting the old names.
MockGenerativeModel = GenerativeModel
MockResponse = GenerativeResponse


class QueryExpander:
    def __init__(self, generative_model: object, cfg: QueryExpansionConfig) -> None:
        self._model = generative_model
        self._cfg = cfg

    def expand(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if not self._cfg.enabled or self._cfg.provider == "none":
            return query.strip()
        prompt = self._cfg.prompt_template.format(query=query.strip())
        response = self._model.generate_content(prompt)
        expanded = getattr(response, "text", str(response)).strip()
        if not expanded:
            return query.strip()
        max_c = self._cfg.expansion_max_chars
        if len(expanded) <= max_c:
            return expanded
        cut = expanded[:max_c].rsplit(" ", 1)[0]
        return cut if cut else expanded[:max_c]
