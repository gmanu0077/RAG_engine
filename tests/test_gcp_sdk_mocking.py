"""Assessment brief: mocks at ``vertexai.language_models`` literal import path."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from rag_engine.vertex_stubs import ensure_vertexai_stub_modules


def test_vertexai_module_surface_is_present() -> None:
    ensure_vertexai_stub_modules()
    import vertexai.language_models as vlm

    assert hasattr(vlm, "TextEmbeddingModel")
    assert hasattr(vlm, "GenerativeModel")
    assert hasattr(vlm, "TextEmbeddingInput")


def test_text_embedding_model_surface_matches_sdk_shape() -> None:
    ensure_vertexai_stub_modules()
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

    m = TextEmbeddingModel.from_pretrained("textembedding-gecko-mock", dimensions=8)
    out = m.get_embeddings([TextEmbeddingInput(text="hello")])
    assert len(out) == 1
    assert len(out[0].values) == 8


def test_query_expansion_uses_vertex_generative_model_path() -> None:
    class SpyGen:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name

        def generate_content(self, prompt: str):
            from vertexai.language_models import GenerativeResponse

            return GenerativeResponse("mock-expanded-query-unique")

    with mock.patch("vertexai.language_models.GenerativeModel", SpyGen):
        import vertexai.language_models as vlm

        from rag_engine.config.loader import load_engine_config
        from rag_engine.retrieval.query_expander import QueryExpander

        cfg = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
        ex = QueryExpander(vlm.GenerativeModel("test-model"), cfg.query_expansion)
        out = ex.expand("How does the system handle peak load?")
        assert out == "mock-expanded-query-unique"


def test_rage_engine_uses_vertex_stub_generative_model() -> None:
    """Default ``RAGEngine`` wires a ``GenerativeModel`` instance from the Vertex module path."""
    from vertexai.language_models import GenerativeModel

    from rag_engine.app import RAGEngine
    from rag_engine.config.loader import load_engine_config

    cfg = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    cfg.embedding.provider = "mock_vertex"
    cfg.embedding.mock_vertex = cfg.embedding.mock_vertex.model_copy(update={"dimensions": 16})
    cfg.vector_store.provider = "numpy"
    eng = RAGEngine(config=cfg)
    assert isinstance(eng.retriever._expander._model, GenerativeModel)
