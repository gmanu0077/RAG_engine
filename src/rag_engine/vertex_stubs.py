"""Install minimal ``vertexai.language_models`` stubs when the real SDK is not present.

Allows ``from vertexai.language_models import GenerativeModel`` in application code
(literal path named in the assessment brief) while keeping CI and local runs free of
``google-cloud-aiplatform`` unless installed.
"""

from __future__ import annotations

import sys
import types

_STUB_FLAG = "_teleport_vertex_stub"


def ensure_vertexai_stub_modules() -> None:
    from rag_engine import gcp_mocks

    lm_existing = sys.modules.get("vertexai.language_models")
    if lm_existing is not None and getattr(lm_existing, _STUB_FLAG, False):
        return

    try:
        import vertexai.language_models as vlm  # type: ignore[import-not-found,import-untyped]

        if hasattr(vlm, "GenerativeModel"):
            return
    except ImportError:
        pass

    vertexai = types.ModuleType("vertexai")
    lm = types.ModuleType("vertexai.language_models")
    setattr(lm, _STUB_FLAG, True)
    vertexai.language_models = lm  # type: ignore[attr-defined]
    sys.modules["vertexai"] = vertexai
    sys.modules["vertexai.language_models"] = lm
    lm.TextEmbeddingModel = gcp_mocks.TextEmbeddingModel  # type: ignore[attr-defined]
    lm.TextEmbeddingInput = gcp_mocks.TextEmbeddingInput  # type: ignore[attr-defined]
    lm.Embedding = gcp_mocks.Embedding  # type: ignore[attr-defined]
    lm.GenerativeModel = gcp_mocks.GenerativeModel  # type: ignore[attr-defined]
    lm.GenerativeResponse = gcp_mocks.GenerativeResponse  # type: ignore[attr-defined]
