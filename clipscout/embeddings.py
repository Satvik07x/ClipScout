"""Text -> vector with Gemini's embedding API, wrapped as a LangChain Embeddings class.

Why an API instead of a local model: no PyTorch to install or load (the app starts in a
second), and the model is multilingual, so "laal gaadi" and "red car" land close together.

Documents and queries are embedded with different task types (RETRIEVAL_DOCUMENT vs
RETRIEVAL_QUERY); the model is trained to match a short question to a longer description.
"""
import math
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from clipscout import config
from clipscout.llm import gemini_client, with_retry

BATCH = 50  # texts per request


def _normalize(v: list[float]) -> list[float]:
    """Vectors shorter than 3072 dims come back un-normalised; cosine search needs unit length."""
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _embed(texts: list[str], task: str) -> list[list[float]]:
    from google.genai import types
    cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=config.EMBED_DIM)

    def call(model: str):
        return gemini_client().models.embed_content(model=model, contents=texts, config=cfg)

    result = with_retry(call, [config.EMBED_MODEL])
    return [_normalize(list(e.values)) for e in result.embeddings]


@lru_cache(maxsize=512)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    return tuple(_embed([text], "RETRIEVAL_QUERY")[0])   # repeated searches skip the API call


class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(_embed(texts[i:i + BATCH], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return list(_embed_query_cached(text.strip()))
