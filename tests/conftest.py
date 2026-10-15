"""Test setup: no API keys, no internet. Real Chroma + BM25, a tiny fake embedding model,
and fake Groq/Gemini where needed."""
import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings

from clipscout import config, search


class BagOfWordsEmbeddings(Embeddings):
    """Deterministic stand-in for MiniLM: hashed bag of words, L2-normalised."""
    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIBRARY_DIR", str(tmp_path / "library"))
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION", f"test_{tmp_path.name}")
    monkeypatch.setattr(search, "get_embeddings", lambda: BagOfWordsEmbeddings())
    search.reset()
    yield
    search.reset()
