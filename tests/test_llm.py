"""gemini_json: request shape, 429 backoff, model fallback, broken-JSON retry."""
import pytest

from clipscout import config, llm


class FakeInteractions:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return type("Interaction", (), {"output_text": step})()


def fake_client(monkeypatch, script):
    inter = FakeInteractions(script)
    monkeypatch.setattr(llm, "gemini_client", lambda: type("C", (), {"interactions": inter})())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "GEMINI_MODELS", ["model-a", "model-b"])
    return inter


def test_request_shape(monkeypatch):
    inter = fake_client(monkeypatch, ['{"ok": true}'])
    assert llm.gemini_json([{"type": "text", "text": "hi"}], {"type": "object"}) == {"ok": True}
    req = inter.requests[0]
    assert req["model"] == "model-a"
    assert req["response_format"] == {"type": "text", "mime_type": "application/json", "schema": {"type": "object"}}


def test_busy_model_hands_over_to_next_model(monkeypatch):
    inter = fake_client(monkeypatch, [Exception("429 RESOURCE_EXHAUSTED"), '{"n": 1}'])
    assert llm.gemini_json([], {}) == {"n": 1}
    assert [r["model"] for r in inter.requests] == ["model-a", "model-b"]   # no waiting on a busy model


def test_waits_only_after_every_model_failed(monkeypatch):
    waits = []
    inter = fake_client(monkeypatch, [Exception("503 UNAVAILABLE"), Exception("429"), '{"n": 1}'])
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    assert llm.gemini_json([], {}) == {"n": 1}
    assert [r["model"] for r in inter.requests] == ["model-a", "model-b", "model-a"] and waits == [5]


def test_gives_up_with_a_clear_message(monkeypatch):
    fake_client(monkeypatch, [Exception("429")] * 8)
    with pytest.raises(RuntimeError, match="busy or out of free quota"):
        llm.gemini_json([], {})


def test_falls_back_to_next_model(monkeypatch):
    inter = fake_client(monkeypatch, [Exception("404 model not found"), '{"n": 2}'])
    assert llm.gemini_json([], {}) == {"n": 2}
    assert [r["model"] for r in inter.requests] == ["model-a", "model-b"]


def test_retries_broken_json(monkeypatch):
    fake_client(monkeypatch, ['{"n": ', '{"n": 3}'])
    assert llm.gemini_json([], {}) == {"n": 3}


def test_other_errors_raise(monkeypatch):
    fake_client(monkeypatch, [ValueError("API key not valid")])
    with pytest.raises(ValueError):
        llm.gemini_json([], {})


def test_embeddings_are_normalised_and_query_cached(monkeypatch):
    from clipscout import embeddings
    calls = []

    class Emb:
        def __init__(self, v):
            self.values = v

    class Models:
        def embed_content(self, model, contents, config):
            calls.append((config.task_type, list(contents)))
            return type("R", (), {"embeddings": [Emb([3.0, 4.0]) for _ in contents]})()

    monkeypatch.setattr(embeddings, "gemini_client", lambda: type("C", (), {"models": Models()})())
    embeddings._embed_query_cached.cache_clear()
    e = embeddings.GeminiEmbeddings()
    assert e.embed_documents(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    e.embed_query("red car")
    e.embed_query("red car")
    assert [c[0] for c in calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]   # second query came from cache
