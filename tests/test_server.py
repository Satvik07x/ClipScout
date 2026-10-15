"""API layer with fake models: add a video, watch it index, search over SSE, delete."""
import json

import pytest
from fastapi.testclient import TestClient

from clipscout import agent, config, vision
from tests.test_agent import FakeLLM


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(vision, "gemini_json", lambda parts, schema, **kw: {"moments": [
        {"start": 0, "end": 10, "visual": "a man rides a red motorcycle through the gate", "speech": "", "on_screen_text": ""},
        {"start": 10, "end": 20, "visual": "a delivery van stops", "speech": "", "on_screen_text": "MP04"}]})
    monkeypatch.setattr(agent, "groq", lambda: FakeLLM())
    monkeypatch.setattr(agent, "verify_clip", lambda src, a, b, q: {"present": True, "reason": "bike visible", "best_second": a + 1})
    import server
    server.jobs.clear()
    with TestClient(server.app) as c:
        yield c


def sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_add_index_search_delete(client):
    r = client.post("/api/videos/youtube", json={"url": "https://youtu.be/9hE5-98ZeCg", "minutes": 0.5, "title": "Gate"})
    assert r.status_code == 202
    vid = r.json()["video_id"]

    v = client.get(f"/api/videos/{vid}").json()            # background task already ran in TestClient
    assert v["status"] == "ready" and v["moments"] == 2 and v["youtube_id"] == "9hE5-98ZeCg"

    events = sse(client.get("/api/search", params={"q": "red bike kab aaya"}).text)
    assert events[-1]["type"] == "done" and events[-1]["found"]
    assert events[-1]["clips"][0]["verified"] is True

    assert client.delete(f"/api/videos/{vid}").status_code == 204
    assert client.get("/api/videos").json() == []


def test_validation_errors(client):
    assert client.post("/api/videos/youtube", json={"url": "https://example.com", "minutes": 3}).status_code == 422
    assert client.post("/api/videos/youtube", json={"url": "https://youtu.be/9hE5-98ZeCg", "minutes": 0}).status_code == 422
    assert client.get("/api/videos/nope").status_code == 404


def test_upload_needs_length_when_unreadable(client):
    r = client.post("/api/videos/upload", files={"file": ("clip.mp4", b"not really a video", "video/mp4")})
    assert r.status_code == 422
    r = client.post("/api/videos/upload", files={"file": ("clip.mp4", b"not really a video", "video/mp4")},
                    data={"minutes": "0.3", "title": "My clip"})
    assert r.status_code == 202 and r.json()["file_url"].endswith("/file")
