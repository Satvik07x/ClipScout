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


# ---------------------------------------------------------------- public mode

A = {"X-Visitor": "visitor-aaaaaaaa"}
B = {"X-Visitor": "visitor-bbbbbbbb"}


def yt(video: str, minutes: float = 0.5) -> dict:
    return {"url": f"https://youtu.be/{video}", "minutes": minutes}


@pytest.fixture
def public(client, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_MODE", True)
    monkeypatch.setattr(config, "ADDS_PER_IP_PER_DAY", 100)   # every test request shares one fake IP
    return client


def test_public_visitors_only_see_their_own_videos(public):
    vid = public.post("/api/videos/youtube", json=yt("aaaaaaaaaaa"), headers=A).json()["video_id"]
    assert [v["video_id"] for v in public.get("/api/videos", headers=A).json()] == [vid]
    assert public.get("/api/videos", headers=B).json() == []
    assert public.get(f"/api/videos/{vid}", headers=B).status_code == 404
    assert public.delete(f"/api/videos/{vid}", headers=B).status_code == 404

    mine = sse(public.get("/api/search", params={"q": "red bike", "video_id": vid, "visitor": A["X-Visitor"]}).text)
    assert mine[-1]["type"] == "done" and mine[-1]["found"]
    theirs = sse(public.get("/api/search", params={"q": "red bike", "video_id": vid, "visitor": B["X-Visitor"]}).text)
    assert theirs[-1]["type"] == "error"


def test_public_needs_visitor_id_and_a_video_to_search(public):
    assert public.get("/api/videos").status_code == 400
    assert public.get("/api/videos", headers={"X-Visitor": "x"}).status_code == 400
    events = sse(public.get("/api/search", params={"q": "red bike", "visitor": A["X-Visitor"]}).text)
    assert events[-1]["type"] == "error"                      # public search must name one of your videos


def test_public_same_video_is_shared_and_costs_no_extra_quota(public):
    from clipscout.limits import DAY, limiter
    vid = public.post("/api/videos/youtube", json=yt("aaaaaaaaaaa"), headers=A).json()["video_id"]
    assert limiter.used("global", DAY) == 1

    again = public.post("/api/videos/youtube", json=yt("aaaaaaaaaaa"), headers=B)
    assert again.status_code == 202 and again.json()["status"] == "ready"
    assert limiter.used("global", DAY) == 1                    # B reused A's index: Gemini did not watch it again

    public.delete(f"/api/videos/{vid}", headers=A)             # A leaves, B keeps it
    assert [v["video_id"] for v in public.get("/api/videos", headers=B).json()] == [vid]
    public.delete(f"/api/videos/{vid}", headers=B)             # last owner leaves: really deleted
    assert public.get("/api/videos", headers=B).json() == []
    assert public.get(f"/api/videos/{vid}", headers=A).status_code == 404


def test_public_limits(public, monkeypatch):
    assert public.get("/api/config").json()["public"] is True
    assert public.post("/api/videos/youtube", json=yt("aaaaaaaaaaa", 6), headers=A).status_code == 422   # > 5 minutes

    assert public.post("/api/videos/youtube", json=yt("aaaaaaaaaaa"), headers=A).status_code == 202
    assert public.post("/api/videos/youtube", json=yt("bbbbbbbbbbb"), headers=A).status_code == 202
    assert public.post("/api/videos/youtube", json=yt("ccccccccccc"), headers=A).status_code == 429       # 2 per visitor

    monkeypatch.setattr(config, "DAILY_INDEX_LIMIT", 2)                                                   # budget already spent
    assert public.post("/api/videos/youtube", json=yt("ddddddddddd"), headers=B).status_code == 429
    assert public.get("/api/config").json()["daily_left"] == 0

    upload = public.post("/api/videos/upload", files={"file": ("clip.mp4", b"x", "video/mp4")}, headers=A)
    assert upload.status_code == 403


def test_public_search_rate_limit(public, monkeypatch):
    monkeypatch.setattr(config, "SEARCHES_PER_HOUR", 1)
    vid = public.post("/api/videos/youtube", json=yt("aaaaaaaaaaa"), headers=A).json()["video_id"]
    params = {"q": "red bike", "video_id": vid, "visitor": A["X-Visitor"]}
    assert sse(public.get("/api/search", params=params).text)[-1]["type"] == "done"
    second = sse(public.get("/api/search", params=params).text)
    assert second[-1]["type"] == "error" and "limit" in second[-1]["message"].lower()
