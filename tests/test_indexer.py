"""Indexing with a fake Gemini: segment planning, time normalisation, checkpoint + resume, cache."""
import pytest

from clipscout import config, indexer, library, search, vision
from clipscout.video import youtube_source


@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setattr(config, "SEGMENT_SEC", 60)
    calls = []

    def fake(parts, schema, **kw):
        proc = parts[0]["processing"]
        a, b = float(proc["start_offset"][:-1]), float(proc["end_offset"][:-1])
        calls.append((a, b))
        if fake.fail_at == a:
            raise RuntimeError("quota exhausted")
        # Gemini answers in clip-relative seconds
        return {"moments": [{"start": 0, "end": 10, "visual": f"scene at {int(a)}", "speech": "", "on_screen_text": ""},
                            {"start": 10, "end": 20, "visual": "", "speech": "", "on_screen_text": ""}]}  # empty -> dropped
    fake.fail_at = None
    monkeypatch.setattr(vision, "gemini_json", fake)
    return fake, calls


def test_index_resume_and_cache(fake_gemini):
    fake, calls = fake_gemini
    src = youtube_source("https://youtu.be/9hE5-98ZeCg", minutes=2.5)          # 150 s -> pieces 0-60, 60-120, 120-150

    fake.fail_at = 120.0
    with pytest.raises(RuntimeError):
        indexer.index_video(src)
    partial = library.load(src.video_id)
    assert sorted(partial.segments_done) == [0, 60] and not partial.is_indexed   # finished pieces saved

    fake.fail_at = None
    done = indexer.index_video(youtube_source("https://youtu.be/9hE5-98ZeCg", minutes=2.5))
    assert sorted(calls) == [(0, 60), (60, 120), (120, 150), (120, 150)]     # only the failed piece re-ran
    assert [m.start for m in done.moments] == [0, 60, 120]                   # relative -> absolute
    assert done.is_indexed and search.count(src.video_id) == 3

    indexer.index_video(youtube_source("https://youtu.be/9hE5-98ZeCg", minutes=2.5))
    assert len(calls) == 4                                                   # cached: no more Gemini calls
