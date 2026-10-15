import pytest
from langchain_core.documents import Document

from clipscout import search
from clipscout.schemas import Moment, VideoSource


def doc(vid, start, text="x"):
    return Document(page_content=text, metadata={"video_id": vid, "title": vid, "start": start, "end": start + 10})


def test_rrf_rewards_agreement():
    a, b, c = doc("v", 0), doc("v", 10), doc("v", 20)
    fused = search.rrf([[a, b, c], [b, c]])
    order = [d.metadata["start"] for d, _ in fused]
    assert order == [10, 20, 0]   # a is 1st in one list only; b is high in both -> b wins
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_merge_grows_strong_neighbours_and_drops_weak_ones():
    scored = [(doc("v", 0), 0.1), (doc("v", 10), 0.5), (doc("v", 20), 0.4), (doc("v", 100), 0.3), (doc("w", 10), 0.2)]
    clips = search.merge_into_clips(scored, gap=5, min_ratio=0.6)
    assert [(c.video_id, c.start, c.end) for c in clips] == [("v", 10, 30), ("v", 100, 110), ("w", 10, 20)]
    assert clips[0].best_start == 10 and clips[0].score == 0.5   # v@0 (0.1) was too weak to join


def test_merge_respects_max_clip_length():
    scored = [(doc("v", t), 1.0) for t in range(0, 200, 10)]
    clips = search.merge_into_clips(scored, gap=5, min_ratio=0.6, max_len=60)
    assert all(c.end - c.start <= 60 for c in clips)


def sample_video() -> VideoSource:
    return VideoSource(video_id="cam1", title="Gate camera", kind="youtube", url="u", duration=60, segments_total=1, segments_done=[0.0],
                       moments=[Moment(start=0, end=10, visual="empty street, parked white car"),
                                Moment(start=10, end=20, visual="a man rides a red motorcycle through the gate"),
                                Moment(start=20, end=30, visual="the man parks the red motorcycle near the wall"),
                                Moment(start=40, end=50, visual="a delivery van stops", on_screen_text="MP04 AB 1234")])


def test_hybrid_search_end_to_end():
    assert search.add_video(sample_video()) == 4
    clips = search.hybrid_search("red motorcycle")
    assert clips[0].start == 10 and clips[0].end == 30          # two neighbouring moments merged
    # exact token only BM25 can really nail
    assert search.hybrid_search("MP04")[0].start == 40
    # filter by video works, re-adding replaces instead of duplicating
    assert search.add_video(sample_video()) == 4 and search.count("cam1") == 4
    assert search.hybrid_search("red motorcycle", video_id="other") == []


def test_remove_video():
    search.add_video(sample_video())
    search.remove_video("cam1")
    assert search.count("cam1") == 0 and search.hybrid_search("van") == []
