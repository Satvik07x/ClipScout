from clipscout.indexer import plan_segments
from clipscout.schemas import Clip, Moment, fmt_time
from clipscout.video import to_absolute, video_part, youtube_id, youtube_source


def test_fmt_time():
    assert fmt_time(0) == "00:00"
    assert fmt_time(75.9) == "01:15"
    assert fmt_time(3725) == "1:02:05"


def test_youtube_id_variants():
    for url in ["https://www.youtube.com/watch?v=9hE5-98ZeCg&t=10s", "https://youtu.be/9hE5-98ZeCg",
                "https://www.youtube.com/shorts/9hE5-98ZeCg", "https://www.youtube.com/embed/9hE5-98ZeCg"]:
        assert youtube_id(url) == "9hE5-98ZeCg"
    assert youtube_id("https://example.com/video") is None


def test_to_absolute_relative_and_absolute():
    assert to_absolute(12, clip_start=0, clip_len=300) == 12           # first piece: always relative
    assert to_absolute(12, clip_start=600, clip_len=300) == 612        # relative -> shifted
    assert to_absolute(612, clip_start=600, clip_len=300) == 612       # already absolute -> kept


def test_plan_segments():
    assert plan_segments(700, size=300) == [(0, 300), (300, 600), (600, 700)]
    assert plan_segments(600, size=300) == [(0, 300), (300, 600)]
    assert plan_segments(0, size=300) == []


def test_old_library_files_are_upgraded():
    from clipscout.schemas import VideoSource
    done = VideoSource.model_validate({"video_id": "v", "title": "t", "kind": "youtube", "duration": 700,
                                        "indexed_until": 700})
    half = VideoSource.model_validate({"video_id": "v", "title": "t", "kind": "youtube", "duration": 700,
                                        "indexed_until": 300})
    assert done.is_indexed and done.segments_total == 3
    assert not half.is_indexed and half.segments_done == [0.0]


def test_video_part_clipping():
    src = youtube_source("https://youtu.be/9hE5-98ZeCg", minutes=10)
    part = video_part(src, 300, 600)
    assert part["uri"].endswith("9hE5-98ZeCg")
    assert part["processing"] == {"type": "static", "start_offset": "300s", "end_offset": "600s"}


def test_moment_text_and_clip_overlap():
    m = Moment(start=0, end=10, visual="man on red bike", speech="chalo", on_screen_text="MP04")
    assert m.search_text() == "man on red bike | Speech: chalo | On screen: MP04"
    c = Clip(video_id="v", start=100, end=110, best_start=100)
    assert c.overlaps(108, 120) and c.overlaps(112, 120, tolerance=5) and not c.overlaps(120, 130)

