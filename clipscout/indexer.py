"""Indexing pipeline: VideoSource -> Gemini watches 5-minute pieces in parallel -> moments
-> library JSON + vector store.

Resumable: every finished piece is saved immediately (`segments_done`). If the free-tier quota
runs out or the app stops halfway, the next run only watches the pieces that are missing.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from clipscout import config, library, search
from clipscout.schemas import VideoSource, fmt_time
from clipscout.video import ensure_uploaded
from clipscout.vision import describe_segment

log = logging.getLogger("clipscout.indexer")

Progress = Callable[[float, str], None]   # (fraction 0..1, message)


def plan_segments(duration: float, size: float | None = None) -> list[tuple[float, float]]:
    size = size or config.SEGMENT_SEC
    out, t = [], 0.0
    while t < duration - 0.5:
        out.append((t, min(t + size, duration)))
        t += size
    return out


def index_video(src: VideoSource, progress: Progress | None = None, force: bool = False) -> VideoSource:
    report = progress or (lambda f, m: None)
    saved = library.load(src.video_id)
    if saved and not force:
        if saved.is_indexed:
            if search.count(saved.video_id) == 0:    # e.g. data/chroma was deleted
                search.add_video(saved)
            report(1.0, "Already indexed")
            return saved
        src = saved                                  # resume a half-finished index
    if force:
        src.moments, src.segments_done = [], []
    if src.duration <= 0:
        raise ValueError("Video length is unknown. Enter it in minutes.")

    segments = plan_segments(src.duration)
    src.segments_total = len(segments)
    todo = [(a, b) for a, b in segments if a not in src.segments_done]
    library.save(src)
    report(src.progress, "Preparing video")
    ensure_uploaded(src)

    lock, errors = threading.Lock(), []
    t0 = time.perf_counter()
    report(src.progress, f"Gemini is watching {len(todo)} piece(s)")
    with ThreadPoolExecutor(max_workers=config.INDEX_WORKERS) as pool:
        jobs = {pool.submit(describe_segment, src, a, b): (a, b) for a, b in todo}
        for job in as_completed(jobs):
            a, b = jobs[job]
            try:
                moments = job.result()
            except Exception as e:                   # keep the finished pieces, report the first failure
                errors.append(e)
                log.warning("piece %s-%s failed: %s", fmt_time(a), fmt_time(b), e)
                continue
            with lock:
                src.moments.extend(moments)
                src.segments_done.append(a)
                library.save(src)                    # checkpoint after every piece
            report(src.progress, f"Watched {fmt_time(a)}–{fmt_time(b)}")
    if errors:
        raise errors[0]

    src.moments.sort(key=lambda m: m.start)
    library.save(src)
    report(0.99, "Building the search index")
    search.add_video(src)
    log.info("indexed %s: %d moments in %.1fs", src.video_id, len(src.moments), time.perf_counter() - t0)
    report(1.0, f"Ready: {len(src.moments)} moments")
    return src


def remove(video_id: str) -> None:
    search.remove_video(video_id)
    library.delete(video_id)


def sync_vector_store() -> int:
    """Make sure every finished video is in the vector store (e.g. after an embedding-model change).
    Only re-embeds text; Gemini never re-watches a video."""
    added = 0
    for src in library.all_videos():
        if search.count(src.video_id) == 0:
            added += search.add_video(src)
    return added
