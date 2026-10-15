"""Getting a video ready for Gemini: ids, duration, upload, and clipping. No OpenCV, no FFmpeg."""
import hashlib
import logging
import os
import re
import time

from clipscout.llm import gemini_client
from clipscout.schemas import VideoSource

log = logging.getLogger("clipscout.video")

_YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")


def youtube_id(url: str) -> str | None:
    m = _YT_ID.search(url)
    return m.group(1) if m else None


def file_fingerprint(path: str) -> str:
    """Stable id for a local file: hash of its size + first 1 MB (fast even for big files)."""
    h = hashlib.sha1(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:12]


def read_duration(path: str) -> float:
    """Video length from the container header (MP4/MOV), pure Python. 0.0 if unknown."""
    try:
        from mutagen import File as MutagenFile
        info = MutagenFile(path)
        if info is not None and info.info and getattr(info.info, "length", 0):
            return float(info.info.length)
    except Exception:
        pass
    return 0.0


def youtube_source(url: str, minutes: float, title: str = "") -> VideoSource:
    vid = youtube_id(url)
    if not vid:
        raise ValueError("That does not look like a YouTube video link.")
    return VideoSource(video_id=f"yt_{vid}", title=title or f"YouTube {vid}", kind="youtube",
                       url=f"https://www.youtube.com/watch?v={vid}", duration=minutes * 60)


def file_source(path: str, title: str = "", minutes: float = 0.0) -> VideoSource:
    duration = read_duration(path) or minutes * 60
    return VideoSource(video_id=f"f_{file_fingerprint(path)}", title=title or os.path.basename(path),
                       kind="file", local_path=path, duration=duration)


def ensure_uploaded(src: VideoSource, timeout_sec: int = 900) -> None:
    """Local files must live on Gemini's Files API. Uploads expire, so check and re-upload if needed."""
    if src.kind != "file":
        return
    client = gemini_client()
    if src.gemini_file_name:
        try:
            f = client.files.get(name=src.gemini_file_name)
            if f.state and f.state.name == "ACTIVE":
                return
        except Exception:
            log.info("Gemini copy of %s expired, uploading again", src.title)

    f = client.files.upload(file=src.local_path)
    waited = 0
    while not f.state or f.state.name != "ACTIVE":
        if f.state and f.state.name == "FAILED":
            raise RuntimeError("Gemini could not process this video file.")
        if waited >= timeout_sec:
            raise TimeoutError("Gemini took too long to process the upload.")
        time.sleep(5)
        waited += 5
        f = client.files.get(name=f.name)

    src.gemini_file_name, src.gemini_file_uri = f.name, f.uri
    src.mime_type = f.mime_type or "video/mp4"
    meta = getattr(f, "video_metadata", None)
    if not src.duration and isinstance(meta, dict):        # Gemini sometimes reports e.g. "734.2s"
        dur = str(meta.get("videoDuration") or meta.get("video_duration") or "")
        if dur.endswith("s"):
            src.duration = float(dur[:-1])


def video_part(src: VideoSource, start: float | None = None, end: float | None = None,
               resolution: str = "low") -> dict:
    """Gemini 'video' input, optionally clipped to [start, end] seconds (static 1-fps processing).

    Low resolution = ~70 tokens per frame instead of ~260: about 3x cheaper and faster, which keeps
    us under the free tier's tokens-per-minute limit. Plenty for describing scenes.
    """
    if src.kind == "youtube":
        part = {"type": "video", "uri": src.url, "resolution": resolution}
    else:
        part = {"type": "video", "uri": src.gemini_file_uri, "mime_type": src.mime_type or "video/mp4",
                "resolution": resolution}
    if start is not None and end is not None:
        part["processing"] = {"type": "static", "start_offset": f"{start:.0f}s", "end_offset": f"{end:.0f}s"}
    return part


def to_absolute(t: float, clip_start: float, clip_len: float) -> float:
    """Gemini may report a time relative to the clip or to the whole video. Return whole-video seconds.

    Relative times fall in [0, clip_len]; absolute ones in [clip_start, clip_start + clip_len].
    For every clip after the first those ranges do not overlap, so we can tell them apart.
    """
    if clip_start > 0 and clip_start - 2 <= t <= clip_start + clip_len + 2:
        return t
    return clip_start + t

