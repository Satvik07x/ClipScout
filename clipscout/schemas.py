"""Typed data that flows through the system. If you understand these three classes,
you understand the project:

    VideoSource  --(Gemini)-->  [Moment, Moment, ...]  --(search)-->  [Clip, Clip, ...]
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def fmt_time(t: float) -> str:
    """Seconds -> 'MM:SS' (or 'H:MM:SS')."""
    t = int(max(0, t))
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Moment(BaseModel):
    """~10 seconds of video, described by Gemini. This is the unit we search over."""
    start: float
    end: float
    visual: str = ""
    speech: str = ""
    on_screen_text: str = ""

    def search_text(self) -> str:
        parts = [self.visual]
        if self.speech:
            parts.append(f"Speech: {self.speech}")
        if self.on_screen_text:
            parts.append(f"On screen: {self.on_screen_text}")
        return " | ".join(p for p in parts if p)


class VideoSource(BaseModel):
    """One video in the library, plus its index."""
    video_id: str
    title: str
    kind: Literal["youtube", "file"]
    url: str = ""                 # YouTube link
    local_path: str = ""          # uploaded file on disk
    duration: float = 0.0         # seconds
    gemini_file_name: str = ""    # Files API handle ("files/abc"); uploads expire, re-uploaded on demand
    gemini_file_uri: str = ""
    mime_type: str = ""
    moments: list[Moment] = Field(default_factory=list)
    segments_done: list[float] = Field(default_factory=list)  # start second of each finished piece
    segments_total: int = 0                                   # -> indexing resumes after a crash

    @model_validator(mode="before")
    @classmethod
    def _upgrade_old_files(cls, data):
        """Older library files stored progress as `indexed_until`; convert instead of re-watching the video."""
        if isinstance(data, dict) and "indexed_until" in data and not data.get("segments_done"):
            from clipscout.config import SEGMENT_SEC
            until = float(data.pop("indexed_until") or 0)
            dur = float(data.get("duration") or 0)
            starts = [float(s) for s in range(0, int(dur + 0.5), SEGMENT_SEC)] if dur else []
            data["segments_total"] = len(starts)
            data["segments_done"] = [s for s in starts if min(s + SEGMENT_SEC, dur) <= until + 0.5]
        return data

    @property
    def is_indexed(self) -> bool:
        return self.segments_total > 0 and len(self.segments_done) >= self.segments_total

    @property
    def progress(self) -> float:
        return len(self.segments_done) / self.segments_total if self.segments_total else 0.0

    @property
    def playable(self) -> str:
        return self.url if self.kind == "youtube" else self.local_path

    @property
    def youtube_id(self) -> str:
        return self.video_id[3:] if self.kind == "youtube" else ""


class Clip(BaseModel):
    """A search result: one or more neighbouring moments merged together."""
    video_id: str
    title: str = ""
    start: float
    end: float
    best_start: float              # where to start playback
    score: float = 0.0
    text: str = ""
    verified: bool | None = None   # None = not checked, True/False = Gemini's verdict
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.title} @ {fmt_time(self.best_start)}"

    def overlaps(self, a: float, b: float, tolerance: float = 0.0) -> bool:
        return self.start <= b + tolerance and self.end >= a - tolerance

    def lands_in(self, a: float, b: float, tolerance: float = 0.0) -> bool:
        """Does playback start inside [a, b] (± tolerance)? This is what the user actually experiences,
        so eval uses it (a huge clip that merely overlaps the answer should not count as a hit)."""
        return a - tolerance <= self.best_start <= b + tolerance
