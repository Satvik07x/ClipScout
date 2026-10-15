"""Saved videos and their indexes: one JSON file per video in data/library/.

This is the "index once" cache: a video is watched by Gemini a single time, and every search
afterwards only reads this JSON (plus the vector store built from it).
"""
import os

from clipscout import config
from clipscout.schemas import VideoSource


def _path(video_id: str) -> str:
    return os.path.join(config.LIBRARY_DIR, f"{video_id}.json")


def save(src: VideoSource) -> None:
    os.makedirs(config.LIBRARY_DIR, exist_ok=True)
    tmp = _path(src.video_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src.model_dump_json(indent=2))
    os.replace(tmp, _path(src.video_id))   # atomic: a crash never leaves a half-written file


def load(video_id: str) -> VideoSource | None:
    p = _path(video_id)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return VideoSource.model_validate_json(f.read())


def all_videos(only_indexed: bool = True) -> list[VideoSource]:
    if not os.path.isdir(config.LIBRARY_DIR):
        return []
    out = []
    for name in sorted(os.listdir(config.LIBRARY_DIR)):
        if name.endswith(".json"):
            src = load(name[:-5])
            if src and (src.is_indexed or not only_indexed):
                out.append(src)
    return out


def delete(video_id: str) -> None:
    if os.path.exists(_path(video_id)):
        os.remove(_path(video_id))
