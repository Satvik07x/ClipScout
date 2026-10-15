"""Who may see which video, in the public deployment.

Every browser gets a random visitor id (made by the web app). A video can have several owners:
if two visitors add the same YouTube link, Gemini watches it only once and both share the index.
This is scoping, not security: the id is a random string, not a login.

Kept in its own small file (data/owners.json) so it never races with the indexer, which keeps
re-saving the video's own JSON while it works.
"""
import json
import os
import threading

from clipscout import config

_lock = threading.Lock()


def _path() -> str:
    return os.path.join(config.DATA_DIR, "owners.json")


def _read() -> dict[str, list[str]]:
    try:
        with open(_path(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write(data: dict[str, list[str]]) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp = _path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, _path())


def add(video_id: str, visitor: str) -> None:
    with _lock:
        data = _read()
        if visitor not in data.setdefault(video_id, []):
            data[video_id].append(visitor)
            _write(data)


def has(video_id: str, visitor: str) -> bool:
    with _lock:
        return visitor in _read().get(video_id, [])


def of(visitor: str) -> list[str]:
    """Video ids this visitor keeps."""
    with _lock:
        return [vid for vid, people in _read().items() if visitor in people]


def remove(video_id: str, visitor: str) -> bool:
    """Drop this visitor's claim. True when nobody keeps the video any more (so it can be deleted)."""
    with _lock:
        data = _read()
        people = [p for p in data.get(video_id, []) if p != visitor]
        if people:
            data[video_id] = people
        else:
            data.pop(video_id, None)
        _write(data)
        return not people
