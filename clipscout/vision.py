"""The two things Gemini does for us, each with a strict JSON schema:

1. describe_segment(): watch a 5-minute piece and describe it every ~10 seconds  (indexing)
2. verify_clip():      look at one short clip and say if the searched thing is really there  (search)
"""
from clipscout import config
from clipscout.llm import gemini_json
from clipscout.schemas import Moment, VideoSource
from clipscout.video import ensure_uploaded, to_absolute, video_part

MOMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "visual": {"type": "string"},
                    "speech": {"type": "string"},
                    "on_screen_text": {"type": "string"},
                },
                "required": ["start", "end", "visual", "speech", "on_screen_text"],
            },
        }
    },
    "required": ["moments"],
}

DESCRIBE_PROMPT = """You are building a search index for this video clip ({length} seconds long).
Split it into consecutive moments of about {moment} seconds that cover the WHOLE clip with no gaps.
For each moment return:
- start, end: seconds from the START OF THIS CLIP.
- visual: concrete, searchable English: people (clothes, colours), objects, vehicles (type, colour),
  actions, place, scene or camera changes. Be specific. Never guess who a person is.
- speech: what is said, in the original language; if not English, add a short English gloss in [brackets].
  Empty string if nobody speaks.
- on_screen_text: readable text (slides, boards, signs, number plates, captions). Empty string if none.
Describe only what is really in the clip."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "reason": {"type": "string"},
        "best_second": {"type": "number"},
    },
    "required": ["present", "reason", "best_second"],
}

VERIFY_PROMPT = """A user is searching a video for: "{question}"
This clip is {length} seconds long. Check the frames and audio of THIS clip only.
present: true only if the clip really shows or says what the user is looking for.
reason: one short sentence about what you saw or heard that decides it.
best_second: seconds from the start of this clip where it is clearest (0 if not present)."""


def describe_segment(src: VideoSource, seg_start: float, seg_end: float) -> list[Moment]:
    length = seg_end - seg_start
    data = gemini_json(
        [video_part(src, seg_start, seg_end),
         {"type": "text", "text": DESCRIBE_PROMPT.format(length=int(length), moment=config.MOMENT_SEC)}],
        MOMENTS_SCHEMA,
    )
    moments = []
    for raw in data.get("moments", []):
        start = to_absolute(float(raw.get("start", 0) or 0), seg_start, length)
        end = to_absolute(float(raw.get("end", 0) or 0), seg_start, length)
        start = min(max(start, seg_start), seg_end)       # clamp into this segment
        end = min(max(end, start + 1), seg_end)
        m = Moment(start=round(start, 1), end=round(end, 1),
                   visual=raw.get("visual", "").strip(), speech=raw.get("speech", "").strip(),
                   on_screen_text=raw.get("on_screen_text", "").strip())
        if m.search_text():
            moments.append(m)
    return moments


def verify_clip(src: VideoSource, start: float, end: float, question: str) -> dict:
    """Send the real clip back to Gemini: is the thing the user asked for actually in it?"""
    ensure_uploaded(src)
    a = max(0.0, start - config.VERIFY_PAD_SEC)
    b = end + config.VERIFY_PAD_SEC
    if src.duration:
        b = min(b, src.duration)
    b = max(b, a + 2)
    out = gemini_json(
               [video_part(src, a, b, resolution="high"),   # short clip, so we can afford sharper frames
         {"type": "text", "text": VERIFY_PROMPT.format(question=question, length=int(b - a))}],
         VERIFY_SCHEMA,
    )
    best = to_absolute(float(out.get("best_second", 0) or 0), a, b - a)
    return {"present": bool(out.get("present")), "reason": str(out.get("reason", "")).strip(),
            "best_second": min(max(best, a), b)}
