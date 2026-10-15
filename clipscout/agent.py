"""The search agent, built with LangGraph.

    START ─► rewrite ─► retrieve ─► grade ─► verify ─► answer ─► END
                ▲           │          │        │
                │      nothing      none     Gemini: "not
                │       found      match    really there"
                └───────────┴──────────┴────────┘  (one retry with a better query)
                                                └─► not_found ─► END

rewrite   Groq turns "red bike wala kab aaya?" into "person on a red motorcycle arrives"
retrieve  hybrid search (vector + BM25 + RRF) over the moment index
grade     Groq reads the clip descriptions and drops the ones that don't match   (cheap, text only)
verify    Gemini looks at the ACTUAL frames of the top 2 clips, in parallel: is it really there?
          (the expensive step, so it runs last and only on clips that survived grading)
answer    Groq writes a 1-3 line answer with timestamps, from verified clips only

Every step can be switched off with Settings, so eval/run_eval.py can measure what each one adds.
stream() yields each step as it finishes, so the UI shows results before verification is done.
"""
import operator
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Annotated, Iterator, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from clipscout import config, library, search
from clipscout.llm import groq
from clipscout.schemas import Clip, fmt_time
from clipscout.vision import verify_clip


@dataclass(frozen=True)
class Settings:
    rewrite: bool = True
    keyword: bool = True    # BM25 next to vector search
    grade: bool = True
    verify: bool = True
    answer: bool = True


class State(TypedDict, total=False):
    question: str
    video_id: str | None
    settings: Settings
    query: str
    attempts: int
    candidates: list[Clip]
    kept: list[Clip]
    checked: list[Clip]   # clips Gemini looked at, with verified=True/False (for the UI)
    final: list[Clip]
    answer: str
    found: bool
    trace: Annotated[list[str], operator.add]   # each node appends a line; shown in the UI


# ---------------------------------------------------------------- prompts + structured outputs

class RewriteOut(BaseModel):
    search_query: str = Field(description="Short English description of what would be seen, heard or written")


class GradeOut(BaseModel):
    matching_ids: list[int] = Field(description="ids of clips whose description matches what the user wants")


REWRITE_PROMPT = """Rewrite the user's question as a short English search query for a video index.
The index describes what is VISIBLE (people, clothes, colours, objects, actions), what is SAID, and TEXT on screen.
- Translate Hindi/Hinglish to English.
- Drop question words ("when", "kab", "where", "find", "show me", "video").
- Keep names, numbers and technical terms exactly.
{feedback}
Question: {question}"""

GRADE_PROMPT = """A user is searching a video for: "{question}"
Below are candidate clips with descriptions. Return the ids of the clips that clearly match.
Return an empty list if none do.

{clips}"""

ANSWER_PROMPT = """Answer the user's question about their video in 1-3 sentences using ONLY the clips below.
Always cite timestamps as MM:SS. Reply in the same language and style as the question (Hinglish is fine).

Question: {question}

Clips:
{clips}"""


def ask(schema: type[BaseModel], prompt: str) -> BaseModel:
    """Groq call that must return `schema`. JSON-schema mode first; tool calling as a fallback."""
    try:
        return groq().with_structured_output(schema, method="json_schema").invoke(prompt)
    except Exception:
        return groq().with_structured_output(schema).invoke(prompt)


def _listing(clips: list[Clip], with_ids: bool = False) -> str:
    rows = []
    for i, c in enumerate(clips):
        head = f"[{i}] " if with_ids else "- "
        extra = f" (verified: {c.reason})" if c.verified and c.reason else ""
        rows.append(f"{head}{fmt_time(c.start)}-{fmt_time(c.end)} [{c.title}]: {c.text[:400]}{extra}")
    return "\n".join(rows)


# ---------------------------------------------------------------- nodes

def rewrite(state: State) -> dict:
    is_retry = bool(state.get("query"))
    attempts = state.get("attempts", 0) + (1 if is_retry else 0)
    feedback = ""
    if is_retry:
        feedback = (f'The previous query "{state["query"]}" found nothing that matched. Use different words: '
                    "synonyms, a broader visual description, or what someone would say out loud.")
    out = ask(RewriteOut, REWRITE_PROMPT.format(question=state["question"], feedback=feedback))
    return {"query": out.search_query, "attempts": attempts,
            "trace": [f'{"retry → " if is_retry else ""}rewrite: "{out.search_query}"']}


def retrieve(state: State) -> dict:
    query = state.get("query") or state["question"]
    clips = search.hybrid_search(query, top_k=config.TOP_K, video_id=state.get("video_id"),
                                 use_keyword=state["settings"].keyword)
    shown = ", ".join(c.label for c in clips[:3]) or "none"
    return {"query": query, "candidates": clips, "trace": [f"retrieve: {len(clips)} clips ({shown})"]}


def grade(state: State) -> dict:
    clips = state["candidates"]
    out = ask(GradeOut, GRADE_PROMPT.format(question=state["question"], clips=_listing(clips, with_ids=True)))
    kept = [clips[i] for i in dict.fromkeys(out.matching_ids) if 0 <= i < len(clips)]
    kept.sort(key=lambda c: c.score, reverse=True)
    return {"kept": kept, "trace": [f"grade: {len(kept)}/{len(clips)} descriptions match"]}


def _check(clip: Clip, question: str) -> Clip:
    """One Gemini look at one clip. Returns the clip with verified=True/False (None if the check failed)."""
    src = library.load(clip.video_id)
    if src is None:
        return clip
    try:
        v = verify_clip(src, clip.start, clip.end, question)
    except Exception as e:              # one failed check must not kill the whole search
        return clip.model_copy(update={"verified": None, "reason": f"check failed ({type(e).__name__})"})
    update = {"verified": v["present"], "reason": v["reason"]}
    if v["present"]:
        update["best_start"] = v["best_second"]
    return clip.model_copy(update=update)


def verify(state: State) -> dict:
    pool = (state["kept"] if state["settings"].grade else state["candidates"])[: config.VERIFY_TOP_N]
    if not pool:
        return {"kept": [], "checked": [], "trace": ["verify: nothing to check"]}
    with ThreadPoolExecutor(max_workers=len(pool)) as ex:      # the slow calls run side by side
        checked = list(ex.map(lambda c: _check(c, state["question"]), pool))
    confirmed = [c for c in checked if c.verified]
    notes = [f"{c.label}: {'yes' if c.verified else 'no'} ({c.reason})" for c in checked]
    return {"kept": confirmed, "checked": checked, "trace": ["verify on real frames: " + " | ".join(notes)]}


def answer(state: State) -> dict:
    s = state["settings"]
    clips = state["kept"] if (s.grade or s.verify) else state["candidates"]
    if not s.answer:
        return {"final": clips, "found": bool(clips), "answer": "", "trace": ["answer: skipped"]}
    msg = groq().invoke(ANSWER_PROMPT.format(question=state["question"], clips=_listing(clips)))
    return {"final": clips, "found": True, "answer": msg.content.strip(), "trace": ["answer: written from the clips"]}


def not_found(state: State) -> dict:
    guesses = [c.model_copy(update={"verified": False}) for c in state.get("candidates", [])]
    return {"final": guesses, "found": False,
            "answer": "Confident match nahi mila. Neeche sabse kareeb ke results hain (unverified).",
            "trace": ["not found: returning best unverified guesses"]}


# ---------------------------------------------------------------- routing

def _retry_or_stop(state: State) -> str:
    if state["settings"].rewrite and state.get("attempts", 0) < config.MAX_RETRIES:
        return "rewrite"
    return "not_found"


def after_start(state: State) -> str:
    return "rewrite" if state["settings"].rewrite else "retrieve"


def after_retrieve(state: State) -> str:
    s = state["settings"]
    if not state.get("candidates"):
        return _retry_or_stop(state)
    return "grade" if s.grade else "verify" if s.verify else "answer"


def after_grade(state: State) -> str:
    if not state.get("kept"):
        return _retry_or_stop(state)
    return "verify" if state["settings"].verify else "answer"


def after_verify(state: State) -> str:
    return "answer" if state.get("kept") else _retry_or_stop(state)


def build_graph():
    g = StateGraph(State)
    for name, fn in [("rewrite", rewrite), ("retrieve", retrieve), ("grade", grade),
                     ("verify", verify), ("answer", answer), ("not_found", not_found)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, after_start, ["rewrite", "retrieve"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["grade", "verify", "answer", "rewrite", "not_found"])
    g.add_conditional_edges("grade", after_grade, ["verify", "answer", "rewrite", "not_found"])
    g.add_conditional_edges("verify", after_verify, ["answer", "rewrite", "not_found"])
    g.add_edge("answer", END)
    g.add_edge("not_found", END)
    return g.compile()


_graph = None


@dataclass
class SearchResult:
    question: str
    query: str
    answer: str
    clips: list[Clip]
    found: bool
    trace: list[str]
    latency_ms: int


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _start(question: str, video_id: str | None, settings: Settings | None) -> dict:
    return {"question": question, "video_id": video_id, "settings": settings or Settings(), "attempts": 0, "trace": []}


def run(question: str, video_id: str | None = None, settings: Settings | None = None) -> SearchResult:
    """Whole search in one go (CLI and eval)."""
    t0 = time.perf_counter()
    out = _get_graph().invoke(_start(question, video_id, settings))
    return SearchResult(question=question, query=out.get("query", question), answer=out.get("answer", ""),
                        clips=out.get("final", []), found=out.get("found", False), trace=out.get("trace", []),
                        latency_ms=round((time.perf_counter() - t0) * 1000))


def _clips_json(clips: list[Clip]) -> list[dict]:
    return [{**c.model_dump(), "label": c.label} for c in clips]


def stream(question: str, video_id: str | None = None, settings: Settings | None = None) -> Iterator[dict]:
    """Same search, but yields an event after every step (the web UI shows them live).

    {"type": "step", "node": "grade", "text": "...", "clips": [...]}   clips present when the step changed them
    {"type": "done", "answer": "...", "found": true, "clips": [...], "query": "...", "latency_ms": 1234}
    """
    t0 = time.perf_counter()
    query = question
    for update in _get_graph().stream(_start(question, video_id, settings), stream_mode="updates"):
        for node, delta in update.items():
            query = delta.get("query", query)
            event = {"type": "step", "node": node, "text": (delta.get("trace") or [""])[-1]}
            changed = delta.get("checked") if "checked" in delta else delta.get("kept", delta.get("candidates"))
            if changed is not None and node not in ("answer", "not_found"):
                event["clips"] = _clips_json(changed)
            yield event
            if node in ("answer", "not_found"):
                yield {"type": "done", "answer": delta.get("answer", ""), "found": delta.get("found", False),
                       "clips": _clips_json(delta.get("final", [])), "query": query,
                       "latency_ms": round((time.perf_counter() - t0) * 1000)}
