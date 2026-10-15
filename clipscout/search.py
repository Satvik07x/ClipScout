"""Hybrid retrieval over moments.

    query ──► vector search (Gemini embeddings + Chroma) ─┐  finds MEANING: "two-wheeler" ≈ "red motorbike"
          └─► BM25 keyword search                      ─┴─► Reciprocal Rank Fusion ─► merge ─► clips
                                                              finds EXACT words: "MP04", "Dijkstra"
"""
import re
import threading
from functools import lru_cache

from langchain_core.documents import Document

from clipscout import config
from clipscout.schemas import Clip, VideoSource

# ---------------------------------------------------------------- vector store

@lru_cache(maxsize=1)
def get_embeddings():
    from clipscout.embeddings import GeminiEmbeddings
    return GeminiEmbeddings()


_store = None
_bm25_cache: dict = {}
_lock = threading.RLock()   # the web server runs indexing and searches on different threads


def get_store():
    """Persistent Chroma collection on disk (data/chroma), so the index survives restarts."""
    global _store
    with _lock:
        if _store is None:
            from langchain_chroma import Chroma
            _store = Chroma(collection_name=config.COLLECTION, embedding_function=get_embeddings(),
                            persist_directory=config.CHROMA_DIR, collection_metadata={"hnsw:space": "cosine"})
        return _store


def reset() -> None:
    """Forget cached store/indexes (used by tests and after config changes)."""
    global _store
    _store = None
    _bm25_cache.clear()


def _documents(src: VideoSource) -> list[Document]:
    return [Document(page_content=m.search_text(),
                     metadata={"video_id": src.video_id, "title": src.title, "start": m.start, "end": m.end})
            for m in src.moments]


def add_video(src: VideoSource) -> int:
    """(Re)place one video's moments in the vector store. Returns how many were added."""
    docs = _documents(src)
    with _lock:
        remove_video(src.video_id)
        if docs:
            get_store().add_documents(docs, ids=[f"{src.video_id}:{i}" for i in range(len(docs))])
        _bm25_cache.clear()
    return len(docs)


def count(video_id: str) -> int:
    return len(get_store().get(where={"video_id": video_id}, include=[])["ids"])


def total() -> int:
    return len(get_store().get(include=[])["ids"])


def remove_video(video_id: str) -> None:
    with _lock:
        old = get_store().get(where={"video_id": video_id})
        if old and old.get("ids"):
            get_store().delete(ids=old["ids"])
        _bm25_cache.clear()


# ---------------------------------------------------------------- the two retrievers

def vector_search(query: str, k: int | None = None, video_id: str | None = None) -> list[Document]:
    return get_store().similarity_search(query, k=k or config.FETCH_K, filter={"video_id": video_id} if video_id else None)


_STOP = set("a an the is are was were be of in on at to for and or with by from this that it its he she they "
             "his her them there here when where what who how kab kahan kya hai ka ki ke wala wali mein me se".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, minus filler words (English + common Hinglish)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


def _bm25(video_id: str | None):
    """BM25 index built from the documents already in Chroma; rebuilt only when the library changes."""
    key = video_id or "*"
    with _lock:
        return _bm25_cache[key] if key in _bm25_cache else _build_bm25(key, video_id)


def _build_bm25(key: str, video_id: str | None):
    # BM25Plus instead of plain BM25Okapi: Okapi gives a word ZERO weight when it appears in half
    # of the moments (common in short videos, e.g. "car" in a parking-lot clip). Plus never does.
    from rank_bm25 import BM25Plus
    got = get_store().get(where={"video_id": video_id} if video_id else None, include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(got["documents"], got["metadatas"])]
    tokens = [tokenize(d.page_content) for d in docs]
    _bm25_cache[key] = (BM25Plus(tokens) if docs else None, docs, [set(t) for t in tokens])
    return _bm25_cache[key]


def keyword_search(query: str, k: int | None = None, video_id: str | None = None) -> list[Document]:
    k = k or config.FETCH_K
    index, docs, vocab = _bm25(video_id)
    if index is None:
        return []
    q = tokenize(query)
    scores = index.get_scores(q)
    matching = [i for i in range(len(docs)) if vocab[i].intersection(q)]   # must share at least one word
    matching.sort(key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in matching[:k]]


# ---------------------------------------------------------------- fusion + merging (pure functions)

def _key(d: Document) -> tuple:
    return d.metadata["video_id"], d.metadata["start"]


def rrf(ranked_lists: list[list[Document]], k: int | None = None) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_in_list), rank starting at 1.

    Vector distances and BM25 scores live on different scales, so they cannot be added.
    Ranks can. Something near the top of BOTH lists wins; k=60 stops one list from dominating.
    """
    k = k or config.RRF_K
    scores: dict[tuple, float] = {}
    first_seen: dict[tuple, Document] = {}
    for docs in ranked_lists:
        for rank, d in enumerate(docs, start=1):
            key = _key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, d)
    return sorted(((first_seen[key], s) for key, s in scores.items()), key=lambda x: x[1], reverse=True)


def merge_into_clips(scored: list[tuple[Document, float]], gap: float | None = None,
                     min_ratio: float | None = None, max_len: float | None = None) -> list[Clip]:
    """Seed-and-grow: the best moment starts a clip; a neighbouring moment (within `gap` s) joins it only
    if it is nearly as relevant (score >= min_ratio x the clip's best) and the clip stays under `max_len` s.

    Weak neighbours are dropped instead of glued on, so one good hit does not swallow the whole video.
    """
    gap = config.MERGE_GAP_SEC if gap is None else gap
    min_ratio = config.MERGE_MIN_RATIO if min_ratio is None else min_ratio
    max_len = config.MAX_CLIP_SEC if max_len is None else max_len
    clips: list[dict] = []
    for d, s in sorted(scored, key=lambda x: x[1], reverse=True):
        vid, start, end = d.metadata["video_id"], d.metadata["start"], d.metadata["end"]
        near = next((c for c in clips if c["video_id"] == vid
                     and start <= c["end"] + gap and end >= c["start"] - gap), None)
        if near is None:
            clips.append({"video_id": vid, "title": d.metadata.get("title", ""), "start": start, "end": end,
                          "best_start": start, "score": s, "parts": [(start, d.page_content)]})
        elif s >= min_ratio * near["score"] and max(near["end"], end) - min(near["start"], start) <= max_len:
            near["start"], near["end"] = min(near["start"], start), max(near["end"], end)
            near["parts"].append((start, d.page_content))
        # else: a weak neighbour of an existing clip -> ignore it

    out = []
    for c in clips:
        text = " / ".join(t for _, t in sorted(c.pop("parts")))
        out.append(Clip(**c, text=text))
    return out   # already in score order: clips were created from the best moment down


def hybrid_search(query: str, top_k: int | None = None, video_id: str | None = None,
                  use_keyword: bool = True) -> list[Clip]:
    lists = [vector_search(query, video_id=video_id)]
    if use_keyword:
        lists.append(keyword_search(query, video_id=video_id))
    return merge_into_clips(rrf(lists))[: top_k or config.TOP_K]
