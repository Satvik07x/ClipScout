"""Model clients.

- Gemini: watches video (gemini_json) and turns text into vectors (embeddings.py).
- Groq: fast text reasoning for the agent (query rewrite, grading, final answer).

with_retry() is shared by every Gemini call: exponential backoff on rate limits (HTTP 429)
and falling back to the next model in the list if one is not available on this key.
"""
import json
import logging
import time
from functools import lru_cache
from typing import Callable, TypeVar

from clipscout import config

log = logging.getLogger("clipscout.llm")
T = TypeVar("T")

_RATE_LIMIT_HINTS = ("429", "resource_exhausted", "rate limit", "503", "unavailable", "timed out", "timeout")
_BAD_MODEL_HINTS = ("404", "not found", "not available", "not supported", "unknown model", "is not allowed")


@lru_cache(maxsize=1)
def gemini_client():
    from google import genai
    return genai.Client(api_key=config.get_secret("GEMINI_API_KEY"))


def with_retry(call: Callable[[str], T], models: list[str], max_rounds: int = 4) -> T:
    """Run call(model) until one model answers.

    Busy (503) or rate-limited (429) model -> try the NEXT model straight away; only when every
    model in the list has failed in a round do we wait (5s, 10s, 20s) before the next round.
    A model that doesn't exist on this key is dropped for good.
    """
    pool = list(models)
    last_error: Exception | None = None
    for round_no in range(max_rounds):
        for model in list(pool):
            try:
                return call(model)
            except json.JSONDecodeError as e:      # truncated / broken JSON: next model / next round
                last_error = e
                log.warning("invalid JSON from %s, trying again", model)
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if any(h in msg for h in _RATE_LIMIT_HINTS):
                    log.warning("%s is busy or rate-limited (%s); trying the next model", model, str(e)[:120])
                elif any(h in msg for h in _BAD_MODEL_HINTS):
                    log.warning("model %s not usable on this key; dropping it", model)
                    pool.remove(model)
                else:
                    raise
        if not pool:
            break
        if round_no < max_rounds - 1:
            wait = 5 * 2 ** round_no
            log.warning("all models busy, waiting %ss (round %d of %d)", wait, round_no + 1, max_rounds)
            time.sleep(wait)
    raise RuntimeError(f"Gemini is busy or out of free quota right now. Try again in a minute. ({last_error})")


def gemini_json(parts: list[dict], schema: dict, *, system: str | None = None) -> dict:
    """Send video/text parts to Gemini; get back a dict that follows the JSON `schema`."""
    client = gemini_client()

    def call(model: str) -> dict:
        request = dict(model=model, input=parts,
                       response_format={"type": "text", "mime_type": "application/json", "schema": schema})
        if system:
            request["system_instruction"] = system
        return json.loads(client.interactions.create(**request).output_text or "")

    return with_retry(call, config.GEMINI_MODELS)


@lru_cache(maxsize=1)
def groq():
    from langchain_groq import ChatGroq
    kwargs = dict(model=config.GROQ_MODEL, api_key=config.get_secret("GROQ_API_KEY"), temperature=0, max_retries=3)
    if config.GROQ_REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    return ChatGroq(**kwargs)
