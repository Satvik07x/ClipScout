"""Agent routing with fake Groq + fake Gemini verify: checks the graph, not the models."""
import pytest

from clipscout import agent, library, search
from clipscout.agent import GradeOut, RewriteOut, Settings
from tests.test_search import sample_video


class FakeLLM:
    def __init__(self, rewrites=("red motorcycle",), grade_ids=(0,), answer="Woh 00:10 pe aaya."):
        self.rewrites, self.grade_ids, self.answer, self.calls = list(rewrites), list(grade_ids), answer, []

    def with_structured_output(self, schema, **kw):
        llm = self

        class Runner:
            def invoke(self, prompt):
                llm.calls.append(schema.__name__)
                if schema is RewriteOut:
                    return RewriteOut(search_query=llm.rewrites.pop(0) if llm.rewrites else "nothing")
                return GradeOut(matching_ids=llm.grade_ids)
        return Runner()

    def invoke(self, prompt):
        self.calls.append("answer")
        return type("Msg", (), {"content": self.answer})()


@pytest.fixture
def indexed():
    src = sample_video()
    library.save(src)
    search.add_video(src)
    return src


def use(monkeypatch, llm, present=True):
    monkeypatch.setattr(agent, "groq", lambda: llm)
    seen = []

    def fake_verify(src, start, end, question):
        seen.append((start, end))
        return {"present": present, "reason": "red motorcycle visible" if present else "no bike", "best_second": start + 2}
    monkeypatch.setattr(agent, "verify_clip", fake_verify)
    return seen


def test_happy_path(monkeypatch, indexed):
    llm = FakeLLM()
    use(monkeypatch, llm)
    r = agent.run("red bike wala kab aaya?")
    assert r.found and r.answer == "Woh 00:10 pe aaya."
    assert r.clips[0].verified and r.clips[0].best_start == 12
    assert llm.calls == ["RewriteOut", "GradeOut", "answer"]
    assert [t.split(":")[0] for t in r.trace] == ["rewrite", "retrieve", "grade", "verify on real frames", "answer"]


def test_verify_rejects_then_retry_then_not_found(monkeypatch, indexed):
    llm = FakeLLM(rewrites=["red motorcycle", "bike rider"])
    use(monkeypatch, llm, present=False)
    r = agent.run("red bike wala kab aaya?")
    assert not r.found and r.clips and all(c.verified is False for c in r.clips)
    assert llm.calls.count("RewriteOut") == 2       # exactly one retry (MAX_RETRIES = 1)
    assert r.trace[-1].startswith("not found")


def test_grade_rejects_all(monkeypatch, indexed):
    llm = FakeLLM(grade_ids=[])
    seen = use(monkeypatch, llm)
    r = agent.run("elephant")
    assert not r.found and seen == []                # Gemini never called when grading finds nothing


def test_retrieval_only_mode_needs_no_llm(monkeypatch, indexed):
    llm = FakeLLM()
    seen = use(monkeypatch, llm)
    r = agent.run("red motorcycle", settings=Settings(rewrite=False, grade=False, verify=False, answer=False))
    assert r.clips[0].start == 10 and llm.calls == [] and seen == []


def test_stream_emits_steps_then_done(monkeypatch, indexed):
    use(monkeypatch, FakeLLM())
    events = list(agent.stream("red bike wala kab aaya?"))
    assert [e["node"] for e in events if e["type"] == "step"] == ["rewrite", "retrieve", "grade", "verify", "answer"]
    assert "clips" in events[1] and "clips" in events[3]           # results reach the UI before the answer
    assert events[3]["clips"][0]["verified"] is True
    done = events[-1]
    assert done["type"] == "done" and done["found"] and done["clips"][0]["label"].startswith("Gate camera @")


def test_verify_runs_in_parallel(monkeypatch, indexed):
    import threading
    import time
    llm = FakeLLM(grade_ids=[0, 1])
    monkeypatch.setattr(agent, "groq", lambda: llm)
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow_verify(src, start, end, question):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        return {"present": True, "reason": "seen", "best_second": start}
    monkeypatch.setattr(agent, "verify_clip", slow_verify)
    search.add_video(sample_video_two_hits())
    agent.run("red motorcycle")
    assert peak[0] == 2


def sample_video_two_hits():
    from clipscout.schemas import Moment
    src = sample_video()
    src.moments.append(Moment(start=200, end=210, visual="another red motorcycle leaves"))
    library.save(src)
    return src
