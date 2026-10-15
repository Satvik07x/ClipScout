"""Measure search quality on your own labelled queries, and what each agent step adds (ablation).

1. Index your eval videos first (app or `python cli.py index ...`).
2. cp eval/queries.example.json eval/queries.json and write ~100 queries with the true time range.
   Write the queries BEFORE looking at search results, otherwise you will unconsciously write easy ones.
3. python -m eval.run_eval                 # cheap modes only (no Gemini verify)
   python -m eval.run_eval --all           # includes the Gemini verify mode (slow, uses quota)

A query counts as a hit at rank r if the r-th returned clip STARTS PLAYING inside the true range
(±TOLERANCE s). Using the playback point, not mere overlap, stops long clips from gaming the metric.
Metrics: Recall@1, Recall@5, MRR (mean of 1/rank of the first hit), latency p50/p95.
"""
import argparse
import json
import os
import statistics
import time

from clipscout import agent, library

TOLERANCE = 5.0
HERE = os.path.dirname(__file__)

MODES = {
    "A. vector only":            agent.Settings(rewrite=False, keyword=False, grade=False, verify=False, answer=False),
    "B. + BM25 (hybrid, RRF)":   agent.Settings(rewrite=False, keyword=True, grade=False, verify=False, answer=False),
    "C. + LLM query rewrite":    agent.Settings(rewrite=True, keyword=True, grade=False, verify=False, answer=False),
    "D. + LLM grading":          agent.Settings(rewrite=True, keyword=True, grade=True, verify=False, answer=False),
    "E. + Gemini frame verify":  agent.Settings(rewrite=True, keyword=True, grade=True, verify=True, answer=False),
}
EXPENSIVE = {"E. + Gemini frame verify"}


def resolve_video(ref: str) -> str | None:
    """Queries may name a video by id or by title."""
    for v in library.all_videos():
        if ref in (v.video_id, v.title):
            return v.video_id
    return None


def first_hit_rank(clips, start: float, end: float) -> int | None:
    for rank, c in enumerate(clips, start=1):
        if c.lands_in(start, end, TOLERANCE):
            return rank
    return None


def evaluate(queries: list[dict], settings: agent.Settings, pause: float) -> dict:
    ranks, latencies, misses = [], [], []
    for q in queries:
        vid = resolve_video(q["video"])
        if vid is None:
            print(f"  ! video not indexed: {q['video']}")
            continue
        r = agent.run(q["query"], video_id=vid, settings=settings)
        rank = first_hit_rank(r.clips, q["start"], q["end"])
        ranks.append(rank)
        latencies.append(r.latency_ms)
        if rank is None:
            misses.append(q["query"])
        if pause:
            time.sleep(pause)
    n = len(ranks) or 1
    lat = sorted(latencies) or [0]
    return {
        "n": len(ranks),
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "p50_ms": statistics.median(lat),
        "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=os.path.join(HERE, "queries.json"))
    ap.add_argument("--all", action="store_true", help="also run the Gemini verify mode")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between queries (free-tier limits)")
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)
    print(f"{len(queries)} queries\n")

    rows = []
    for name, settings in MODES.items():
        if name in EXPENSIVE and not args.all:
            continue
        uses_llm = settings.rewrite or settings.grade or settings.verify
        print(f"running {name} ...")
        m = evaluate(queries, settings, args.pause if uses_llm else 0)
        rows.append((name, m))
        print(f"  R@1 {m['recall@1']:.0%}  R@5 {m['recall@5']:.0%}  MRR {m['mrr']:.2f}  p50 {m['p50_ms']:.0f} ms")

    table = ["| Setup | Recall@1 | Recall@5 | MRR | p50 latency | p95 latency |", "|---|---|---|---|---|---|"]
    for name, m in rows:
        table.append(f"| {name} | {m['recall@1']:.0%} | {m['recall@5']:.0%} | {m['mrr']:.2f} | "
                     f"{m['p50_ms']:.0f} ms | {m['p95_ms']:.0f} ms |")
    best_name, best = rows[-1]
    report = [f"# ClipScout eval ({time.strftime('%Y-%m-%d')}, {rows[0][1]['n']} queries, ±{TOLERANCE:.0f}s tolerance)", "",
              *table, "", f"## Misses in the last mode ({best_name})", ""]
    report += [f"- {q}" for q in best["misses"]] or ["- none"]
    out = os.path.join(HERE, "results.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n" + "\n".join(table) + f"\n\nwritten to {out}")


if __name__ == "__main__":
    main()
