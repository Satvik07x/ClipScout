# ClipScout · Multimodal Video Search Engine

Ask a question in plain Hindi or English, get the exact clip.
"red bike wala aadmi gate pe kab aaya?" → **02:34**.

Video is the largest kind of data we create and the least searchable. A guard scrubs hours of CCTV to find one
moment; a student scrubs a 90-minute lecture to find one explanation. ClipScout watches a video once, builds a
searchable index of what is **seen**, **said** and **written on screen**, and answers questions with timestamps
it has double-checked against the real frames.

## How it works

```
INDEXING (once per video)                         SEARCH (LangGraph agent, streamed to the UI step by step)

video / YouTube link                              question ─► rewrite ─► retrieve ─► grade ─► verify ─► answer
   │  5-minute pieces, 3 in parallel                           ▲   (Groq)    (hybrid)  (Groq)  (Gemini,  (Groq)
   ▼                                                           │                               top 2 in
Gemini: describe every ~10 s                                   └──── retry once with a better   parallel)
 {start, end, visual, speech, on_screen_text}                        query if nothing matches
   │
   ▼
library/<id>.json + ChromaDB (Gemini embeddings) + BM25
```

| Step | What happens | Why |
|---|---|---|
| Index | Gemini watches 5-min pieces (3 at a time) and returns timestamped JSON moments | Fits free-tier limits; every finished piece is saved, so a crash or quota hit resumes where it stopped |
| Embed | Gemini embedding API (multilingual, 768-dim), queries cached | No local model to load; "laal gaadi" and "red car" land close together |
| Retrieve | Vector search (meaning) + BM25 (exact words like "MP04"), fused with Reciprocal Rank Fusion | Each covers the other's blind spot; RRF works on ranks, so different score scales don't matter |
| Merge | Strong neighbouring moments join into one clip; weak ones are dropped | One hit shouldn't become five near-identical results |
| Grade | Groq reads the candidate descriptions and keeps the matching ones | Cheap text-only filter before the expensive step |
| Verify | Gemini looks at the actual frames of the top 2 clips, in parallel | Descriptions can be wrong; pixels are the ground truth |
| Answer | Groq writes 1–3 lines with timestamps from verified clips only | No answer from unverified evidence |

## Run it

You need Python 3.11+, Node.js 18+, and two free API keys:
[Google AI Studio](https://aistudio.google.com/apikey) (Gemini) and [Groq](https://console.groq.com/keys).

**1. Backend (terminal 1)**

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Mac/Linux: cp .env.example .env   -> paste both keys into .env
uvicorn server:app --reload
```

API docs (try every endpoint in the browser): http://localhost:8000/docs

**2. Frontend (terminal 2)**

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173, add a video, search.

**One server instead of two (for a demo):** `cd web && npm run build`, then only `uvicorn server:app`
and open http://localhost:8000.

Free-tier inputs may be used by Google to improve its models, so index public videos or your own footage.

## API

| Method | Path | Does |
|---|---|---|
| GET | `/api/videos` | Library with indexing status and progress |
| POST | `/api/videos/youtube` | Add a YouTube video; indexing runs in the background (202) |
| POST | `/api/videos/upload` | Same for an uploaded file (multipart) |
| POST | `/api/videos/{id}/index` | Resume or retry indexing |
| DELETE | `/api/videos/{id}` | Remove a video and its index |
| GET | `/api/videos/{id}/file` | Stream an uploaded file to the player (supports seeking) |
| GET | `/api/search?q=…&video_id=…&verify=true` | Server-Sent Events: one event per agent step, then the answer |

## Evaluation

```bash
copy eval\queries.example.json eval\queries.json   # write ~100 labelled queries (video, query, start, end)
python -m eval.run_eval          # modes A-D
python -m eval.run_eval --all    # + Gemini verify (uses quota)
```

A query is a hit at rank *r* if the *r*-th clip starts playing inside the true range (±5 s). Results go to
`eval/results.md`:

| Setup | Recall@1 | Recall@5 | MRR | p50 latency |
|---|---|---|---|---|
| A. vector only | | | | |
| B. + BM25 (hybrid, RRF) | | | | |
| C. + LLM query rewrite | | | | |
| D. + LLM grading | | | | |
| E. + Gemini frame verify | | | | |

## Project layout

```
clipscout/            the AI core (everything that matters lives here)
  schemas.py          VideoSource -> Moment -> Clip, the data model
  config.py           every model name and number in one place (overridable from .env)
  llm.py              Gemini client with retry/backoff/model fallback, Groq client
  embeddings.py       Gemini embeddings as a LangChain Embeddings class (+ query cache)
  video.py            YouTube ids, file fingerprint, duration, Files API upload, clipping
  vision.py           the two Gemini tasks: describe_segment, verify_clip
  indexer.py          parallel, resumable indexing pipeline
  library.py          one JSON per video (the "index once" cache), atomic writes
  search.py           Chroma + BM25 + RRF + clip merging
  agent.py            LangGraph agent: rewrite -> retrieve -> grade -> verify -> answer, + streaming
server.py             thin FastAPI layer: REST + background jobs + SSE
web/                  React (Vite) UI: search, player with result markers on the timeline, live agent steps
cli.py                index / search / list / sync from the terminal
eval/run_eval.py      Recall@1/5, MRR, latency, ablation A-E
tests/                offline tests (no API keys): fusion, merging, agent routing, streaming, retries,
                      parallel verify, resume, API endpoints
```

Run the tests with `pytest -q`.

## Limits and next steps

- Gemini samples 1 frame per second, so very fast events can be missed.
- Counting and negation queries ("3 people", "without helmet") are hard for a description-based index.
- Background jobs live in server memory; with many users they would move to a queue (Redis + workers).
- Scaling to thousands of hours: approximate vector index (HNSW), per-camera/date partitions with metadata
  filters, and live streams indexed in short rolling chunks.
