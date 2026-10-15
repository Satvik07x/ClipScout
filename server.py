"""ClipScout web API (FastAPI). Deliberately thin: the AI lives in the clipscout/ package;
this file only exposes it over HTTP.

    GET    /api/config              which mode the server runs in + the public-demo limits
    GET    /api/videos              library (with indexing status)
    POST   /api/videos/youtube      add a YouTube video  -> indexing starts in the background
    POST   /api/videos/upload       add a video file     -> same (disabled in public mode)
    POST   /api/videos/{id}/index   resume / retry indexing
    DELETE /api/videos/{id}         remove from library and index
    GET    /api/videos/{id}/file    stream an uploaded file to the player (supports seeking)
    GET    /api/search?q=...        Server-Sent Events: one event per agent step, then the answer

Run:  uvicorn server:app --reload      (API docs at http://localhost:8000/docs)

Public mode (CLIPSCOUT_PUBLIC=1): each browser sends a random visitor id and sees only its own videos;
per-visitor, per-network and daily limits protect the shared free Gemini quota. See config.py.
"""
import json
import logging
import os
import re
import shutil
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clipscout import agent, config, indexer, library, owners
from clipscout.limits import DAY, HOUR, limiter
from clipscout.schemas import VideoSource
from clipscout.video import file_source, youtube_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("clipscout.server")

# ---------------------------------------------------------------- background indexing jobs
# In-memory job table: enough for one machine. (At scale: a queue like Redis + separate workers.)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def set_job(video_id: str, **fields) -> None:
    with jobs_lock:
        jobs.setdefault(video_id, {}).update(fields)


def run_indexing(src: VideoSource, force: bool = False) -> None:
    """Runs in FastAPI's thread pool after the request has already returned."""
    set_job(src.video_id, status="indexing", progress=0.0, message="Starting", error="")
    try:
        indexer.index_video(src, force=force,
                            progress=lambda f, m: set_job(src.video_id, progress=round(f, 3), message=m))
        set_job(src.video_id, status="ready", progress=1.0, message="Ready")
    except Exception as e:
        log.exception("indexing failed for %s", src.video_id)
        set_job(src.video_id, status="error", error=str(e), message="Indexing stopped")


def start_indexing(src: VideoSource, background: BackgroundTasks, force: bool = False) -> None:
    if jobs.get(src.video_id, {}).get("status") == "indexing":
        raise HTTPException(409, "This video is already being indexed.")
    set_job(src.video_id, status="indexing", progress=0.0, message="Queued", error="")
    library.save(library.load(src.video_id) or src)   # appears in the library straight away
    background.add_task(run_indexing, src, force)


# ---------------------------------------------------------------- response shape

class VideoOut(BaseModel):
    video_id: str
    title: str
    kind: str
    youtube_id: str
    file_url: str
    duration: float
    moments: int
    status: str      # ready | indexing | paused | error
    progress: float
    message: str
    error: str


def to_out(src: VideoSource) -> VideoOut:
    job = jobs.get(src.video_id, {})
    status = job.get("status") or ("ready" if src.is_indexed else "paused")
    return VideoOut(video_id=src.video_id, title=src.title, kind=src.kind, youtube_id=src.youtube_id,
                    file_url=f"/api/videos/{src.video_id}/file" if src.kind == "file" else "",
                    duration=src.duration, moments=len(src.moments), status=status,
                    progress=job.get("progress", src.progress), message=job.get("message", ""),
                    error=job.get("error", ""))


# ---------------------------------------------------------------- who is asking (public mode)

_VISITOR_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def get_visitor(request: Request) -> str:
    """Public mode: the web app gives every browser a random id, sent as the X-Visitor header
    (or ?visitor= for the search stream, because EventSource cannot set headers).
    Locally there are no visitors: everything is visible and the id is ""."""
    if not config.PUBLIC_MODE:
        return ""
    visitor = request.headers.get("x-visitor") or request.query_params.get("visitor", "")
    if not _VISITOR_RE.match(visitor):
        raise HTTPException(400, "Missing visitor id. Reload the page.")
    return visitor


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def visible(video_id: str, visitor: str) -> bool:
    return not config.PUBLIC_MODE or owners.has(video_id, visitor)


def get_or_404(video_id: str, visitor: str = "") -> VideoSource:
    src = library.load(video_id)
    if src is None or not visible(video_id, visitor):
        raise HTTPException(404, "Video not found.")
    return src


def daily_left() -> int:
    return max(0, config.DAILY_INDEX_LIMIT - limiter.used("global", DAY))


def charge_quota(ip: str) -> None:
    """Public mode: a video that Gemini has to watch costs shared free quota. Refuse when it is used up."""
    if not limiter.has_room(f"ip:{ip}", config.ADDS_PER_IP_PER_DAY, DAY):
        raise HTTPException(429, "Daily limit reached for your network. Try again tomorrow.")
    if daily_left() == 0:
        raise HTTPException(429, "Today's free Gemini quota for the demo is used up. Try again tomorrow, "
                                 "or run ClipScout on your own machine with your own key.")
    limiter.hit(f"ip:{ip}")
    limiter.hit("global")


# ---------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(_app: FastAPI):
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    try:
        added = indexer.sync_vector_store()   # no-op normally; re-embeds text if the vector store is new
        if added:
            log.info("re-embedded %d moments into the vector store", added)
    except Exception as e:
        log.warning("could not sync vector store at startup: %s", e)
    yield


app = FastAPI(title="ClipScout API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/config")
def app_config():
    """The web app reads this to adapt itself (hide uploads, show limits) and to wake a sleeping server."""
    return {"public": config.PUBLIC_MODE, "uploads": not config.PUBLIC_MODE,
            "max_videos": config.MAX_VIDEOS_PER_VISITOR, "max_minutes": config.MAX_VIDEO_MIN,
            "daily_left": daily_left()}


@app.get("/api/videos", response_model=list[VideoOut])
def list_videos(request: Request):
    visitor = get_visitor(request)
    return [to_out(v) for v in library.all_videos(only_indexed=False) if visible(v.video_id, visitor)]


@app.get("/api/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str, request: Request):
    return to_out(get_or_404(video_id, get_visitor(request)))


class YouTubeIn(BaseModel):
    url: str
    minutes: float
    title: str = ""


def add_public(src: VideoSource, visitor: str, ip: str, background: BackgroundTasks) -> VideoOut:
    """Public mode: enforce the limits. A video someone already indexed is shared at no cost."""
    mine = owners.of(visitor)
    if src.video_id not in mine and len(mine) >= config.MAX_VIDEOS_PER_VISITOR:
        raise HTTPException(429, f"You can keep {config.MAX_VIDEOS_PER_VISITOR} videos on the public demo. "
                                 "Remove one to add another.")
    existing = library.load(src.video_id)
    if existing and (existing.is_indexed or jobs.get(src.video_id, {}).get("status") == "indexing"):
        owners.add(src.video_id, visitor)          # already paid for: no Gemini quota used
        return to_out(existing)
    charge_quota(ip)
    owners.add(src.video_id, visitor)
    start_indexing(src, background)
    return to_out(library.load(src.video_id))


@app.post("/api/videos/youtube", response_model=VideoOut, status_code=202)
def add_youtube(body: YouTubeIn, background: BackgroundTasks, request: Request):
    visitor = get_visitor(request)
    if body.minutes <= 0:
        raise HTTPException(422, "Enter the video length in minutes.")
    if config.PUBLIC_MODE and body.minutes > config.MAX_VIDEO_MIN:
        # Gemini is only ever asked to watch the minutes entered here, so the cap holds even if the number is wrong.
        raise HTTPException(422, f"The public demo takes videos up to {config.MAX_VIDEO_MIN:g} minutes. "
                                 "Run ClipScout on your own machine for longer ones.")
    try:
        src = youtube_source(body.url, body.minutes, body.title.strip())
    except ValueError as e:
        raise HTTPException(422, str(e))
    if config.PUBLIC_MODE:
        return add_public(src, visitor, client_ip(request), background)
    start_indexing(src, background)
    return to_out(library.load(src.video_id))


@app.post("/api/videos/upload", response_model=VideoOut, status_code=202)
def add_upload(background: BackgroundTasks, file: UploadFile = File(...),
               title: str = Form(""), minutes: float = Form(0.0)):
    if config.PUBLIC_MODE:
        raise HTTPException(403, "Uploads are turned off on the public demo. Paste a YouTube link instead.")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "video.mp4")
    path = os.path.join(config.UPLOAD_DIR, name)
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)           # streamed to disk, never fully in memory
    src = file_source(path, title.strip() or (file.filename or name), minutes)
    if src.duration <= 0:
        raise HTTPException(422, "Couldn't read the video length. Enter it in minutes.")
    start_indexing(src, background)
    return to_out(library.load(src.video_id))


@app.post("/api/videos/{video_id}/index", response_model=VideoOut, status_code=202)
def resume_indexing(video_id: str, background: BackgroundTasks, request: Request, force: bool = False):
    visitor = get_visitor(request)
    src = get_or_404(video_id, visitor)
    if config.PUBLIC_MODE:
        force = False                                # a visitor can never trigger a full re-watch
        if src.is_indexed:
            return to_out(src)
        charge_quota(client_ip(request))             # retries cost Gemini calls too
    start_indexing(src, background, force=force)
    return to_out(src)


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: str, request: Request):
    visitor = get_visitor(request)
    src = get_or_404(video_id, visitor)
    if jobs.get(video_id, {}).get("status") == "indexing":
        raise HTTPException(409, "Wait for indexing to finish before deleting.")
    if config.PUBLIC_MODE and not owners.remove(video_id, visitor):
        return                                       # someone else still keeps it: only drop this visitor's claim
    indexer.remove(video_id)
    jobs.pop(video_id, None)
    if src.kind == "file" and os.path.exists(src.local_path):
        os.remove(src.local_path)


@app.get("/api/videos/{video_id}/file")
def video_file(video_id: str, request: Request):
    src = get_or_404(video_id, get_visitor(request))
    if src.kind != "file" or not os.path.exists(src.local_path):
        raise HTTPException(404, "No local file for this video.")
    return FileResponse(src.local_path)              # handles Range requests, so the player can seek


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _stream(chunks) -> StreamingResponse:
    return StreamingResponse(chunks, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _stream_error(message: str) -> StreamingResponse:
    """A browser's EventSource can't read the body of an HTTP error, so refusals travel inside the stream."""
    return _stream(iter([_sse({"type": "error", "message": message})]))


@app.get("/api/search")
def search(request: Request, q: str, video_id: str | None = None, verify: bool = True):
    """Server-Sent Events: the browser receives each agent step the moment it finishes."""
    visitor = get_visitor(request)
    if not q.strip():
        raise HTTPException(422, "Type a question.")
    if config.PUBLIC_MODE:
        src = library.load(video_id) if video_id else None
        if not video_id or src is None or not visible(video_id, visitor):
            return _stream_error("Pick one of your videos to search.")
        if not src.is_indexed:
            return _stream_error("This video is still being indexed. Try again in a moment.")
        if not limiter.allow(f"search:{visitor}", config.SEARCHES_PER_HOUR, HOUR):
            return _stream_error(f"Search limit reached ({config.SEARCHES_PER_HOUR} per hour on the public demo). "
                                 "Try again later.")
    settings = agent.Settings(verify=verify)

    def events():
        try:
            for event in agent.stream(q.strip(), video_id=video_id or None, settings=settings):
                yield _sse(event)
        except Exception as e:
            log.exception("search failed")
            yield _sse({"type": "error", "message": str(e)})

    return _stream(events())


# Production: `npm run build` in web/, then this one server also serves the UI at http://localhost:8000
if os.path.isdir("web/dist"):
    app.mount("/", StaticFiles(directory="web/dist", html=True), name="web")
