"""ClipScout web API (FastAPI). Deliberately thin: the AI lives in the clipscout/ package;
this file only exposes it over HTTP.

    GET    /api/videos              library (with indexing status)
    POST   /api/videos/youtube      add a YouTube video  -> indexing starts in the background
    POST   /api/videos/upload       add a video file     -> same
    POST   /api/videos/{id}/index   resume / retry indexing
    DELETE /api/videos/{id}         remove from library and index
    GET    /api/videos/{id}/file    stream an uploaded file to the player (supports seeking)
    GET    /api/search?q=...        Server-Sent Events: one event per agent step, then the answer

Run:  uvicorn server:app --reload      (API docs at http://localhost:8000/docs)
"""
import json
import logging
import os
import re
import shutil
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clipscout import agent, config, indexer, library
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


def get_or_404(video_id: str) -> VideoSource:
    src = library.load(video_id)
    if src is None:
        raise HTTPException(404, "Video not found.")
    return src


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
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/videos", response_model=list[VideoOut])
def list_videos():
    return [to_out(v) for v in library.all_videos(only_indexed=False)]


@app.get("/api/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str):
    return to_out(get_or_404(video_id))


class YouTubeIn(BaseModel):
    url: str
    minutes: float
    title: str = ""


@app.post("/api/videos/youtube", response_model=VideoOut, status_code=202)
def add_youtube(body: YouTubeIn, background: BackgroundTasks):
    if body.minutes <= 0:
        raise HTTPException(422, "Enter the video length in minutes.")
    try:
        src = youtube_source(body.url, body.minutes, body.title.strip())
    except ValueError as e:
        raise HTTPException(422, str(e))
    start_indexing(src, background)
    return to_out(library.load(src.video_id))


@app.post("/api/videos/upload", response_model=VideoOut, status_code=202)
def add_upload(background: BackgroundTasks, file: UploadFile = File(...),
               title: str = Form(""), minutes: float = Form(0.0)):
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
def resume_indexing(video_id: str, background: BackgroundTasks, force: bool = False):
    src = get_or_404(video_id)
    start_indexing(src, background, force=force)
    return to_out(src)


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: str):
    src = get_or_404(video_id)
    if jobs.get(video_id, {}).get("status") == "indexing":
        raise HTTPException(409, "Wait for indexing to finish before deleting.")
    indexer.remove(video_id)
    jobs.pop(video_id, None)
    if src.kind == "file" and os.path.exists(src.local_path):
        os.remove(src.local_path)


@app.get("/api/videos/{video_id}/file")
def video_file(video_id: str):
    src = get_or_404(video_id)
    if src.kind != "file" or not os.path.exists(src.local_path):
        raise HTTPException(404, "No local file for this video.")
    return FileResponse(src.local_path)              # handles Range requests, so the player can seek


@app.get("/api/search")
def search(q: str, video_id: str | None = None, verify: bool = True):
    """Server-Sent Events: the browser receives each agent step the moment it finishes."""
    if not q.strip():
        raise HTTPException(422, "Type a question.")
    settings = agent.Settings(verify=verify)

    def events():
        try:
            for event in agent.stream(q.strip(), video_id=video_id or None, settings=settings):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.exception("search failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Production: `npm run build` in web/, then this one server also serves the UI at http://localhost:8000
if os.path.isdir("web/dist"):
    app.mount("/", StaticFiles(directory="web/dist", html=True), name="web")
