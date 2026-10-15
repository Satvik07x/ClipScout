import { useCallback, useEffect, useRef, useState } from "react";
import { listVideos, streamSearch } from "./api.js";
import Library from "./components/Library.jsx";
import AddVideoDialog from "./components/AddVideoDialog.jsx";
import SearchPanel from "./components/SearchPanel.jsx";
import Player from "./components/Player.jsx";
import Results from "./components/Results.jsx";

const IDLE = { status: "idle", steps: [], clips: [], answer: "", found: false, query: "", latencyMs: 0, error: "" };

export default function App() {
  const [videos, setVideos] = useState([]);
  const [loadError, setLoadError] = useState("");
  const [scopeId, setScopeId] = useState(null); // null = search all videos
  const [activeId, setActiveId] = useState(null); // video shown in the player
  const [seek, setSeek] = useState({ t: 0, nonce: 0 });
  const [search, setSearch] = useState(IDLE);
  const [adding, setAdding] = useState(false);
  const cancelSearch = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const list = await listVideos();
      setVideos(list);
      setLoadError("");
      setActiveId((current) => current ?? list.find((v) => v.status === "ready")?.video_id ?? null);
    } catch {
      setLoadError("Can't reach the server. Start it with: uvicorn server:app --reload");
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // While anything is indexing, ask the server for progress every 2 seconds.
  const indexing = videos.some((v) => v.status === "indexing");
  useEffect(() => {
    if (!indexing) return;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [indexing, refresh]);

  const playAt = (videoId, t) => {
    setActiveId(videoId);
    setSeek((s) => ({ t, nonce: s.nonce + 1 }));
  };

  const runSearch = (q, verify) => {
    cancelSearch.current?.();
    setSearch({ ...IDLE, status: "running" });
    cancelSearch.current = streamSearch({ q, videoId: scopeId, verify }, (event) => {
      if (event.type === "step") {
        setSearch((s) => ({
          ...s,
          steps: [...s.steps, { node: event.node, text: event.text }],
          clips: event.clips ?? s.clips,
        }));
      } else if (event.type === "done") {
        setSearch((s) => ({
          ...s,
          status: "done",
          clips: event.clips,
          answer: event.answer,
          found: event.found,
          query: event.query,
          latencyMs: event.latency_ms,
        }));
        if (event.found && event.clips.length) playAt(event.clips[0].video_id, event.clips[0].best_start);
      } else if (event.type === "error") {
        setSearch((s) => ({ ...s, status: "error", error: event.message }));
      }
    });
  };

  useEffect(() => () => cancelSearch.current?.(), []);

  const selectVideo = (id) => {
    setScopeId(id);
    if (id) setActiveId(id);
  };

  const active = videos.find((v) => v.video_id === activeId) ?? null;
  const ready = videos.filter((v) => v.status === "ready");

  return (
    <div className="app">
      <Library
        videos={videos}
        scopeId={scopeId}
        onSelect={selectVideo}
        onAdd={() => setAdding(true)}
        onChanged={refresh}
      />

      <main className="main">
        <header className="masthead">
          <h1 className="wordmark">ClipScout</h1>
          <p className="tagline">Ask a question. Jump to the moment.</p>
        </header>

        {loadError && <p className="banner banner-error">{loadError}</p>}

        {videos.length === 0 && !loadError ? (
          <section className="empty">
            <h2>Add your first video</h2>
            <p>Paste a YouTube link or upload a file. Gemini watches it once, then every search is instant.</p>
            <button className="btn btn-primary" onClick={() => setAdding(true)}>
              Add video
            </button>
          </section>
        ) : (
          <>
            <SearchPanel
              disabled={ready.length === 0}
              scopeLabel={scopeId ? videos.find((v) => v.video_id === scopeId)?.title : "all videos"}
              search={search}
              onSearch={runSearch}
            />
            <div className={`stage ${search.status === "idle" ? "" : "stage-split"}`}>
              <Player video={active} seek={seek} clips={search.clips} onSeek={(t) => playAt(activeId, t)} />
              <Results search={search} videos={videos} activeId={activeId} onPlay={playAt} />
            </div>
          </>
        )}
      </main>

      {adding && (
        <AddVideoDialog
          onClose={() => setAdding(false)}
          onAdded={(video) => {
            setAdding(false);
            refresh();
            selectVideo(video.video_id);
          }}
        />
      )}
    </div>
  );
}
