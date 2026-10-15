import { useCallback, useEffect, useRef, useState } from "react";
import { API, getConfig, listVideos, streamSearch } from "./api.js";
import Library from "./components/Library.jsx";
import AddVideoDialog from "./components/AddVideoDialog.jsx";
import SearchPanel from "./components/SearchPanel.jsx";
import Player from "./components/Player.jsx";
import Results from "./components/Results.jsx";

const LOCAL = { public: false, uploads: true, max_videos: 0, max_minutes: 0, daily_left: 0 };
const IDLE = { status: "idle", steps: [], clips: [], answer: "", found: false, query: "", latencyMs: 0, error: "" };

export default function App() {
  const [videos, setVideos] = useState([]);
  const [loadError, setLoadError] = useState("");
  const [scopeId, setScopeId] = useState(null); // null = search all videos
  const [activeId, setActiveId] = useState(null); // video shown in the player
  const [seek, setSeek] = useState({ t: 0, nonce: 0 });
  const [search, setSearch] = useState(IDLE);
  const [adding, setAdding] = useState(false);
  const [cfg, setCfg] = useState(LOCAL);
  const cancelSearch = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const [list, conf] = await Promise.all([listVideos(), getConfig()]);
      setVideos(list);
      setCfg(conf);
      setLoadError("");
      const stillThere = (id) => list.some((v) => v.video_id === id);
      setActiveId((cur) => (cur && stillThere(cur) ? cur : list.find((v) => v.status === "ready")?.video_id ?? null));
      setScopeId((cur) => (cur && stillThere(cur) ? cur : null));
    } catch {
      setLoadError(
        API
          ? "The server is waking up (free hosting sleeps when idle). This takes about a minute, the page retries by itself."
          : "Can't reach the server. Start it with: uvicorn server:app --reload"
      );
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Server unreachable (asleep, restarting): keep trying until it answers.
  useEffect(() => {
    if (!loadError) return;
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [loadError, refresh]);

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
    cancelSearch.current = streamSearch({ q, videoId: scope, verify }, (event) => {
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
  // The public demo searches one video at a time (the one in the player); locally "all videos" is allowed.
  const scope = cfg.public ? scopeId ?? activeId : scopeId;
  const scopeVideo = videos.find((v) => v.video_id === scope);
  const canSearch = cfg.public ? scopeVideo?.status === "ready" : ready.length > 0;

  return (
    <div className="app">
      <Library
        videos={videos}
        scopeId={scope}
        showAll={!cfg.public}
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

        {cfg.public && !loadError && (
          <p className="banner banner-note">
            Public demo. It runs on one shared free Gemini key, so videos are limited to {cfg.max_minutes} minutes,{" "}
            {cfg.max_videos} per visitor, and {cfg.daily_left} new videos are left today.
          </p>
        )}

        {videos.length === 0 && !loadError ? (
          <section className="empty">
            <h2>Add your first video</h2>
            <p>
              {cfg.uploads ? "Paste a YouTube link or upload a file." : "Paste a YouTube link."} Gemini watches it once,
              then every search is instant.
            </p>
            <button className="btn btn-primary" onClick={() => setAdding(true)}>
              Add video
            </button>
          </section>
        ) : (
          <>
            <SearchPanel
              disabled={!canSearch}
              scopeLabel={scope ? scopeVideo?.title : "all videos"}
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
          cfg={cfg}
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
