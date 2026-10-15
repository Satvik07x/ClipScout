import { useState } from "react";
import { deleteVideo, formatTime, resumeIndexing } from "../api.js";

export default function Library({ videos, scopeId, showAll = true, onSelect, onAdd, onChanged }) {
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState("");

  const act = async (id, fn) => {
    setBusy(id);
    setError("");
    try {
      await fn(id);
      onChanged();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  const remove = (video) => {
    if (window.confirm(`Remove "${video.title}" and its index?`)) {
      if (scopeId === video.video_id) onSelect(null);
      act(video.video_id, deleteVideo);
    }
  };

  return (
    <aside className="library">
      <div className="library-head">
        <h2>Library</h2>
        <button className="btn btn-small" onClick={onAdd}>
          Add video
        </button>
      </div>

      <ul className="library-list">
        {showAll && (
          <li>
            <button className={`lib-item ${scopeId === null ? "is-selected" : ""}`} onClick={() => onSelect(null)}>
              <span className="lib-title">All videos</span>
              <span className="lib-meta">{videos.filter((v) => v.status === "ready").length} ready</span>
            </button>
          </li>
        )}

        {videos.map((v) => (
          <li key={v.video_id}>
            <div className={`lib-item ${scopeId === v.video_id ? "is-selected" : ""}`}>
              <button className="lib-main" onClick={() => onSelect(v.video_id)} title={v.title}>
                <span className="lib-title">{v.title}</span>
                <span className="lib-meta">
                  <span className={`dot dot-${v.status}`} aria-hidden="true" />
                  {v.status === "ready" && `${formatTime(v.duration)}, ${v.moments} moments`}
                  {v.status === "indexing" && (v.message || "Indexing")}
                  {v.status === "paused" && "Indexing paused"}
                  {v.status === "error" && "Indexing stopped"}
                </span>
              </button>

              {v.status === "indexing" && (
                <div className="progress" role="progressbar" aria-valuenow={Math.round(v.progress * 100)}>
                  <div className="progress-fill" style={{ width: `${Math.max(4, v.progress * 100)}%` }} />
                </div>
              )}

              {v.status === "error" && <p className="lib-error">{v.error}</p>}

              <div className="lib-actions">
                {(v.status === "paused" || v.status === "error") && (
                  <button className="link" disabled={busy === v.video_id} onClick={() => act(v.video_id, resumeIndexing)}>
                    {v.status === "error" ? "Retry" : "Resume"}
                  </button>
                )}
                {v.status !== "indexing" && (
                  <button className="link link-danger" disabled={busy === v.video_id} onClick={() => remove(v)}>
                    Remove
                  </button>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>

      {error && <p className="lib-error">{error}</p>}
    </aside>
  );
}
