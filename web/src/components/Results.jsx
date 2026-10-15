import { formatTime } from "../api.js";

const STATUS = {
  true: { text: "Seen in the video", cls: "verified" },
  false: { text: "Not in the frames", cls: "rejected" },
  null: { text: "Not checked", cls: "unchecked" },
};

export default function Results({ search, videos, activeId, onPlay }) {
  if (search.status === "idle") return null;

  if (search.status === "error") {
    return (
      <section className="results">
        <p className="banner banner-error">{search.error}</p>
      </section>
    );
  }

  const title = (id) => videos.find((v) => v.video_id === id)?.title ?? "";
  const youtubeId = (id) => videos.find((v) => v.video_id === id)?.youtube_id ?? "";

  return (
    <section className="results">
      {search.status === "done" && (
        <div className={`answer ${search.found ? "" : "answer-miss"}`}>
          <p>{search.answer || (search.found ? "Here's what matched." : "No confident match.")}</p>
          <small>
            Searched for “{search.query}” in {(search.latencyMs / 1000).toFixed(1)} s
          </small>
        </div>
      )}

      {search.clips.length > 0 && (
        <ol className="result-list">
          {search.clips.map((c, i) => {
            const status = STATUS[String(c.verified ?? null)];
            const yt = youtubeId(c.video_id);
            const playing = c.video_id === activeId;
            return (
              <li key={`${c.video_id}-${c.start}`} className={`result result-${status.cls}`}>
                <button className="result-time" onClick={() => onPlay(c.video_id, c.best_start)}>
                  <span className="rank">{i + 1}</span>
                  {formatTime(c.best_start)}
                </button>
                <div className="result-body">
                  <p className="result-text">{c.reason || c.text}</p>
                  <p className="result-meta">
                    <span className={`badge badge-${status.cls}`}>{status.text}</span>
                    <span>
                      {title(c.video_id)}
                      {playing ? "" : ", opens this video"}
                    </span>
                  </p>
                </div>
                <div className="result-actions">
                  <button className="btn btn-small" onClick={() => onPlay(c.video_id, c.best_start)}>
                    Play
                  </button>
                  {yt && (
                    <a
                      className="link"
                      href={`https://youtu.be/${yt}?t=${Math.floor(c.best_start)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open on YouTube
                    </a>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}

      {search.status === "running" && search.clips.length === 0 && <p className="hint">Looking through the index…</p>}
    </section>
  );
}
