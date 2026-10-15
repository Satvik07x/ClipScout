import { useEffect, useRef, useState } from "react";
import { formatTime } from "../api.js";

// Pick a ruler step so the timeline shows at most ~6 labels, whatever the video length.
function tickStep(duration) {
  const steps = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  return steps.find((s) => duration / s <= 6) ?? 3600;
}

function Timeline({ duration, playhead, markers, onSeek }) {
  const track = useRef(null);
  const step = tickStep(duration);
  const ticks = [];
  for (let t = 0; t <= duration; t += step) ticks.push(t);
  const pct = (t) => `${Math.min(100, (t / duration) * 100)}%`;

  const clickTrack = (e) => {
    const box = track.current.getBoundingClientRect();
    onSeek(((e.clientX - box.left) / box.width) * duration);
  };

  return (
    <div className="timeline">
      <div className="track" ref={track} onClick={clickTrack} title="Click to jump">
        {markers.map((m) => (
          <button
            key={`${m.rank}-${m.start}`}
            className={`cue cue-${m.verified === true ? "verified" : m.verified === false ? "rejected" : "unchecked"}`}
            style={{ left: pct(m.start), width: `max(6px, ${((m.end - m.start) / duration) * 100}%)` }}
            onClick={(e) => {
              e.stopPropagation();
              onSeek(m.best_start);
            }}
            aria-label={`Result ${m.rank} at ${formatTime(m.best_start)}`}
          >
            <span className="cue-flag">{m.rank}</span>
          </button>
        ))}
        <span className="playhead" style={{ left: pct(playhead) }} />
      </div>
      <div className="ruler" aria-hidden="true">
        {ticks.map((t) => (
          <span key={t} className="tick" style={{ left: pct(t) }}>
            {formatTime(t)}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Player({ video, seek, clips, onSeek }) {
  const videoTag = useRef(null);
  const [current, setCurrent] = useState(0);

  // Local files: move the real <video> element. YouTube: the iframe is reloaded at the new start time.
  useEffect(() => {
    setCurrent(seek.t);
    if (videoTag.current) {
      videoTag.current.currentTime = seek.t;
      videoTag.current.play().catch(() => {});
    }
  }, [seek]);

  if (!video) {
    return (
      <section className="player player-empty">
        <p>Pick a video from the library, or search and ClipScout will open the right one.</p>
      </section>
    );
  }

  const markers = clips
    .map((c, i) => ({ ...c, rank: i + 1 }))
    .filter((c) => c.video_id === video.video_id);

  return (
    <section className="player">
      <div className="screen">
        {video.kind === "youtube" ? (
          <iframe
            key={`${video.video_id}-${seek.nonce}`}
            src={`https://www.youtube.com/embed/${video.youtube_id}?start=${Math.floor(seek.t)}&autoplay=${seek.nonce ? 1 : 0}&rel=0`}
            title={video.title}
            allow="autoplay; encrypted-media; picture-in-picture"
            allowFullScreen
          />
        ) : (
          <video
            ref={videoTag}
            src={video.file_url}
            controls
            onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
          />
        )}
      </div>
      <div className="player-bar">
        <span className="now-playing">{video.title}</span>
        <span className="timecode">
          {formatTime(current)} / {formatTime(video.duration)}
        </span>
      </div>
      {video.duration > 0 && (
        <Timeline duration={video.duration} playhead={current} markers={markers} onSeek={onSeek} />
      )}
    </section>
  );
}
