import { useEffect, useRef, useState } from "react";
import { addYouTube, uploadVideo } from "../api.js";

export default function AddVideoDialog({ onClose, onAdded }) {
  const [mode, setMode] = useState("youtube");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [minutes, setMinutes] = useState("");
  const [title, setTitle] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef(null);

  useEffect(() => {
    dialog.current?.showModal();
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    try {
      const video = mode === "youtube" ? await addYouTube(url, minutes, title) : await uploadVideo(file, title, minutes);
      onAdded(video);
    } catch (err) {
      setError(err.message);
      setSaving(false);
    }
  };

  const canSubmit = !saving && (mode === "youtube" ? url && Number(minutes) > 0 : file);

  return (
    <dialog ref={dialog} className="dialog" onClose={onClose} onCancel={onClose}>
      <form onSubmit={submit}>
        <h2>Add a video</h2>

        <div className="segmented" role="tablist">
          <button type="button" role="tab" aria-selected={mode === "youtube"} onClick={() => setMode("youtube")}>
            YouTube link
          </button>
          <button type="button" role="tab" aria-selected={mode === "file"} onClick={() => setMode("file")}>
            Upload a file
          </button>
        </div>

        {mode === "youtube" ? (
          <>
            <label className="field">
              <span>Public YouTube link</span>
              <input
                type="url"
                required
                autoFocus
                placeholder="https://youtu.be/..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
            </label>
            <label className="field">
              <span>Length in minutes</span>
              <input
                type="number"
                required
                min="0.1"
                step="0.1"
                placeholder="e.g. 8.5"
                value={minutes}
                onChange={(e) => setMinutes(e.target.value)}
              />
              <small>Shown under the YouTube player. Used to split the video into pieces.</small>
            </label>
          </>
        ) : (
          <>
            <label className="field">
              <span>Video file</span>
              <input type="file" accept="video/*" required onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
            <label className="field">
              <span>Length in minutes (only if we can't read it)</span>
              <input type="number" min="0" step="0.1" value={minutes} onChange={(e) => setMinutes(e.target.value)} />
            </label>
          </>
        )}

        <label className="field">
          <span>Title (optional)</span>
          <input placeholder="e.g. Gate camera, Monday" value={title} onChange={(e) => setTitle(e.target.value)} />
        </label>

        <p className="note">Gemini watches the video once. Searches after that never re-watch it.</p>
        {error && <p className="form-error">{error}</p>}

        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={!canSubmit}>
            {saving ? (mode === "file" ? "Uploading…" : "Adding…") : "Add and index"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
