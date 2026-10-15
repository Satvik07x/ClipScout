// All calls to the Python server live here. Components never call fetch directly.

// Locally the Vite proxy forwards /api to the server. When the site is hosted somewhere else than the API
// (Vercel + Render), VITE_API_URL holds the server's address.
export const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

// A random id per browser. In the public demo it keeps each visitor's videos separate (not a login).
let fallbackId = null;
function visitorId() {
  try {
    let id = localStorage.getItem("clipscout-visitor");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("clipscout-visitor", id);
    }
    return id;
  } catch {
    fallbackId ??= crypto.randomUUID();
    return fallbackId;
  }
}

async function request(path, options = {}) {
  const res = await fetch(API + path, { ...options, headers: { ...options.headers, "X-Visitor": visitorId() } });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : body.detail[0]?.msg || detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

export const getConfig = () => request("/api/config");

export const listVideos = async () => {
  const list = await request("/api/videos");
  return list.map((v) => ({ ...v, file_url: v.file_url ? API + v.file_url : "" }));
};

export const addYouTube = (url, minutes, title) =>
  request("/api/videos/youtube", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, minutes: Number(minutes), title }),
  });

export const uploadVideo = (file, title, minutes) => {
  const form = new FormData();
  form.append("file", file);
  form.append("title", title);
  form.append("minutes", String(minutes || 0));
  return request("/api/videos/upload", { method: "POST", body: form });
};

export const resumeIndexing = (id) => request(`/api/videos/${id}/index`, { method: "POST" });

export const deleteVideo = (id) => request(`/api/videos/${id}`, { method: "DELETE" });

// Search streams one event per agent step (Server-Sent Events). Returns a function that cancels it.
export function streamSearch({ q, videoId, verify }, onEvent) {
  const params = new URLSearchParams({ q, verify: String(verify), visitor: visitorId() });
  if (videoId) params.set("video_id", videoId);
  const source = new EventSource(`${API}/api/search?${params}`);
  source.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    if (event.type === "done" || event.type === "error") source.close(); // stop the browser auto-reconnecting
    onEvent(event);
  };
  source.onerror = () => {
    if (source.readyState !== EventSource.CLOSED) {
      source.close();
      onEvent({ type: "error", message: "Lost connection to the server. If it was asleep, wait a minute and try again." });
    }
  };
  return () => source.close();
}

export function formatTime(seconds) {
  const t = Math.max(0, Math.floor(seconds));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${String(m).padStart(2, "0")}:${s}`;
}
