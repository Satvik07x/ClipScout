import { useState } from "react";

// The agent's pipeline, in the order LangGraph runs it.
const STEPS = [
  { node: "rewrite", label: "Understand the question" },
  { node: "retrieve", label: "Search the index" },
  { node: "grade", label: "Read the matches" },
  { node: "verify", label: "Check the video frames" },
  { node: "answer", label: "Write the answer" },
];

function stepState(node, search, verifyOn) {
  if (node === "verify" && !verifyOn) return "skipped";
  const seen = search.steps.some((s) => s.node === node || (node === "answer" && s.node === "not_found"));
  if (seen) return "done";
  if (search.status !== "running") return "waiting";
  const next = STEPS.find((s) => !(s.node === "verify" && !verifyOn) && !search.steps.some((x) => x.node === s.node));
  return next?.node === node ? "active" : "waiting";
}

export default function SearchPanel({ disabled, scopeLabel, search, onSearch }) {
  const [q, setQ] = useState("");
  const [verify, setVerify] = useState(true);
  const [lastVerify, setLastVerify] = useState(true);

  const submit = (e) => {
    e.preventDefault();
    if (!q.trim() || disabled) return;
    setLastVerify(verify);
    onSearch(q.trim(), verify);
  };

  const retries = search.steps.filter((s) => s.node === "rewrite").length - 1;
  const showPipeline = search.status !== "idle";

  return (
    <section className="search">
      <form className="search-form" onSubmit={submit}>
        <input
          className="search-input"
          aria-label="Search the video"
          placeholder={disabled ? "Waiting for a video to finish indexing" : "e.g. red bike wala aadmi gate pe kab aaya?"}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          disabled={disabled}
        />
        <button className="btn btn-primary btn-search" disabled={disabled || !q.trim() || search.status === "running"}>
          {search.status === "running" ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="search-options">
        <span className="scope">Searching in {scopeLabel}</span>
        <label className="toggle">
          <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} />
          <span>Check matches against the video</span>
          <small>slower, more accurate</small>
        </label>
      </div>

      {showPipeline && (
        <div className="pipeline">
          <ol className="steps">
            {STEPS.map((s) => (
              <li key={s.node} className={`step step-${stepState(s.node, search, lastVerify)}`}>
                <span className="step-mark" aria-hidden="true" />
                <span className="step-label">{s.label}</span>
              </li>
            ))}
          </ol>
          {retries > 0 && <p className="retry-note">Nothing matched at first, so the agent rephrased the question and tried again.</p>}
          {search.steps.length > 0 && (
            <details className="trace">
              <summary>How it searched</summary>
              <ul>
                {search.steps.map((s, i) => (
                  <li key={i}>{s.text}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
