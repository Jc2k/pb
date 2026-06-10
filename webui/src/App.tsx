import { useEffect, useRef, useState } from "react";

/* ─── types ──────────────────────────────────────────────────── */

type AgentEvent =
  | { type: "started"; task: string; model: string; workspace: string; branch: string }
  | { type: "step_started"; step: number; max_steps: number }
  | { type: "reasoning"; content: string }
  | { type: "tool_call"; tool: string; arguments: unknown }
  | { type: "tool_result"; tool: string; result: string }
  | { type: "diff"; path: string; diff: string }
  | { type: "final"; content: string }
  | { type: "session_summary"; branch: string; commits: string }
  | { type: "error"; message: string }
  | { type: string; [key: string]: unknown };

interface EventEnvelope {
  version: string;
  event: AgentEvent;
}

interface SessionItem {
  session_id: string;
  task: string;
  running: boolean;
  branch?: string;
  workdir?: string;
  updated_at_ms: number;
}

interface SessionDetails {
  session_id: string;
  task: string;
  running: boolean;
  branch?: string;
  events: EventEnvelope[];
}

/* ─── helpers ────────────────────────────────────────────────── */

function projectName(workdir?: string): string {
  if (!workdir) return "Unknown project";
  const parts = workdir.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts[parts.length - 1] || workdir;
}

function relativeTime(ms: number): string {
  const diff = Date.now() - ms;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(ms).toLocaleDateString();
}

/* ─── diff view ──────────────────────────────────────────────── */

function DiffView({ diff }: { diff: string }) {
  return (
    <pre className="diff-block mb-0">
      {diff.split("\n").map((line, i) => {
        let cls = "";
        if (line.startsWith("+")) cls = "diff-add";
        else if (line.startsWith("-")) cls = "diff-del";
        else if (line.startsWith("@@")) cls = "diff-hunk";
        return (
          <span key={i} className={cls}>
            {line}
            {"\n"}
          </span>
        );
      })}
    </pre>
  );
}

/* ─── event item ─────────────────────────────────────────────── */

function EventItem({ envelope }: { envelope: EventEnvelope }) {
  const e = envelope.event;
  switch (e.type) {
    case "started":
      return (
        <div className="alert alert-info py-2 mb-2">
          <div className="fw-semibold">Session started</div>
          <div className="small mt-1 d-flex flex-wrap gap-3">
            <span>
              <span className="text-body-secondary">model </span>
              <code>{e.model}</code>
            </span>
            <span>
              <span className="text-body-secondary">branch </span>
              <code>{e.branch}</code>
            </span>
            <span>
              <span className="text-body-secondary">workspace </span>
              <code>{e.workspace}</code>
            </span>
          </div>
        </div>
      );

    case "step_started":
      return (
        <div className="step-marker text-body-secondary small mb-1">
          <hr className="my-1" />
          <span>
            Step {e.step} / {e.max_steps}
          </span>
        </div>
      );

    case "reasoning":
      return (
        <details className="card border-0 bg-body-secondary mb-2">
          <summary className="card-header border-0 bg-body-secondary py-2 small fw-semibold">
            Reasoning
          </summary>
          <div className="card-body py-2">
            <pre className="mb-0 small">{e.content}</pre>
          </div>
        </details>
      );

    case "tool_call":
      return (
        <details className="card border-secondary mb-2" open>
          <summary className="card-header py-2 small d-flex align-items-center gap-2">
            <span className="badge bg-secondary">tool</span>
            <code>{e.tool}</code>
          </summary>
          <div className="card-body py-2">
            <pre className="mb-0 small">{JSON.stringify(e.arguments, null, 2)}</pre>
          </div>
        </details>
      );

    case "tool_result":
      return (
        <details className="card border-0 bg-body-tertiary mb-2">
          <summary className="card-header border-0 bg-body-tertiary py-2 small d-flex align-items-center gap-2">
            <span className="badge bg-light text-dark">result</span>
            <code>{e.tool}</code>
          </summary>
          <div className="card-body py-2">
            <pre className="mb-0 small result-pre">{e.result}</pre>
          </div>
        </details>
      );

    case "diff":
      return (
        <details className="card border-info mb-2">
          <summary className="card-header py-2 small d-flex align-items-center gap-2">
            <span className="badge bg-info text-dark">diff</span>
            <code>{e.path}</code>
          </summary>
          <div className="card-body p-0 overflow-auto">
            <DiffView diff={e.diff} />
          </div>
        </details>
      );

    case "final":
      return (
        <div className="alert alert-success py-2 mb-2">
          <div className="fw-semibold mb-1">Final response</div>
          <pre className="mb-0 small">{e.content}</pre>
        </div>
      );

    case "session_summary":
      return (
        <div className="alert alert-success py-2 mb-2">
          <div className="fw-semibold mb-1">
            Session complete &middot; <code>{e.branch}</code>
          </div>
          <pre className="mb-0 small">{e.commits}</pre>
        </div>
      );

    case "error":
      return (
        <div className="alert alert-danger py-2 mb-2">
          <div className="fw-semibold mb-1">Error</div>
          {String(e.message)}
        </div>
      );

    default:
      return (
        <div className="text-body-secondary small mb-1">
          <code>{e.type}</code>
        </div>
      );
  }
}

/* ─── session card ───────────────────────────────────────────── */

function SessionCard({
  session,
  selected,
  onClick,
}: {
  session: SessionItem;
  selected: boolean;
  onClick: () => void;
}) {
  let badge: React.ReactNode;
  if (session.running) {
    badge = (
      <span className="badge bg-primary d-flex align-items-center gap-1">
        <span
          className="spinner-border spinner-border-sm"
          style={{ width: "0.6rem", height: "0.6rem" }}
        />
        In&nbsp;progress
      </span>
    );
  } else if (session.branch) {
    badge = <span className="badge bg-success">{session.branch}</span>;
  } else {
    badge = <span className="badge bg-secondary">Done</span>;
  }

  return (
    <button
      type="button"
      className={`list-group-item list-group-item-action py-2${selected ? " active" : ""}`}
      onClick={onClick}
    >
      <div className="d-flex justify-content-between align-items-start gap-2">
        <div className="fw-semibold text-truncate flex-grow-1 small">{session.task}</div>
        {badge}
      </div>
      <div className="small mt-1 text-body-secondary">
        <span className="me-2">{projectName(session.workdir)}</span>
        <span>{relativeTime(session.updated_at_ms)}</span>
      </div>
    </button>
  );
}

/* ─── main app ───────────────────────────────────────────────── */

import type React from "react";

export default function App() {
  const [task, setTask] = useState("");
  const [workdir, setWorkdir] = useState("");
  const [branch, setBranch] = useState("main");
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [projects, setProjects] = useState<string[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [sessionRunning, setSessionRunning] = useState(false);
  const [followUp, setFollowUp] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const feedRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  const fetchSessions = async () => {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const data = (await res.json()) as SessionItem[];
    setSessions(data);
    setSelectedId((prev) => {
      if (prev) {
        const s = data.find((x) => x.session_id === prev);
        if (s) setSessionRunning(s.running);
      }
      return prev;
    });
  };

  const fetchProjects = async () => {
    const res = await fetch("/api/projects");
    if (!res.ok) return;
    setProjects((await res.json()) as string[]);
  };

  const openEvents = (id: string) => {
    if (sourceRef.current) sourceRef.current.close();
    const src = new EventSource(`/api/sessions/${id}/events`);
    sourceRef.current = src;
    src.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data) as EventEnvelope;
        setEvents((prev) => [...prev, parsed]);
      } catch (err) {
        console.error(err);
      }
    };
    src.onerror = () => src.close();
  };

  const selectSession = async (id: string) => {
    const res = await fetch(`/api/sessions/${id}`);
    if (!res.ok) return;
    const details = (await res.json()) as SessionDetails;
    setSelectedId(details.session_id);
    setEvents(details.events);
    setSessionRunning(details.running);
    atBottomRef.current = true;
    openEvents(details.session_id);
  };

  const startSession = async () => {
    if (!task.trim()) return;
    setIsSubmitting(true);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          workdir: workdir.trim() || undefined,
          branch: branch.trim() || "main",
        }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { session_id: string };
      setTask("");
      setEvents([]);
      setSelectedId(data.session_id);
      setSessionRunning(true);
      atBottomRef.current = true;
      openEvents(data.session_id);
      await fetchSessions();
    } finally {
      setIsSubmitting(false);
    }
  };

  const continueSession = async () => {
    if (!selectedId || !followUp.trim()) return;
    await fetch(`/api/sessions/${selectedId}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: followUp.trim() }),
    });
    setFollowUp("");
    setSessionRunning(true);
  };

  const onFeedScroll = () => {
    const el = feedRef.current;
    if (!el) return;
    atBottomRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 80;
  };

  useEffect(() => {
    if (atBottomRef.current && feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [events]);

  useEffect(() => {
    fetchSessions();
    fetchProjects();
    const timer = window.setInterval(fetchSessions, 5000);
    return () => {
      window.clearInterval(timer);
      sourceRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedSession = sessions.find((s) => s.session_id === selectedId);

  return (
    <>
      <nav className="navbar bg-body-secondary border-bottom px-3 py-2 mb-3">
        <span className="navbar-brand fw-bold mb-0 h5">pb</span>
        {selectedSession && (
          <span className="navbar-text small text-body-secondary">
            {projectName(selectedSession.workdir)}
            {" · "}
            {selectedSession.running ? (
              <span className="text-primary">in progress</span>
            ) : selectedSession.branch ? (
              <code>{selectedSession.branch}</code>
            ) : (
              "done"
            )}
          </span>
        )}
      </nav>

      <div className="container-fluid px-3 pb-4">
        <div className="row g-3">
          {/* ── left column: form + session list ── */}
          <div className="col-lg-4 col-xl-3 d-flex flex-column gap-3">
            <div className="card">
              <div className="card-body d-flex flex-column gap-2">
                <textarea
                  className="form-control"
                  rows={4}
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="Describe the task…"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void startSession();
                  }}
                />
                <div>
                  <input
                    className="form-control"
                    list="projects-list"
                    value={workdir}
                    onChange={(e) => setWorkdir(e.target.value)}
                    placeholder="Project folder (optional)"
                  />
                  <datalist id="projects-list">
                    {projects.map((p) => (
                      <option key={p} value={p} />
                    ))}
                  </datalist>
                </div>
                <input
                  className="form-control"
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                  placeholder="Branch"
                />
                <button
                  className="btn btn-primary"
                  onClick={() => void startSession()}
                  disabled={!task.trim() || isSubmitting}
                >
                  {isSubmitting ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" />
                      Starting…
                    </>
                  ) : (
                    "Start task"
                  )}
                </button>
              </div>
            </div>

            <div className="list-group">
              {sessions.length === 0 ? (
                <div className="list-group-item text-body-secondary small">No sessions yet</div>
              ) : (
                sessions.map((s) => (
                  <SessionCard
                    key={s.session_id}
                    session={s}
                    selected={s.session_id === selectedId}
                    onClick={() => void selectSession(s.session_id)}
                  />
                ))
              )}
            </div>
          </div>

          {/* ── right column: event feed ── */}
          <div className="col-lg-8 col-xl-9">
            {!selectedId ? (
              <div className="d-flex align-items-center justify-content-center text-body-secondary h-100 py-5">
                Select a session to view its timeline
              </div>
            ) : (
              <div className="card">
                <div className="card-header d-flex justify-content-between align-items-center gap-2">
                  <div className="fw-semibold text-truncate flex-grow-1">
                    {selectedSession?.task ?? selectedId}
                  </div>
                  {selectedSession && (
                    <div className="small text-body-secondary flex-shrink-0">
                      {projectName(selectedSession.workdir)}
                    </div>
                  )}
                </div>
                <div
                  className="card-body event-feed overflow-auto"
                  ref={feedRef}
                  onScroll={onFeedScroll}
                >
                  {events.length === 0 ? (
                    <div className="text-body-secondary small">Waiting for events…</div>
                  ) : (
                    events.map((env, i) => <EventItem key={i} envelope={env} />)
                  )}
                </div>
                {!sessionRunning && selectedId && (
                  <div className="card-footer">
                    <div className="input-group">
                      <textarea
                        className="form-control"
                        rows={2}
                        value={followUp}
                        onChange={(e) => setFollowUp(e.target.value)}
                        placeholder="Follow-up task…"
                      />
                      <button
                        className="btn btn-outline-primary"
                        onClick={() => void continueSession()}
                        disabled={!followUp.trim()}
                      >
                        Continue
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
