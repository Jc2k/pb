import type React from "react";
import { useEffect, useRef, useState } from "react";

/* ─── constants ──────────────────────────────────────────────── */

const SCROLL_THRESHOLD = 80;

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

interface ProjectEntry {
  name: string;
  path: string;
}

/* ─── simple router ──────────────────────────────────────────── */

function useRoute(): string {
  const [path, setPath] = useState(() => window.location.pathname);
  useEffect(() => {
    const handler = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);
  return path;
}

function navigate(to: string) {
  window.history.pushState(null, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
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
  onClick,
}: {
  session: SessionItem;
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
      className="list-group-item list-group-item-action py-2"
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

/* ─── home page (/) ──────────────────────────────────────────── */

function HomePage() {
  const [task, setTask] = useState("");
  const [workdir, setWorkdir] = useState("");
  const [branch, setBranch] = useState("main");
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const fetchSessions = async () => {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    setSessions((await res.json()) as SessionItem[]);
  };

  const fetchProjects = async () => {
    const res = await fetch("/api/projects");
    if (!res.ok) return;
    const entries = (await res.json()) as ProjectEntry[];
    setProjects(entries);
    setWorkdir((current) => current || entries[0]?.path || "");
  };

  const startSession = async () => {
    if (!task.trim() || !workdir) return;
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
      navigate(`/sessions/${data.session_id}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    void fetchSessions();
    void fetchProjects();
    const timer = window.setInterval(() => void fetchSessions(), 5000);
    return () => window.clearInterval(timer);
    // fetchSessions and fetchProjects are stable async closures defined in
    // this component; we intentionally run them only once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <nav className="navbar navbar-dark bg-dark border-bottom px-3 py-2 mb-3">
        <span className="navbar-brand fw-bold mb-0 d-flex align-items-center gap-2">
          <img src="/logo.svg" alt="pb" width="32" height="32" style={{ borderRadius: "6px" }} />
          pb
        </span>
      </nav>

      <div className="container-fluid px-3 pb-4">
        <div className="row g-3 justify-content-center">
          <div className="col-lg-6 col-xl-5 d-flex flex-column gap-3">
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
                  <select
                    className="form-select"
                    value={workdir}
                    onChange={(e) => setWorkdir(e.target.value)}
                    disabled={projects.length === 0}
                  >
                    {projects.length === 0 ? (
                      <option value="">No registered projects</option>
                    ) : (
                      projects.map((project) => (
                        <option key={project.name} value={project.path}>
                          {project.name} — {project.path}
                        </option>
                      ))
                    )}
                  </select>
                  <div className="form-text">
                    Register projects with <code>pb projects add</code>.
                  </div>
                </div>
                <input
                  className="form-control"
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                  placeholder="Branch (default: main)"
                />
                <button
                  className="btn btn-primary"
                  onClick={() => void startSession()}
                  disabled={!task.trim() || !workdir || isSubmitting}
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
                    onClick={() => navigate(`/sessions/${s.session_id}`)}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/* ─── session page (/sessions/:id) ──────────────────────────── */

function SessionPage({ sessionId }: { sessionId: string }) {
  const [session, setSession] = useState<SessionItem | null>(null);
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [sessionRunning, setSessionRunning] = useState(false);
  const [followUp, setFollowUp] = useState("");
  const sourceRef = useRef<EventSource | null>(null);
  const feedRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

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

  const fetchSession = async () => {
    const res = await fetch(`/api/sessions/${sessionId}`);
    if (!res.ok) return;
    const details = (await res.json()) as SessionDetails;
    setSession({
      session_id: details.session_id,
      task: details.task,
      running: details.running,
      branch: details.branch,
      workdir: undefined,
      updated_at_ms: 0,
    });
    setEvents(details.events);
    setSessionRunning(details.running);
  };

  const continueSession = async () => {
    if (!followUp.trim()) return;
    await fetch(`/api/sessions/${sessionId}/continue`, {
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
    atBottomRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - SCROLL_THRESHOLD;
  };

  useEffect(() => {
    if (atBottomRef.current && feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [events]);

  useEffect(() => {
    atBottomRef.current = true;
    void fetchSession().then(() => openEvents(sessionId));
    return () => sourceRef.current?.close();
    // sessionId is stable for the lifetime of this component instance.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  return (
    <>
      <nav className="navbar navbar-dark bg-dark border-bottom px-3 py-2 mb-3">
        <button
          type="button"
          className="btn btn-sm btn-outline-light me-3"
          onClick={() => navigate("/")}
        >
          ← Back
        </button>
        <span className="navbar-brand fw-bold mb-0 d-flex align-items-center gap-2">
          <img src="/logo.svg" alt="pb" width="32" height="32" style={{ borderRadius: "6px" }} />
          pb
        </span>
        {session && (
          <span className="navbar-text small text-body-secondary ms-auto">
            {session.running ? (
              <span className="text-primary">in progress</span>
            ) : session.branch ? (
              <code>{session.branch}</code>
            ) : (
              "done"
            )}
          </span>
        )}
      </nav>

      <div className="container-fluid px-3 pb-4">
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center gap-2">
            <div className="fw-semibold text-truncate flex-grow-1">
              {session?.task ?? sessionId}
            </div>
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
          {!sessionRunning && (
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
      </div>
    </>
  );
}

/* ─── main app ───────────────────────────────────────────────── */

export default function App() {
  const path = useRoute();

  const sessionMatch = path.match(/^\/sessions\/([^/]+)$/);
  if (sessionMatch) {
    return <SessionPage sessionId={sessionMatch[1]} />;
  }

  return <HomePage />;
}
