import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Aside } from "../Aside";
import type { ProjectEntry, SessionItem } from "../types";
import { formatStartTime, projectName, sessionTitle, useProjectFinishNotifications } from "../lib/helpers";

export function HomePage() {
  const [task, setTask] = useState("");
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const navigate = useNavigate();

  const queuedCount = sessions.filter(
    (session) => session.status === "queued",
  ).length;
  const runningCount = sessions.filter(
    (session) => session.status === "running",
  ).length;
  const pausedCount = sessions.filter(
    (session) => session.status === "paused",
  ).length;
  const completedCount = sessions.filter(
    (session) => session.status === "completed",
  ).length;

  useProjectFinishNotifications(sessions, projects);

  const fetchSessions = async () => {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    setSessions((await res.json()) as SessionItem[]);
  };

  const fetchProjects = async () => {
    const res = await fetch(`/api/projects`);
    if (!res.ok) return;
    const entries = (await res.json()) as ProjectEntry[];
    setProjects(entries);
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
  }, []);

  return (
    <>
      <div className="app-shell">
        <Aside />

        <section className="main-panel">
          <header className="mobile-topbar d-lg-none d-flex align-items-center justify-content-between px-3 py-2">
            <div className="brand compact d-flex align-items-center gap-2">
              <div className="brand-mark">&gt;_</div>
              <strong>LocalAgent</strong>
            </div>
            <button className="btn btn-light btn-icon" aria-label="Open menu">
              ☰
            </button>
          </header>

          <div className="content-wrap">
            <section className="hero-section">
              <h1>Start a new session</h1>
              <p className="text-secondary mb-3">
                Describe what you'd like the agent to work on.
              </p>

              <form
                className="start-card card"
                onSubmit={(e) => {
                  e.preventDefault();
                  void startSession();
                }}
              >
                <div className="task-editor position-relative">
                  <textarea
                    className="form-control"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    placeholder="What would you like the agent to do?"
                    rows={4}
                  />
                  <div className="editor-actions position-absolute end-0 bottom-0 p-2">
                    <button
                      type="button"
                      className="btn btn-sm border rounded-2 text-secondary bg-transparent"
                      aria-label="Attach context"
                    >
                      ⌕
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm border rounded-2 text-secondary bg-transparent"
                      aria-label="Improve prompt"
                    >
                      ✣
                    </button>
                  </div>
                </div>

                <div className="session-controls d-flex flex-column flex-md-row gap-3 align-items-md-center justify-content-between p-3">
                  <p className="text-secondary small m-0">
                    Home sessions start without a repository. Ask a research question, or say
                    <code> Create a new repo called my-app…</code> to bootstrap a project.
                    Project-specific work lives under Projects.
                  </p>
                  <button
                    className="btn btn-primary start-button"
                    type="submit"
                    disabled={!task.trim() || isSubmitting}
                  >
                    ▷ Start session
                  </button>
                </div>
              </form>
            </section>

            <section className="sessions-section">
              <div className="section-header d-flex align-items-center justify-content-between mb-3">
                <h2 className="h6 fw-bold m-0">Recent sessions</h2>
                <a
                  href="#"
                  className="text-decoration-none small fw-medium text-blue"
                >
                  View all sessions
                </a>
              </div>

              <div className="session-list list-group">
                {sessions.length === 0 ? (
                  <div className="list-group-item text-secondary small">
                    No sessions yet
                  </div>
                ) : (
                  sessions.map((s) => {
                    let statusClass = "";
                    let statusText: string = s.status;
                    if (s.status === "running") {
                      statusClass = "status-running";
                      statusText = "Running";
                    } else if (s.status === "completed") {
                      statusClass = "status-completed";
                      statusText = "Completed";
                    } else if (s.status === "queued") {
                      statusClass = "status-queued";
                      statusText = "Queued";
                    }

                    return (
                      <button
                        key={s.session_id}
                        type="button"
                        className={`session-row list-group-item list-group-item-action py-3 px-4 ${s.status}`}
                        onClick={() => navigate(`/sessions/${s.session_id}`)}
                      >
                        <div
                          className={`state-dot rounded-circle bg-${s.status === "running" ? "green" : s.status === "completed" ? "blue" : "gray"}`}
                        />
                        <div className="session-icon">&gt;_</div>
                        <div className="session-main">
                          <strong>{sessionTitle(s)}</strong>
                          <span>
                            {projectName(s.workdir)} ·{" "}
                            {formatStartTime(s.updated_at_ms)}
                          </span>
                        </div>
                        <span className={`status-pill ${statusClass}`}>
                          {statusText}
                        </span>
                        <span className="chevron">›</span>
                      </button>
                    );
                  })
                )}
              </div>
            </section>
          </div>
        </section>
      </div>
    </>
  );
}
