import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageShell } from "../components/PageShell";
import { IntentControl } from "../components/IntentControl";
import { GoalStartSheet } from "../components/GoalStartSheet";
import {
  AttachmentButton,
  ImageAttachments,
  sessionCounts,
  type SessionFilter,
  SessionFilters,
  SessionRows,
  UsageMetrics,
} from "../components/SessionDashboard";
import type {
  ComposerMode,
  ProjectUsageStats,
  SessionAttachment,
} from "../types";
import { relativeTime, usageStatsForToday } from "../lib/helpers";
import { useProjectSessionData } from "../lib/hooks";
import { metricEnergyJoules, metricRuntimeMs } from "../lib/energy";

export function HomePage() {
  const [task, setTask] = useState("");
  const [intent, setIntent] = useState<ComposerMode>("discuss");
  const [goalOpen, setGoalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [images, setImages] = useState<SessionAttachment[]>([]);
  const [filter, setFilter] = useState<SessionFilter>("all");
  const { sessions, projects } = useProjectSessionData();
  const navigate = useNavigate();

  const counts = useMemo(() => sessionCounts(sessions), [sessions]);
  const visibleSessions = filter === "all"
    ? sessions
    : sessions.filter((session) => session.status === filter);
  const usage = useMemo<ProjectUsageStats>(
    () =>
      sessions.reduce<ProjectUsageStats>((totals, session) => {
        if (!session.metrics) return totals;
        totals.tokens += session.metrics.prompt_tokens +
          session.metrics.generated_tokens;
        totals.runtime_ms += metricRuntimeMs(session.metrics);
        totals.tool_calls += session.metrics.tool_calls;
        const energy = metricEnergyJoules(session.metrics);
        if (energy !== undefined) {
          totals.energy_joules = (totals.energy_joules ?? 0) + energy;
          totals.energy_kwh = totals.energy_joules / 3_600_000;
        }
        return totals;
      }, { tokens: 0, runtime_ms: 0, tool_calls: 0 }),
    [sessions],
  );
  const todaysUsage = useMemo(() => usageStatsForToday(sessions), [sessions]);
  const runningSession = sessions.find((session) =>
    session.status === "running"
  );
  const lastActive = sessions[0]?.updated_at_ms
    ? relativeTime(sessions[0].updated_at_ms)
    : "No activity yet";

  const startSession = async () => {
    if (!task.trim()) return;
    if (intent === "goal") {
      setGoalOpen(true);
      return;
    }
    setIsSubmitting(true);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          intent,
          attachments: images,
        }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { session_id: string };
      navigate(`/sessions/${data.session_id}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <PageShell
      pageClassName="project-detail-shell home-detail-shell"
      contentClassName="project-detail-wrap home-detail-wrap"
    >
      <div className="project-layout home-layout">
        <section className="project-content">
          <section className="home-hero">
            <h1>New home session</h1>
            <p>
              Use this for research, planning, or bootstrapping before a project
              exists.
            </p>
          </section>

          <form
            className="card soft-card composer-card home-composer-card"
            onSubmit={(e) => {
              e.preventDefault();
              void startSession();
            }}
          >
            <div className="card-body">
              <textarea
                className="form-control composer-input"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                rows={4}
                placeholder="Describe the work…"
                aria-label="Describe the work"
              />
              <ImageAttachments images={images} setImages={setImages} />
              <div className="composer-actions">
                <div className="quick-actions">
                  <div className="quick-action-row">
                    <IntentControl
                      intent={intent}
                      onChange={setIntent}
                      disabled={isSubmitting}
                    />
                    <button
                      className="btn btn-light"
                      type="button"
                      onClick={() => {
                        setIntent("discuss");
                        setTask("Research ");
                      }}
                    >
                      <i className="bi bi-search"></i> Research
                    </button>
                  </div>
                  <div className="quick-action-row secondary-quick-actions">
                    <button
                      className="btn btn-light optional-action"
                      type="button"
                      onClick={() => {
                        setIntent("deliver");
                        setTask("Create a new repo called ");
                      }}
                    >
                      <i className="bi bi-chat-square-plus"></i> Create repo
                    </button>
                    <button
                      className="btn btn-light optional-action"
                      type="button"
                      onClick={() => {
                        setIntent("deliver");
                        setTask("Fix error ");
                      }}
                    >
                      <i className="bi bi-tools"></i> Fix error
                    </button>
                    <button
                      className="btn btn-light optional-action"
                      type="button"
                    >
                      <span>More</span>
                      <i className="bi bi-chevron-down"></i>
                    </button>
                  </div>
                </div>
                <div className="chat-submit-actions">
                  <AttachmentButton setImages={setImages} images={images} />
                  <button
                    className="btn btn-primary send-btn"
                    type="submit"
                    disabled={!task.trim() || isSubmitting}
                    aria-label="Start home session"
                  >
                    <i className="bi bi-arrow-up"></i>
                  </button>
                </div>
              </div>
            </div>
          </form>

          <section className="sessions-section project-sessions-panel">
            <div className="recent-sessions-heading">
              <h2>Recent sessions</h2>
              <span>{counts.all} total</span>
            </div>
            {counts.all > 0
              ? (
                <SessionFilters
                  filter={filter}
                  counts={counts}
                  onFilterChange={setFilter}
                />
              )
              : null}
            <SessionRows
              sessions={visibleSessions}
              emptyText="No sessions match this filter."
              onOpenSession={(session) =>
                navigate(`/sessions/${session.session_id}`)}
            />
          </section>
        </section>

        <aside className="project-aside home-aside">
          <section className="card soft-card aside-card">
            <div className="card-body">
              <div className="card-title-row">
                <h2>Usage</h2>
                <i className="bi bi-info-circle"></i>
              </div>
              <div className="info-list usage-list">
                <UsageMetrics
                  usage={usage}
                  todaysUsage={todaysUsage}
                  scopeLabel="Across all sessions"
                />
              </div>
            </div>
          </section>
          <section className="card soft-card aside-card">
            <div className="card-body">
              <h2>Overview</h2>
              <div className="info-list key-value-list">
                <div>
                  <span>Current session</span>
                  <strong>{runningSession ? "Running" : "None running"}</strong>
                </div>
                <div>
                  <span>Queue</span>
                  <strong>{counts.queued} waiting</strong>
                </div>
                <div>
                  <span>Last active</span>
                  <strong>{lastActive}</strong>
                </div>
              </div>
            </div>
          </section>
        </aside>
      </div>
      <GoalStartSheet
        open={goalOpen}
        initialObjective={task}
        projects={projects}
        onClose={() => setGoalOpen(false)}
        onStarted={(sessionId) => navigate(`/sessions/${sessionId}`)}
      />
    </PageShell>
  );
}
