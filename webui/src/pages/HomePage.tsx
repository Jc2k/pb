import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageShell } from "../components/PageShell";
import { IntentControl } from "../components/IntentControl";
import { GoalStartSheet } from "../components/GoalStartSheet";
import { VoiceInputButton } from "../components/VoiceInputButton";
import {
  AttachmentButton,
  ImageAttachments,
  sessionCounts,
  type SessionFilter,
  SessionFilters,
  SessionRows,
  UsageMetrics,
} from "../components/SessionDashboard";
import type { ComposerMode, SessionAttachment } from "../types";
import { relativeTime } from "../lib/helpers";
import {
  isAbortError,
  LatestRequest,
  useProjectSessionData,
} from "../lib/hooks";
import { apiErrorMessage } from "../lib/integrationConfig";
import { parseSessionResponseJson } from "../lib/eventContract";

export function HomePage() {
  const [task, setTask] = useState("");
  const [intent, setIntent] = useState<ComposerMode>("discuss");
  const [goalOpen, setGoalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [voiceInputActive, setVoiceInputActive] = useState(false);
  const [images, setImages] = useState<SessionAttachment[]>([]);
  const [filter, setFilter] = useState<SessionFilter>("all");
  const { sessions, projects, overallUsage } = useProjectSessionData();
  const startRequest = useRef(new LatestRequest());
  const navigate = useNavigate();

  useEffect(() => () => startRequest.current.abort(), []);

  const counts = useMemo(() => sessionCounts(sessions), [sessions]);
  const visibleSessions = filter === "all"
    ? sessions
    : sessions.filter((session) => session.status === filter);
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
    setSubmitError("");
    const controller = startRequest.current.start();
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          intent,
          attachments: images,
        }),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Could not start the session"),
        );
      }
      const data = parseSessionResponseJson(await res.text());
      if (!startRequest.current.owns(controller)) return;
      navigate(`/sessions/${data.session_id}`);
    } catch (error) {
      if (isAbortError(error) || !startRequest.current.owns(controller)) return;
      setSubmitError(
        error instanceof Error ? error.message : "Could not start the session",
      );
    } finally {
      if (startRequest.current.owns(controller)) setIsSubmitting(false);
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
              {submitError && (
                <div className="alert alert-danger" role="alert">
                  {submitError}
                </div>
              )}
              <textarea
                className="form-control composer-input"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                rows={4}
                placeholder="Describe the work…"
                aria-label="Describe the work"
                readOnly={voiceInputActive}
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
                  <VoiceInputButton
                    value={task}
                    onValueChange={setTask}
                    disabled={isSubmitting}
                    onActiveChange={setVoiceInputActive}
                  />
                  <button
                    className="btn btn-primary send-btn"
                    type="submit"
                    disabled={!task.trim() || isSubmitting || voiceInputActive}
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
              paginationKey={filter}
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
                  usage={overallUsage.total}
                  todaysUsage={overallUsage.today}
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
