import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Aside } from "../Aside";
import type { EventEnvelope, SessionDetails } from "../types";
import { SCROLL_THRESHOLD } from "../lib/constants";
import { formatStartTime, groupToolEvents, sessionTitle } from "../lib/helpers";
import {
  DrawerPanel,
  InitialUserMessage,
  MessageBubble,
  TodoDrawer,
  ToolDrawerSummary,
  ToolGroupBubble,
} from "../components/Session";
import {
  buildTodoTasks,
  buildToolSummaries,
  chatEventsWithOnlyLatestStep,
  latestAssistantProfile,
} from "../lib/sessionUtils";

export function SessionPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [sessionRunning, setSessionRunning] = useState(false);
  const [followUp, setFollowUp] = useState("");
  const [answer, setAnswer] = useState("");
  const [shareMessage, setShareMessage] = useState("");
  const sourceRef = useRef<EventSource | null>(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  const openEvents = (id: string) => {
    if (sourceRef.current) sourceRef.current.close();
    const src = new EventSource(`/api/sessions/${id}/events`);
    sourceRef.current = src;
    src.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data) as EventEnvelope;
        setEvents((prev) => [...prev, parsed]);
        if (parsed.event.type === "session_title") {
          setSession((current) => current ? { ...current, title: parsed.event.title } : current);
        }
        if (parsed.event.type === "user_question") {
          setSessionRunning(false);
        } else if (parsed.event.type === "user_answer") {
          setSessionRunning(true);
        } else if (
          parsed.event.type === "final" ||
          parsed.event.type === "session_summary"
        ) {
          setSessionRunning(false);
        }
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
    setSession(details);
    setEvents(details.events);
    setSessionRunning(details.running);
    setAnswer("");
  };

  const continueSession = async () => {
    if (!followUp.trim()) return;
    await fetch(`/api/sessions/${sessionId}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: followUp.trim() }),
    });
    setFollowUp("");
    setSessionRunning(false);
  };

  const resumeSession = async () => {
    await fetch(`/api/sessions/${sessionId}/resume`, { method: "POST" });
    setSessionRunning(false);
  };

  const shareSession = async () => {
    if (!session) return;
    const shareUrl = new URL(`/sessions/${session.session_id}`, window.location.origin).toString();
    const shareData: ShareData = {
      title: `pb session: ${sessionTitle(session)}`,
      text: `View this pb session: ${sessionTitle(session)}`,
      url: shareUrl,
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
        setShareMessage("Shared");
      } else {
        await navigator.clipboard.writeText(shareUrl);
        setShareMessage("Link copied");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      console.error(err);
      setShareMessage("Unable to share");
    }
  };

  const answerQuestion = async (choice?: string) => {
    const selectedAnswer = choice ?? answer.trim();
    if (!selectedAnswer || !session?.pending_question) return;
    await fetch(`/api/sessions/${sessionId}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question_id: session.pending_question.question_id,
        answer: selectedAnswer,
      }),
    });
    setAnswer("");
    setSessionRunning(true);
  };

  const onChatScroll = () => {
    const el = chatRef.current;
    if (!el) return;
    atBottomRef.current =
      el.scrollTop + el.clientHeight >= el.scrollHeight - SCROLL_THRESHOLD;
  };

  useEffect(() => {
    if (atBottomRef.current && chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [events]);

  useEffect(() => {
    atBottomRef.current = true;
    void fetchSession().then(() => openEvents(sessionId));
    return () => sourceRef.current?.close();
  }, [sessionId]);

  useEffect(() => {
    if (!shareMessage) return;
    const timer = window.setTimeout(() => setShareMessage(""), 2400);
    return () => window.clearTimeout(timer);
  }, [shareMessage]);

  const isRunning = session?.status === "running" || false;

  if (!session) {
    return (
      <div className="app-shell">
        <Aside />
        <section className="session-panel">
          <header className="session-header">
            <span className="navbar-brand fw-bold mb-0 d-flex align-items-center gap-2">
              <img
                src="/logo.svg"
                alt="pb"
                width="32"
                height="32"
                style={{ borderRadius: "6px" }}
              />
              pb
            </span>
          </header>
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Aside />

      <section className="session-panel">
        <header className="session-header">
          <button
            type="button"
            className="btn btn-link d-lg-none p-0 text-body"
            onClick={() => navigate("/")}
          >
            <i className="bi bi-chevron-left fs-4"></i>
          </button>
          <div className="session-icon d-none d-sm-grid">
            <i className="bi bi-terminal"></i>
          </div>
          <div className="min-w-0 flex-grow-1">
            <h1>{sessionTitle(session)}</h1>
            <div className="status-line">
              {isRunning && <span className="live-dot" />}
              <span>
                {session.status === "running"
                  ? "Running"
                  : session.status === "queued"
                    ? "Queued"
                    : session.status === "paused"
                      ? session.pending_question
                        ? "Waiting for answer"
                        : "Paused"
                      : session.status === "failed"
                        ? "Failed"
                        : "Completed"}
              </span>
              {session.updated_at_ms && (
                <>
                  <span className="dot-sep"></span>
                  <span>{formatStartTime(session.updated_at_ms)}</span>
                </>
              )}
              <span className="dot-sep d-none d-sm-inline"></span>
              <span className="d-none d-sm-inline">
                Model: {session.branch}
              </span>
            </div>
          </div>
          <div className="share-action">
            <button
              type="button"
              className="btn btn-light rounded-pill d-inline-flex align-items-center"
              onClick={shareSession}
              aria-label="Share this session"
            >
              <i className="bi bi-box-arrow-up me-sm-2"></i>
              <span className="d-none d-sm-inline">Share</span>
            </button>
            {shareMessage && (
              <span className="share-status small text-body-secondary" role="status">
                {shareMessage}
              </span>
            )}
          </div>
          <button
            className="btn btn-danger rounded-pill"
            onClick={() => window.location.reload()}
          >
            <i className="bi bi-stop-fill me-1"></i>Stop
          </button>
        </header>

        <div className="session-layout">
          <main className="chat-stream" ref={chatRef} onScroll={onChatScroll}>
            {events.length === 0 ? (
              <InitialUserMessage
                task={session.task}
                timestampMs={session.updated_at_ms}
              />
            ) : (
              groupToolEvents(chatEventsWithOnlyLatestStep(events)).map(
                (grouped, i) => {
                  if ((grouped as any).type === "tool_group") {
                    const tc = (grouped as any).toolCalls;
                    return (
                      <ToolGroupBubble
                        key={i}
                        toolCalls={tc}
                        toolResults={(grouped as any).toolResults}
                      />
                    );
                  }
                  return (
                    <MessageBubble
                      key={i}
                      envelope={grouped as EventEnvelope}
                      activityProfile={latestAssistantProfile(events)}
                    />
                  );
                },
              )
            )}
          </main>

          <aside className="tool-drawer d-none d-xl-block">
            <DrawerPanel
              title="Tools"
              icon="bi bi-tools"
              count={events.filter((e) => e.event.type === "tool_call").length}
            >
              {(() => {
                const toolEvents = events.filter(
                  (e) =>
                    e.event.type === "tool_call" ||
                    e.event.type === "tool_result",
                );

                if (toolEvents.length === 0) {
                  return (
                    <div className="empty-detail compact">
                      <i className="bi bi-file-earmark-code"></i>
                      <h3>No tools yet</h3>
                      <p>
                        Inspect files, commands, and outputs without cluttering
                        the main session.
                      </p>
                    </div>
                  );
                }

                const summaries = buildToolSummaries(events);

                return summaries.map((summary) => (
                  <ToolDrawerSummary key={summary.toolName} summary={summary} />
                ));
              })()}
            </DrawerPanel>

            <DrawerPanel
              title="Tasks"
              icon="bi bi-check2-square"
              count={buildTodoTasks(events).length}
              defaultOpen={false}
            >
              <TodoDrawer tasks={buildTodoTasks(events)} />
            </DrawerPanel>
          </aside>
        </div>

        {session.status === "paused" && session.pending_question ? (
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void answerQuestion();
            }}
          >
            <button className="btn btn-light rounded-circle" type="button">
              <i className="bi bi-plus-lg"></i>
            </button>
            {session.pending_question.choices?.length ? (
              <div className="d-flex gap-2 flex-grow-1 flex-wrap">
                {session.pending_question.choices.map((choice) => (
                  <button
                    key={choice}
                    className="btn btn-warning"
                    type="button"
                    onClick={() => void answerQuestion(choice)}
                  >
                    {choice}
                  </button>
                ))}
              </div>
            ) : (
              <>
                <input
                  className="form-control"
                  rows={2}
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Answer the planning question…"
                />
                <button
                  className="btn btn-warning rounded-circle"
                  type="submit"
                  disabled={!answer.trim()}
                >
                  <i className="bi bi-check-lg"></i>
                </button>
              </>
            )}
          </form>
        ) : session.status === "paused" ? (
          <footer className="composer">
            <div className="flex-grow-1 small text-body-secondary">
              This session was restored after a daemon restart and is paused
              until you resume it.
            </div>
            <button
              className="btn btn-warning"
              onClick={() => void resumeSession()}
            >
              Resume
            </button>
          </footer>
        ) : !isRunning && session.status === "completed" ? (
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void continueSession();
            }}
          >
            <button className="btn btn-light rounded-circle" type="button">
              <i className="bi bi-plus-lg"></i>
            </button>
            <input
              className="form-control"
              value={followUp}
              onChange={(e) => setFollowUp(e.target.value)}
              placeholder="Follow-up task…"
            />
            <button
              className="btn btn-primary rounded-circle"
              type="submit"
              disabled={!followUp.trim()}
            >
              <i className="bi bi-arrow-up"></i>
            </button>
          </form>
        ) : null}
      </section>
    </div>
  );
}
