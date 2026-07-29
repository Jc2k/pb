import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Aside } from "../Aside";
import type { ComposerMode, EventEnvelope, SessionDetails } from "../types";
import { IntentControl } from "../components/IntentControl";
import { GoalAmendmentSheet } from "../components/GoalAmendmentSheet";
import { GoalDrawer } from "../components/GoalDrawer";
import {
  TaskPlanningDetails,
  TaskPlanningRecovery,
  TaskProgress,
} from "../components/TaskProgress";
import { GoalModeBanner } from "../components/GoalModeBanner";
import { GoalPlanReview } from "../components/GoalPlanReview";
import { GoalStartSheet } from "../components/GoalStartSheet";
import { goalStageLabel } from "../lib/goalUtils";
import { SCROLL_THRESHOLD } from "../lib/constants";
import {
  formatStartTime,
  groupActionEvents,
  isControllerActionEvent,
  sessionPageDocumentTitle,
  sessionTitle,
} from "../lib/helpers";
import {
  ActionDrawerItem,
  ActionGroupBubble,
  DrawerPanel,
  InitialUserMessage,
  MessageBubble,
  TodoDrawer,
} from "../components/Session";
import {
  buildActionTimeline,
  buildTodoTasks,
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
  const [runningMessage, setRunningMessage] = useState("");
  const [runningMessageError, setRunningMessageError] = useState("");
  const [intent, setIntent] = useState<ComposerMode>("discuss");
  const [goalStartOpen, setGoalStartOpen] = useState(false);
  const [goalDetailsOpen, setGoalDetailsOpen] = useState(false);
  const [goalEditOpen, setGoalEditOpen] = useState(false);
  const [editAfterPause, setEditAfterPause] = useState(false);
  const [goalBusy, setGoalBusy] = useState(false);
  const [answer, setAnswer] = useState("");
  const [shareMessage, setShareMessage] = useState("");
  const [taskRecoveryBusy, setTaskRecoveryBusy] = useState(false);
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
          const title = parsed.event.title;
          setSession((current) => current ? { ...current, title } : current);
        }
        if (
          parsed.event.type.startsWith("goal_") ||
          parsed.event.type.startsWith("workflow_") ||
          parsed.event.type === "task_plan_accepted" ||
          parsed.event.type === "task_plan_rejected" ||
          parsed.event.type === "tasks_changed" ||
          parsed.event.type === "session_summary"
        ) {
          void fetchSession();
        }
        if (parsed.event.type === "started") {
          setSessionRunning(true);
        } else if (parsed.event.type === "user_question") {
          setSessionRunning(false);
          void fetchSession();
        } else if (parsed.event.type === "user_answer") {
          setSessionRunning(true);
        } else if (
          parsed.event.type === "final" ||
          parsed.event.type === "session_summary"
        ) {
          setSessionRunning(false);
          setIntent("discuss");
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

  const continueSession = async (
    task = followUp.trim(),
    requestedIntent = intent,
    proposalId?: string,
  ) => {
    if (!task) return;
    if (requestedIntent === "goal") {
      setGoalStartOpen(true);
      return;
    }
    await fetch(`/api/sessions/${sessionId}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task,
        intent: requestedIntent,
        proposal_id: proposalId,
      }),
    });
    setFollowUp("");
    setSessionRunning(false);
  };

  const sendRunningMessage = async () => {
    const message = runningMessage.trim();
    if (!message) return;
    setRunningMessageError("");
    const response = await fetch(`/api/sessions/${sessionId}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (response.ok) {
      setRunningMessage("");
      return;
    }
    if (response.status === 409) {
      setRunningMessageError(
        "The task stopped accepting in-flight messages before this could be sent.",
      );
      await fetchSession();
    } else if (response.status === 429) {
      setRunningMessageError("Too many messages are waiting to be picked up.");
    } else {
      setRunningMessageError("Message could not be sent.");
    }
  };

  const mutateGoal = async (
    action: string,
    body: Record<string, unknown> = {},
  ) => {
    const goal = session?.active_goal ? session.goal : undefined;
    if (!goal) return false;
    setGoalBusy(true);
    try {
      const response = await fetch(`/api/goals/${goal.run.id}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal_sha256: goal.sha256, ...body }),
      });
      await fetchSession();
      return response.ok;
    } finally {
      setGoalBusy(false);
    }
  };

  const pauseGoal = () => mutateGoal("pause");
  const resumeGoal = () => mutateGoal("resume");
  const acceptGoal = () => mutateGoal("accept");
  const stopGoal = async () => {
    if (
      !window.confirm(
        "Stop this goal? Completed commits and current workspace changes will be preserved.",
      )
    ) return;
    await mutateGoal("cancel");
  };
  const editGoal = async () => {
    const goal = session?.active_goal ? session.goal : undefined;
    if (!goal) return;
    if (
      goal.run.stage === "paused" ||
      goal.run.stage === "awaiting_user_review" ||
      goal.run.stage === "awaiting_plan_approval"
    ) {
      setGoalEditOpen(true);
      return;
    }
    setEditAfterPause(true);
    await pauseGoal();
  };
  const approveGoalPlan = async (planSha256: string, amendmentId?: string) => {
    const goal = session?.active_goal ? session.goal : undefined;
    if (!goal) return;
    const action = amendmentId
      ? `amendments/${amendmentId}/approve`
      : "approve-plan";
    await mutateGoal(action, { plan_sha256: planSha256 });
  };
  const discardGoalAmendment = async (amendmentId: string) => {
    await mutateGoal(`amendments/${amendmentId}/discard`);
  };

  const resumeSession = async () => {
    await fetch(`/api/sessions/${sessionId}/resume`, { method: "POST" });
    setSessionRunning(false);
    await fetchSession();
  };

  const cancelSession = async () => {
    await fetch(`/api/sessions/${sessionId}/cancel`, { method: "POST" });
    setSessionRunning(false);
    setIntent("discuss");
    await fetchSession();
  };

  const recoverTaskPlanning = async (
    action: "retry-task-planning" | "run-as-one-build",
  ) => {
    setTaskRecoveryBusy(true);
    try {
      await fetch(`/api/sessions/${sessionId}/${action}`, { method: "POST" });
      await fetchSession();
    } finally {
      setTaskRecoveryBusy(false);
    }
  };

  const shareSession = async () => {
    if (!session) return;
    const shareUrl = new URL(
      `/sessions/${session.session_id}`,
      window.location.origin,
    ).toString();
    const shareData: ShareData = {
      title: sessionPageDocumentTitle(session),
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
    if (!sessionId) return;
    atBottomRef.current = true;
    void fetchSession().then(() => openEvents(sessionId));
    return () => sourceRef.current?.close();
  }, [sessionId]);

  useEffect(() => {
    if (!session) return;
    document.title = session.active_goal && session.goal
      ? `${
        goalStageLabel(session.goal.run.stage, session.goal.run.pause_requested)
      } · ${sessionPageDocumentTitle(session)}`
      : sessionPageDocumentTitle(session);
  }, [session]);

  useEffect(() => {
    const goal = session?.active_goal ? session.goal : undefined;
    if (editAfterPause && goal?.run.stage === "paused") {
      setEditAfterPause(false);
      setGoalEditOpen(true);
    }
  }, [session, editAfterPause]);

  useEffect(() => {
    if (!shareMessage) return;
    const timer = window.setTimeout(() => setShareMessage(""), 2400);
    return () => window.clearTimeout(timer);
  }, [shareMessage]);

  const isRunning = sessionRunning;
  const activeGoal = session?.active_goal ? session.goal : undefined;
  const activeGoalBanner = activeGoal
    ? (
      <GoalModeBanner
        goal={activeGoal}
        busy={goalBusy}
        onDetails={() => setGoalDetailsOpen(true)}
        onPause={() => void pauseGoal()}
        onResume={() => void resumeGoal()}
        onAccept={() => void acceptGoal()}
        onEdit={() => void editGoal()}
        onStop={() => void stopGoal()}
      />
    )
    : null;
  const deliveryProposal = latestPendingDeliveryProposal(events);
  const goalProposal = latestPendingGoalProposal(events);
  const goalChangeRequest = latestGoalChangeRequest(events);

  if (!session) {
    return (
      <div className="app-shell session-shell">
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
          <div className="session-loading" role="status" aria-live="polite">
            <span
              className="spinner-border spinner-border-sm"
              aria-hidden="true"
            >
            </span>
            <span>Loading session…</span>
          </div>
        </section>
      </div>
    );
  }

  const actionTimeline = buildActionTimeline(events);
  const todoTasks = buildTodoTasks(events);
  const taskPlanningTranscript = session.task_planning_transcript ??
    session.multi_task?.run.planning_transcript;
  const showWorkDrawer = Boolean(
    session.goal || actionTimeline.length > 0 || todoTasks.length > 0,
  );
  const sessionStartMs =
    events.find((event) => event.event.type === "started")?.event
      .timestamp_ms ?? session.updated_at_ms;

  return (
    <div className="app-shell session-shell">
      <Aside />

      <section className="session-panel">
        <header className="session-header">
          <button
            type="button"
            className="btn btn-link d-xl-none p-0 text-body"
            onClick={() => navigate("/")}
          >
            <i className="bi bi-chevron-left fs-4"></i>
          </button>
          <div className="session-header-copy min-w-0 flex-grow-1">
            <h1>{sessionTitle(session)}</h1>
            <div className="status-line">
              {isRunning && <span className="live-dot" />}
              <span className="session-state-label">
                {activeGoal
                  ? goalStageLabel(
                    activeGoal.run.stage,
                    activeGoal.run.pause_requested,
                  )
                  : session.strict_workflow && session.workflow
                  ? workflowProgressLabel(
                    session.workflow.stage,
                    session.workflow.outcome,
                  )
                  : session.status === "running"
                  ? "Running"
                  : session.status === "queued"
                  ? "Queued"
                  : session.status === "paused"
                  ? session.pending_question ? "Waiting for answer" : "Paused"
                  : session.status === "failed"
                  ? "Failed"
                  : "Completed"}
              </span>
              {sessionStartMs && (
                <>
                  <span className="dot-sep"></span>
                  <span>{formatStartTime(sessionStartMs)}</span>
                </>
              )}
              {session.branch
                ? (
                  <>
                    <span className="dot-sep d-none d-sm-inline"></span>
                    <span className="d-none d-sm-inline">
                      Branch: {session.branch}
                    </span>
                  </>
                )
                : null}
            </div>
          </div>
          <div className="share-action">
            <button
              type="button"
              className="btn session-header-action"
              onClick={shareSession}
              aria-label="Share this session"
            >
              <i className="bi bi-box-arrow-up"></i>
            </button>
            {shareMessage && (
              <span
                className="share-status small text-body-secondary"
                role="status"
              >
                {shareMessage}
              </span>
            )}
          </div>
          <button
            className="btn session-header-action stop-action"
            onClick={() => activeGoal ? void stopGoal() : void cancelSession()}
            disabled={!isRunning && session.status !== "paused"}
            aria-label="Stop session"
          >
            <i className="bi bi-stop-fill"></i>
          </button>
        </header>

        {session.multi_task
          ? (
            <TaskProgress
              checkpoint={session.multi_task}
              activeTaskDetail={activeGoalBanner}
            />
          )
          : activeGoalBanner}

        {session.task_plan_rejected
          ? (
            <TaskPlanningRecovery
              rejection={session.task_plan_rejected}
              busy={taskRecoveryBusy}
              onRetry={() => void recoverTaskPlanning("retry-task-planning")}
              onRunAsBuild={() => void recoverTaskPlanning("run-as-one-build")}
              onEdit={() => {
                setIntent("deliver");
                setFollowUp(session.task);
              }}
            />
          )
          : null}

        {taskPlanningTranscript && taskPlanningTranscript.attempts.length > 0
          ? (
            <TaskPlanningDetails
              transcript={taskPlanningTranscript}
            />
          )
          : null}

        <div
          className={`session-layout${
            showWorkDrawer ? " has-work-drawer" : ""
          }`}
        >
          <main className="chat-stream" ref={chatRef} onScroll={onChatScroll}>
            {events.length === 0
              ? (
                <InitialUserMessage
                  task={session.task}
                  timestampMs={session.updated_at_ms}
                />
              )
              : (
                groupActionEvents(chatEventsWithOnlyLatestStep(events)).map(
                  (grouped, i) => {
                    if ("type" in grouped && grouped.type === "action_group") {
                      return (
                        <ActionGroupBubble
                          key={i}
                          actor={grouped.actor}
                          assistingProfile={grouped.assistingProfile}
                          inferenceEvents={grouped.inferenceEvents}
                          toolCalls={grouped.toolCalls}
                          toolResults={grouped.toolResults}
                          controllerActions={grouped.controllerActions}
                        />
                      );
                    }
                    return (
                      <MessageBubble
                        key={i}
                        envelope={grouped as EventEnvelope}
                        activityProfile={latestAssistantProfile(events)}
                        evidenceEvents={events}
                      />
                    );
                  },
                )
              )}
          </main>

          {showWorkDrawer
            ? (
              <aside
                className="tool-drawer d-none d-xl-block"
                aria-label="Work details"
              >
                {session.goal
                  ? (
                    <DrawerPanel
                      title="Goal"
                      icon="bi bi-bullseye"
                      count={session.goal.run.milestones.length}
                    >
                      <GoalDrawer goal={session.goal} />
                    </DrawerPanel>
                  )
                  : null}
                {actionTimeline.length > 0
                  ? (
                    <DrawerPanel
                      title="Actions"
                      icon="bi bi-lightning-charge"
                      count={events.filter((e) =>
                        e.event.type === "tool_call" ||
                        isControllerActionEvent(e)
                      ).length}
                      defaultOpen={false}
                    >
                      {actionTimeline.map((item, index) => (
                        <ActionDrawerItem
                          key={`action-${index}`}
                          item={item}
                        />
                      ))}
                    </DrawerPanel>
                  )
                  : null}

                {todoTasks.length > 0
                  ? (
                    <DrawerPanel
                      title="Plan"
                      icon="bi bi-check2-square"
                      count={todoTasks.length}
                    >
                      <TodoDrawer tasks={todoTasks} />
                    </DrawerPanel>
                  )
                  : null}
              </aside>
            )
            : null}
        </div>

        {activeGoal?.run.stage === "awaiting_plan_approval"
          ? (
            <GoalPlanReview
              goal={activeGoal}
              busy={goalBusy}
              onApprove={(plan, amendmentId) =>
                void approveGoalPlan(plan, amendmentId)}
              onDiscardAmendment={(amendmentId) =>
                void discardGoalAmendment(amendmentId)}
              onEdit={() => void editGoal()}
              onStop={() => void stopGoal()}
            />
          )
          : null}

        {activeGoal && goalChangeRequest &&
            (activeGoal.run.stage === "paused" ||
              activeGoal.run.pause_requested)
          ? (
            <div className="goal-decision-card" role="status">
              <div>
                <span className="goal-eyebrow">
                  Goal {goalChangeRequest.kind} request
                </span>
                <strong>Review a model-requested change</strong>
                <p>{goalChangeRequest.summary}</p>
              </div>
              <div>
                <button
                  className="btn btn-primary btn-sm"
                  type="button"
                  disabled={goalBusy || activeGoal.run.stage !== "paused"}
                  onClick={() => void editGoal()}
                >
                  Review change
                </button>
                <button
                  className="btn btn-light btn-sm"
                  type="button"
                  disabled={goalBusy || activeGoal.run.stage !== "paused"}
                  onClick={() => void resumeGoal()}
                >
                  Continue without change
                </button>
              </div>
            </div>
          )
          : null}

        {deliveryProposal && !isRunning && session.status === "completed" && (
          <div className="delivery-proposal-card" role="status">
            <div>
              <strong>Ready to turn this into a build?</strong>
              <span>{deliveryProposal.task_summary}</span>
            </div>
            <button
              className="btn btn-primary"
              type="button"
              onClick={() =>
                void continueSession(
                  deliveryProposal.task_summary,
                  "deliver",
                  deliveryProposal.proposal_id,
                )}
            >
              Build this
            </button>
          </div>
        )}

        {goalProposal && !isRunning && session.status === "completed" && (
          <div
            className="delivery-proposal-card goal-proposal-card"
            role="status"
          >
            <div>
              <strong>Make this a durable goal?</strong>
              <span>{goalProposal.objective}</span>
              <small>
                {goalProposal.criteria.length || 1} completion{" "}
                {goalProposal.criteria.length === 1 ? "criterion" : "criteria"}
                {" "}
                · plan approval required
              </small>
            </div>
            <button
              className="btn btn-primary"
              type="button"
              onClick={() => setGoalStartOpen(true)}
            >
              Review goal
            </button>
          </div>
        )}

        {isRunning
          ? (
            <form
              className="composer running-message-composer"
              onSubmit={(e) => {
                e.preventDefault();
                void sendRunningMessage();
              }}
            >
              <div className="running-message-field">
                <input
                  className="form-control"
                  value={runningMessage}
                  onChange={(e) => setRunningMessage(e.target.value)}
                  placeholder="Message the running agent…"
                  aria-label="Message the running agent"
                  maxLength={8000}
                />
                {runningMessageError
                  ? (
                    <small className="text-danger" role="status">
                      {runningMessageError}
                    </small>
                  )
                  : null}
              </div>
              <button
                className="btn btn-primary rounded-circle"
                type="submit"
                disabled={!runningMessage.trim()}
                aria-label="Send message to running agent"
                title="Picked up at the next agent loop"
              >
                <i className="bi bi-arrow-up"></i>
              </button>
            </form>
          )
          : activeGoal
          ? null
          : session.status === "paused" && session.pending_question
          ? (
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
              {session.pending_question.choices?.length
                ? (
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
                )
                : (
                  <>
                    <input
                      className="form-control"
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
          )
          : session.status === "paused" ||
              (session.status === "failed" &&
                session.workflow?.stage === "blocked")
          ? (
            <footer className="composer paused-composer">
              <div className="paused-composer-copy small text-body-secondary">
                {session.workflow?.stage === "blocked"
                  ? "Delivery needs help. Resolve the reported prerequisite, then resume from the preserved stage."
                  : "This session was restored after a daemon restart and is paused until you resume it."}
              </div>
              <button
                className="btn btn-warning composer-action"
                onClick={() => void resumeSession()}
              >
                Resume
              </button>
            </footer>
          )
          : !isRunning &&
              (session.status === "completed" || session.status === "failed")
          ? (
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                void continueSession();
              }}
            >
              <IntentControl intent={intent} onChange={setIntent} />
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
          )
          : null}
        {goalDetailsOpen && session.goal
          ? (
            <div
              className="goal-details-overlay"
              onMouseDown={() => setGoalDetailsOpen(false)}
            >
              <section
                className="goal-details-sheet"
                role="dialog"
                aria-modal="true"
                aria-label="Goal details"
                onMouseDown={(event) => event.stopPropagation()}
              >
                <header>
                  <h2>Goal details</h2>
                  <button
                    className="btn btn-light btn-icon"
                    type="button"
                    onClick={() => setGoalDetailsOpen(false)}
                    aria-label="Close goal details"
                  >
                    <i className="bi bi-x-lg"></i>
                  </button>
                </header>
                <GoalDrawer goal={session.goal} />
                {activeGoal
                  ? (
                    <div className="goal-mobile-actions">
                      {activeGoal.run.stage === "awaiting_plan_approval"
                        ? null
                        : activeGoal.run.stage === "awaiting_user_review"
                        ? (
                          <button
                            className="btn btn-primary"
                            type="button"
                            disabled={goalBusy}
                            onClick={() => {
                              setGoalDetailsOpen(false);
                              void acceptGoal();
                            }}
                          >
                            Accept goal
                          </button>
                        )
                        : activeGoal.run.stage === "paused" ||
                            activeGoal.run.stage === "blocked"
                        ? (
                          <button
                            className="btn btn-primary"
                            type="button"
                            disabled={goalBusy}
                            onClick={() => {
                              setGoalDetailsOpen(false);
                              void resumeGoal();
                            }}
                          >
                            Resume
                          </button>
                        )
                        : (
                          <button
                            className="btn btn-light"
                            type="button"
                            disabled={goalBusy ||
                              activeGoal.run.pause_requested}
                            onClick={() => {
                              setGoalDetailsOpen(false);
                              void pauseGoal();
                            }}
                          >
                            {activeGoal.run.pause_requested
                              ? "Pause requested"
                              : "Pause"}
                          </button>
                        )}
                      <button
                        className="btn btn-light"
                        type="button"
                        onClick={() => {
                          setGoalDetailsOpen(false);
                          void editGoal();
                        }}
                      >
                        {activeGoal.run.stage === "paused" ||
                            activeGoal.run.stage === "awaiting_plan_approval" ||
                            activeGoal.run.stage === "awaiting_user_review"
                          ? "Edit goal"
                          : "Pause and edit"}
                      </button>
                      <button
                        className="btn btn-outline-danger"
                        type="button"
                        onClick={() => {
                          setGoalDetailsOpen(false);
                          void stopGoal();
                        }}
                      >
                        Stop goal
                      </button>
                    </div>
                  )
                  : null}
              </section>
            </div>
          )
          : null}
        {goalEditOpen && activeGoal
          ? (
            <GoalAmendmentSheet
              goal={activeGoal}
              onClose={() => setGoalEditOpen(false)}
              onSubmitted={() => {
                setGoalEditOpen(false);
                void fetchSession();
              }}
            />
          )
          : null}
        <GoalStartSheet
          open={goalStartOpen}
          initialObjective={goalProposal?.objective ?? followUp}
          initialCriteria={goalProposal?.criteria.map((criterion) =>
            criterion.text
          )}
          sessionId={session.session_id}
          onClose={() => setGoalStartOpen(false)}
          onStarted={() => {
            setGoalStartOpen(false);
            setFollowUp("");
            void fetchSession();
          }}
        />
      </section>
    </div>
  );
}

export interface PendingDeliveryProposal {
  proposal_id: string;
  source_turn_id: string;
  task_summary: string;
}

export function workflowStageLabel(stage: string): string {
  switch (stage) {
    case "planning":
      return "Planning";
    case "plan_review":
      return "Challenging the plan";
    case "plan_revision":
      return "Revising the plan";
    case "implementing":
      return "Implementing";
    case "checking":
      return "Running checks";
    case "code_review":
      return "Challenging the code";
    case "repairing":
      return "Repairing";
    case "committing":
      return "Creating reviewed commit";
    case "ready":
      return "Ready";
    case "blocked":
      return "Needs help";
    case "cancelled":
      return "Cancelled";
    default:
      return "Stopped";
  }
}

export function workflowOutcomeLabel(outcome?: string): string {
  switch (outcome) {
    case "ready":
      return "Ready";
    case "no_change":
      return "No code changes";
    case "checks_failed":
    case "review_failed":
    case "repair_cycles_exhausted":
    case "contract_unsatisfied":
      return "Needs another pass";
    case "executor_unavailable":
    case "commit_blocked":
      return "Needs help";
    case "cancelled":
      return "Cancelled — work preserved";
    case undefined:
      return "Strict delivery in progress";
    default:
      return "Stopped safely";
  }
}

export function workflowProgressLabel(stage: string, outcome?: string): string {
  return outcome ? workflowOutcomeLabel(outcome) : workflowStageLabel(stage);
}

export function readyEvidenceLabel(commitOid: string): string {
  return `Reviewed commit ${commitOid.slice(0, 12)} is ready to publish`;
}

export function latestPendingDeliveryProposal(
  events: EventEnvelope[],
): PendingDeliveryProposal | undefined {
  let pending: PendingDeliveryProposal | undefined;
  for (const { event } of events) {
    if (event.type === "delivery_proposed") {
      pending = {
        proposal_id: event.proposal_id,
        source_turn_id: event.source_turn_id,
        task_summary: event.task_summary,
      };
    } else if (
      event.type === "conversation_turn_started" && event.intent === "deliver"
    ) {
      pending = undefined;
    }
  }
  return pending;
}

export interface PendingGoalProposal {
  proposal_id: string;
  source_turn_id: string;
  objective: string;
  criteria: { text: string; verifier?: string }[];
}

export function latestPendingGoalProposal(
  events: EventEnvelope[],
): PendingGoalProposal | undefined {
  let pending: PendingGoalProposal | undefined;
  for (const { event } of events) {
    if (event.type === "goal_proposed") {
      pending = {
        proposal_id: event.proposal_id,
        source_turn_id: event.source_turn_id,
        objective: event.objective,
        criteria: event.criteria,
      };
    } else if (event.type === "goal_started") {
      pending = undefined;
    }
  }
  return pending;
}

export interface PendingGoalChangeRequest {
  goal_id: string;
  kind: string;
  summary: string;
}

export function latestGoalChangeRequest(
  events: EventEnvelope[],
): PendingGoalChangeRequest | undefined {
  let pending: PendingGoalChangeRequest | undefined;
  for (const { event } of events) {
    if (event.type === "goal_change_requested") {
      pending = {
        goal_id: event.goal_id,
        kind: event.kind,
        summary: event.summary,
      };
    } else if (
      event.type === "goal_amendment_requested" ||
      event.type === "goal_resumed" ||
      event.type === "goal_completed" ||
      event.type === "goal_failed" ||
      event.type === "goal_cancelled"
    ) {
      pending = undefined;
    }
  }
  return pending;
}
