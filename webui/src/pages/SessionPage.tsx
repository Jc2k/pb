import { Fragment, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Aside } from "../Aside";
import type {
  ComposerMode,
  EventEnvelope,
  SessionDetails,
  WorkflowSummary,
} from "../types";
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
import { VoiceInputButton } from "../components/VoiceInputButton";
import { goalStageLabel } from "../lib/goalUtils";
import { SCROLL_THRESHOLD } from "../lib/constants";
import {
  buildChatPresentation,
  formatStartTime,
  formatTranscriptTime,
  groupActionEvents,
  isActionGroup,
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
} from "../components/Session";
import {
  buildActionTimeline,
  chatEventsWithOnlyLatestStep,
} from "../lib/sessionUtils";
import { isAbortError, LatestRequest } from "../lib/hooks";
import { apiErrorMessage } from "../lib/integrationConfig";

export function isNewerThanSnapshot(
  sequence: number,
  revision: number | null,
): boolean {
  return revision !== null && sequence > revision;
}

export function workflowRecoveryPresentation(
  workflow?: WorkflowSummary | null,
): {
  title: string;
  description: string;
  label: string;
  action: "resume" | "restart-delivery";
} {
  if (workflow?.recovery === "restart_from_current_files") {
    const contentChanged = workflow.blocked_cause ===
      "repository_content_changed";
    return {
      title: contentChanged
        ? "The project changed during review"
        : "This delivery needs a fresh plan",
      description: contentChanged
        ? "The review stays tied to its earlier snapshot. Restart planning against the current files; the previous plan and review remain in this history."
        : "The old checkpoint cannot continue safely. Restart planning against the current files; the previous attempt remains in this history.",
      label: "Restart with current files",
      action: "restart-delivery",
    };
  }
  if (workflow?.recovery === "resume") {
    return {
      title: "Delivery is waiting on a prerequisite",
      description:
        "Resolve the reported executor problem, then continue from the preserved stage.",
      label: "Resume after fixing",
      action: "resume",
    };
  }
  return {
    title: "Session paused safely",
    description:
      "This session was restored after a service restart and is waiting for you to continue it.",
    label: "Resume",
    action: "resume",
  };
}

export function mergeEventHistory(
  earlier: EventEnvelope[],
  later: EventEnvelope[],
): EventEnvelope[] {
  const merged = [...earlier];
  const positions = new Map(
    merged.map((envelope, index) => [envelope.transcript.entry_key, index]),
  );
  for (const envelope of later) {
    const existing = positions.get(envelope.transcript.entry_key);
    if (existing === undefined) {
      positions.set(envelope.transcript.entry_key, merged.length);
      merged.push(envelope);
    } else {
      merged[existing] = envelope;
    }
  }
  return merged.sort((left, right) =>
    left.transcript.sequence - right.transcript.sequence
  );
}

export function SessionPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [sessionError, setSessionError] = useState("");
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
  const [voiceInputActive, setVoiceInputActive] = useState(false);
  const [shareMessage, setShareMessage] = useState("");
  const [taskRecoveryBusy, setTaskRecoveryBusy] = useState(false);
  const [workflowRecoveryBusy, setWorkflowRecoveryBusy] = useState(false);
  const [workflowRecoveryError, setWorkflowRecoveryError] = useState("");
  const [actionError, setActionError] = useState("");
  const [showMessageTimes, setShowMessageTimes] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const sessionRequestRef = useRef(new LatestRequest());
  const sessionFetchControllerRef = useRef<AbortController | null>(null);
  const sessionRefreshRequestedRef = useRef(false);
  const actionRequestRef = useRef(new LatestRequest());
  const messageRequestRef = useRef(new LatestRequest());
  const snapshotRevisionRef = useRef<number | null>(null);
  const latestRefreshEffectRef = useRef(0);
  const refreshTimerRef = useRef<number | null>(null);
  const latestTitleEffectRef = useRef<
    { sequence: number; title: string } | null
  >(null);
  const latestRunningEffectRef = useRef<
    { sequence: number; running: boolean } | null
  >(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);
  const messageTimePullStartRef = useRef<{ x: number; y: number } | null>(null);

  const scheduleSessionRefresh = () => {
    if (
      refreshTimerRef.current !== null ||
      sessionFetchControllerRef.current !== null
    ) return;
    refreshTimerRef.current = window.setTimeout(() => {
      refreshTimerRef.current = null;
      void fetchSession();
    }, 0);
  };

  const openEvents = (id: string) => {
    if (sourceRef.current) sourceRef.current.close();
    const src = new EventSource(`/api/sessions/${id}/events`);
    sourceRef.current = src;
    src.onmessage = (msg) => {
      if (sourceRef.current !== src) return;
      try {
        const parsed = JSON.parse(msg.data) as EventEnvelope;
        setEvents((previous) => mergeEventHistory(previous, [parsed]));
        const effect = parsed.transcript.session_effect;
        const sequence = parsed.transcript.sequence;
        const newerThanSnapshot = isNewerThanSnapshot(
          sequence,
          snapshotRevisionRef.current,
        );
        if (effect.title) {
          const title = effect.title;
          const currentEffect = latestTitleEffectRef.current;
          if (
            !currentEffect ||
            sequence > currentEffect.sequence
          ) {
            latestTitleEffectRef.current = {
              sequence,
              title,
            };
            if (newerThanSnapshot) {
              setSession((current) =>
                current ? { ...current, title } : current
              );
            }
          }
        }
        if (effect.refresh) {
          latestRefreshEffectRef.current = Math.max(
            latestRefreshEffectRef.current,
            sequence,
          );
          if (newerThanSnapshot) scheduleSessionRefresh();
        }
        if (effect.running === "running") {
          const currentEffect = latestRunningEffectRef.current;
          if (
            !currentEffect ||
            sequence > currentEffect.sequence
          ) {
            latestRunningEffectRef.current = {
              sequence,
              running: true,
            };
            if (newerThanSnapshot) setSessionRunning(true);
          }
        } else if (effect.running === "stopped") {
          const currentEffect = latestRunningEffectRef.current;
          if (
            !currentEffect ||
            sequence > currentEffect.sequence
          ) {
            latestRunningEffectRef.current = {
              sequence,
              running: false,
            };
            if (newerThanSnapshot) setSessionRunning(false);
          }
        }
        if (effect.reset_intent && newerThanSnapshot) {
          setIntent("discuss");
        }
      } catch (err) {
        console.error(err);
      }
    };
  };

  const fetchSession = async () => {
    if (sessionFetchControllerRef.current !== null) {
      sessionRefreshRequestedRef.current = true;
      return;
    }
    sessionRefreshRequestedRef.current = false;
    const controller = sessionRequestRef.current.start();
    sessionFetchControllerRef.current = controller;
    let accepted = false;
    try {
      const res = await fetch(`/api/sessions/${sessionId}`, {
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new Error(`Session request failed (${res.status})`);
      }
      const details = (await res.json()) as SessionDetails;
      if (!sessionRequestRef.current.owns(controller)) return;
      if (
        snapshotRevisionRef.current !== null &&
        details.revision < snapshotRevisionRef.current
      ) {
        accepted = true;
        return;
      }
      snapshotRevisionRef.current = details.revision;
      const titleEffect = latestTitleEffectRef.current;
      const runningEffect = latestRunningEffectRef.current;
      setSession(
        titleEffect && titleEffect.sequence > details.revision
          ? { ...details, title: titleEffect.title }
          : details,
      );
      setEvents((previous) => mergeEventHistory(details.events, previous));
      setSessionRunning(
        runningEffect && runningEffect.sequence > details.revision
          ? runningEffect.running
          : details.running,
      );
      setSessionError("");
      accepted = true;
    } catch (error) {
      if (
        isAbortError(error) || !sessionRequestRef.current.owns(controller)
      ) return;
      setSessionError(
        error instanceof Error ? error.message : "Session request failed",
      );
    } finally {
      if (sessionFetchControllerRef.current === controller) {
        sessionFetchControllerRef.current = null;
        if (
          sessionRefreshRequestedRef.current ||
          (accepted && snapshotRevisionRef.current !== null &&
            latestRefreshEffectRef.current > snapshotRevisionRef.current)
        ) {
          scheduleSessionRefresh();
        }
      }
    }
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
    setActionError("");
    const controller = actionRequestRef.current.start();
    try {
      const response = await fetch(`/api/sessions/${sessionId}/continue`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task,
          intent: requestedIntent,
          proposal_id: proposalId,
        }),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Could not continue the session"),
        );
      }
      if (!actionRequestRef.current.owns(controller)) return;
      setFollowUp("");
      setSessionRunning(false);
    } catch (error) {
      if (isAbortError(error) || !actionRequestRef.current.owns(controller)) {
        return;
      }
      setActionError(
        error instanceof Error
          ? error.message
          : "Could not continue the session",
      );
    }
  };

  const sendRunningMessage = async () => {
    const message = runningMessage.trim();
    if (!message) return;
    setRunningMessageError("");
    const controller = messageRequestRef.current.start();
    try {
      const response = await fetch(`/api/sessions/${sessionId}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Message could not be sent"),
        );
      }
      if (messageRequestRef.current.owns(controller)) setRunningMessage("");
    } catch (error) {
      if (isAbortError(error) || !messageRequestRef.current.owns(controller)) {
        return;
      }
      setRunningMessageError(
        error instanceof Error ? error.message : "Message could not be sent",
      );
      await fetchSession();
    }
  };

  const mutateGoal = async (
    action: string,
    body: Record<string, unknown> = {},
  ) => {
    const goal = session?.active_goal ? session.goal : undefined;
    if (!goal) return false;
    setGoalBusy(true);
    setActionError("");
    const controller = actionRequestRef.current.start();
    try {
      const response = await fetch(`/api/goals/${goal.run.id}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal_sha256: goal.sha256, ...body }),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Could not update the goal"),
        );
      }
      if (!actionRequestRef.current.owns(controller)) return false;
      await fetchSession();
      return true;
    } catch (error) {
      if (!isAbortError(error) && actionRequestRef.current.owns(controller)) {
        setActionError(
          error instanceof Error ? error.message : "Could not update the goal",
        );
        await fetchSession();
      }
      return false;
    } finally {
      if (actionRequestRef.current.owns(controller)) setGoalBusy(false);
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

  const recoverWorkflow = async (action: "resume" | "restart-delivery") => {
    setWorkflowRecoveryBusy(true);
    setWorkflowRecoveryError("");
    const controller = actionRequestRef.current.start();
    try {
      const response = await fetch(`/api/sessions/${sessionId}/${action}`, {
        method: "POST",
        signal: controller.signal,
      });
      if (!actionRequestRef.current.owns(controller)) return;
      if (!response.ok) {
        setWorkflowRecoveryError(
          await apiErrorMessage(
            response,
            "The session is not ready to resume.",
          ),
        );
        return;
      }
      setSessionRunning(false);
      await fetchSession();
    } catch (error) {
      if (!isAbortError(error) && actionRequestRef.current.owns(controller)) {
        setWorkflowRecoveryError(
          error instanceof Error
            ? error.message
            : "Could not resume the session",
        );
      }
    } finally {
      if (actionRequestRef.current.owns(controller)) {
        setWorkflowRecoveryBusy(false);
      }
    }
  };

  const cancelSession = async () => {
    setActionError("");
    const controller = actionRequestRef.current.start();
    try {
      const response = await fetch(`/api/sessions/${sessionId}/cancel`, {
        method: "POST",
        signal: controller.signal,
      });
      if (!actionRequestRef.current.owns(controller)) return;
      if (!response.ok) {
        setActionError(
          await apiErrorMessage(response, "Could not stop the session"),
        );
        return;
      }
      setSessionRunning(false);
      setIntent("discuss");
      await fetchSession();
    } catch (error) {
      if (!isAbortError(error) && actionRequestRef.current.owns(controller)) {
        setActionError(
          error instanceof Error ? error.message : "Could not stop the session",
        );
      }
    }
  };

  const recoverTaskPlanning = async (
    action: "retry-task-planning" | "run-as-one-build",
  ) => {
    setTaskRecoveryBusy(true);
    setActionError("");
    const controller = actionRequestRef.current.start();
    try {
      const response = await fetch(`/api/sessions/${sessionId}/${action}`, {
        method: "POST",
        signal: controller.signal,
      });
      if (!actionRequestRef.current.owns(controller)) return;
      if (!response.ok) {
        setActionError(
          await apiErrorMessage(response, "Could not recover task planning"),
        );
        return;
      }
      await fetchSession();
    } catch (error) {
      if (!isAbortError(error) && actionRequestRef.current.owns(controller)) {
        setActionError(
          error instanceof Error
            ? error.message
            : "Could not recover task planning",
        );
      }
    } finally {
      if (actionRequestRef.current.owns(controller)) setTaskRecoveryBusy(false);
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
    setActionError("");
    const controller = actionRequestRef.current.start();
    try {
      const response = await fetch(`/api/sessions/${sessionId}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_id: session.pending_question.question_id,
          answer: selectedAnswer,
        }),
        signal: controller.signal,
      });
      if (!actionRequestRef.current.owns(controller)) return;
      if (!response.ok) {
        setActionError(
          await apiErrorMessage(response, "Could not submit the answer"),
        );
        await fetchSession();
        return;
      }
      setAnswer("");
      setSessionRunning(true);
    } catch (error) {
      if (!isAbortError(error) && actionRequestRef.current.owns(controller)) {
        setActionError(
          error instanceof Error
            ? error.message
            : "Could not submit the answer",
        );
      }
    }
  };

  const onChatScroll = () => {
    const el = chatRef.current;
    if (!el) return;
    atBottomRef.current =
      el.scrollTop + el.clientHeight >= el.scrollHeight - SCROLL_THRESHOLD;
  };

  const beginMessageTimePull = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.pointerType !== "touch") return;
    messageTimePullStartRef.current = { x: event.clientX, y: event.clientY };
  };

  const updateMessageTimePull = (event: ReactPointerEvent<HTMLElement>) => {
    const start = messageTimePullStartRef.current;
    if (!start || event.pointerType !== "touch") return;
    const horizontalDistance = event.clientX - start.x;
    const verticalDistance = event.clientY - start.y;
    if (
      horizontalDistance < -28 &&
      Math.abs(horizontalDistance) > Math.abs(verticalDistance)
    ) {
      setShowMessageTimes(true);
    }
  };

  const endMessageTimePull = () => {
    messageTimePullStartRef.current = null;
    setShowMessageTimes(false);
  };

  useEffect(() => {
    if (atBottomRef.current && chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [events]);

  useEffect(() => {
    if (!sessionId) return;
    atBottomRef.current = true;
    setSession(null);
    setSessionError("");
    setActionError("");
    setEvents([]);
    setFollowUp("");
    setRunningMessage("");
    setRunningMessageError("");
    setAnswer("");
    setIntent("discuss");
    latestTitleEffectRef.current = null;
    latestRunningEffectRef.current = null;
    snapshotRevisionRef.current = null;
    latestRefreshEffectRef.current = 0;
    sessionRefreshRequestedRef.current = false;
    if (refreshTimerRef.current !== null) {
      window.clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }
    openEvents(sessionId);
    void fetchSession();
    return () => {
      sessionRequestRef.current.abort();
      sessionFetchControllerRef.current = null;
      actionRequestRef.current.abort();
      messageRequestRef.current.abort();
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      sourceRef.current?.close();
    };
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
  const deliveryProposal = session?.pending_delivery_proposal ?? undefined;
  const goalProposal = session?.pending_goal_proposal ?? undefined;
  const goalChangeRequest = session?.pending_goal_change ?? undefined;

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
          {sessionError
            ? (
              <div className="session-loading" role="alert">
                <span>{sessionError}</span>
                <button
                  className="btn btn-sm btn-outline-secondary"
                  type="button"
                  onClick={() => void fetchSession()}
                >
                  Try again
                </button>
              </div>
            )
            : (
              <div
                className="session-loading"
                role="status"
                aria-live="polite"
              >
                <span
                  className="spinner-border spinner-border-sm"
                  aria-hidden="true"
                >
                </span>
                <span>Loading session…</span>
              </div>
            )}
        </section>
      </div>
    );
  }

  const actionTimeline = buildActionTimeline(events);
  const taskPlanningTranscript = session.task_planning_transcript ??
    session.multi_task?.run.planning_transcript;
  const showWorkDrawer = Boolean(
    session.goal || actionTimeline.length > 0,
  );
  const sessionStartMs = session.started_at_ms;
  const workflowRecovery = workflowRecoveryPresentation(session.workflow);

  return (
    <div className="app-shell session-shell">
      <Aside />

      <section className="session-panel">
        <header className="session-header">
          <button
            type="button"
            className="btn btn-link d-xl-none p-0 text-body session-back-button"
            onClick={() => navigate("/")}
            aria-label="Back to sessions"
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

        {actionError
          ? (
            <div className="alert alert-danger m-3 mb-0 py-2" role="alert">
              {actionError}
            </div>
          )
          : null}

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
          <main
            className={`chat-stream${
              showMessageTimes ? " show-message-times" : ""
            }`}
            ref={chatRef}
            onScroll={onChatScroll}
            onPointerDown={beginMessageTimePull}
            onPointerMove={updateMessageTimePull}
            onPointerUp={endMessageTimePull}
            onPointerCancel={endMessageTimePull}
            onPointerLeave={endMessageTimePull}
          >
            {events.length === 0
              ? (
                <InitialUserMessage
                  task={session.task}
                  timestampMs={session.updated_at_ms}
                />
              )
              : (
                buildChatPresentation(
                  groupActionEvents(chatEventsWithOnlyLatestStep(events)),
                ).map(
                  ({ item: grouped, showIdentity, timeDividerMs }, i) => {
                    const row = isActionGroup(grouped)
                      ? (
                        <ActionGroupBubble
                          actor={grouped.actor}
                          assistingProfile={grouped.assistingProfile}
                          inferenceEvents={grouped.inferenceEvents}
                          toolCalls={grouped.toolCalls}
                          toolResults={grouped.toolResults}
                          controllerActions={grouped.controllerActions}
                          showIdentity={showIdentity}
                        />
                      )
                      : (
                        <MessageBubble
                          envelope={grouped as EventEnvelope}
                          evidenceEvents={events}
                          focusRoot={session.workdir}
                          workflow={session.workflow ?? undefined}
                          showIdentity={showIdentity}
                        />
                      );
                    return (
                      <Fragment key={i}>
                        {timeDividerMs
                          ? (
                            <div className="chat-time-divider" role="separator">
                              <time
                                dateTime={new Date(timeDividerMs).toISOString()}
                              >
                                {formatTranscriptTime(timeDividerMs)}
                              </time>
                            </div>
                          )
                          : null}
                        {row}
                      </Fragment>
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
                  deliveryProposal.id,
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
                  readOnly={voiceInputActive}
                />
                {runningMessageError
                  ? (
                    <small className="text-danger" role="status">
                      {runningMessageError}
                    </small>
                  )
                  : null}
              </div>
              <VoiceInputButton
                value={runningMessage}
                onValueChange={setRunningMessage}
                onActiveChange={setVoiceInputActive}
              />
              <button
                className="btn btn-primary rounded-circle"
                type="submit"
                disabled={!runningMessage.trim() || voiceInputActive}
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
                      aria-label="Answer the planning question"
                      readOnly={voiceInputActive}
                    />
                    <VoiceInputButton
                      value={answer}
                      onValueChange={setAnswer}
                      onActiveChange={setVoiceInputActive}
                    />
                    <button
                      className="btn btn-warning rounded-circle"
                      type="submit"
                      disabled={!answer.trim() || voiceInputActive}
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
                <strong>{workflowRecovery.title}</strong>
                <span>{workflowRecovery.description}</span>
                {workflowRecoveryError
                  ? (
                    <span className="text-danger" role="alert">
                      {workflowRecoveryError}
                    </span>
                  )
                  : null}
              </div>
              <button
                className={`btn composer-action ${
                  workflowRecovery.action === "restart-delivery"
                    ? "btn-primary"
                    : "btn-warning"
                }`}
                disabled={workflowRecoveryBusy}
                onClick={() => void recoverWorkflow(workflowRecovery.action)}
              >
                {workflowRecoveryBusy ? "Starting…" : workflowRecovery.label}
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
                aria-label="Follow-up task"
                readOnly={voiceInputActive}
              />
              <VoiceInputButton
                value={followUp}
                onValueChange={setFollowUp}
                onActiveChange={setVoiceInputActive}
              />
              <button
                className="btn btn-primary rounded-circle"
                type="submit"
                disabled={!followUp.trim() || voiceInputActive}
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
    case "plan_rejected":
    case "plan_cycles_exhausted":
    case "step_limit":
    case "invocation_limit":
    case "token_limit":
    case "context_limit":
    case "engine_error":
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
