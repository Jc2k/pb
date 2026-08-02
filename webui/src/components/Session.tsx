import type React from "react";
import { Fragment, useEffect, useId, useRef, useState } from "react";
import type {
  AgentEvent,
  EventEnvelope,
  SessionItem,
  WorkflowSummary,
} from "../types";
import { TOOL_FRIENDLY_NAMES, TOOL_ICONS } from "../lib/constants";
import {
  formatEventTime,
  formatTranscriptTime,
  getAvatarForProfile,
  projectName,
  relativeTime,
  sessionTitle,
  toolResultForCall,
} from "../lib/helpers";
import {
  errorSummary,
  getToolDetail,
  harnessEfficiencyStats,
  profileJobTitle,
  profileName,
  TODO_STATUS_LABELS,
  trustedSessionSummaryCommitLines,
} from "../lib/sessionUtils";
import type {
  ActionTimelineItem,
  HarnessEfficiencyStats,
  TodoTask,
} from "../lib/sessionUtils";
import { parseInlineRichText, parseRichText } from "../lib/richText";
import {
  formatEnergy,
  formatPower,
  ledEquivalent,
  metricEnergyJoules,
  metricRuntimeMs,
} from "../lib/energy";
import {
  profileAccentClass,
  teamActorAccentClass,
  teamActorPresentation,
  workflowStewardActor,
} from "../lib/team";

function formatHumanDurationMs(ms?: number): string {
  if (ms === undefined) return "an unknown amount of time";

  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h${minutes.toString().padStart(2, "0")}`;
  }
  if (minutes > 0) {
    return `${minutes}m${seconds.toString().padStart(2, "0")}`;
  }
  return `${seconds}s`;
}

function formatNumber(value: number): string {
  return Math.trunc(value).toLocaleString("en-US");
}

function sentenceCaseIdentifier(value: string): string {
  const words = value.replaceAll("_", " ").trim();
  return words ? `${words[0].toUpperCase()}${words.slice(1)}` : "Unknown";
}

function prettyTechnicalDetail(value: string): string {
  const detail = value.trim();
  if (!detail) return "";
  try {
    return JSON.stringify(JSON.parse(detail), null, 2);
  } catch {
    return detail;
  }
}

function funEnergySummary(
  runtimeMs: number,
  tokens: number,
  energyJoules?: number,
): string {
  const prefix = `This session ran for ${
    formatHumanDurationMs(runtimeMs)
  }, used ${formatNumber(tokens)} tokens`;
  if (
    energyJoules === undefined || energyJoules < 0 ||
    !Number.isFinite(energyJoules)
  ) {
    return `${prefix}.`;
  }
  const equivalent = ledEquivalent(energyJoules);
  return `${prefix}, and used an estimated ${formatEnergy(energyJoules)}${
    equivalent ? `—the energy a 10 W LED bulb uses in ${equivalent}` : ""
  }.`;
}

function trinityAssistanceSummary(
  stats: HarnessEfficiencyStats,
): string | null {
  if (stats.proactiveActions === 0) return null;
  const actionLabel = stats.proactiveReads === stats.proactiveActions
    ? stats.proactiveActions === 1 ? "repository read" : "repository reads"
    : stats.proactiveInspections === stats.proactiveActions
    ? stats.proactiveActions === 1
      ? "changed-path inspection"
      : "changed-path inspections"
    : "repository checks";
  return `Trinity predicted and completed ${stats.proactiveActions} ${actionLabel} early, potentially avoiding up to ${stats.proactiveActions} extra model ${
    stats.proactiveActions === 1 ? "turn" : "turns"
  }.`;
}

function mutationCandidateCount(
  event: Extract<AgentEvent, { type: "llm_invocation" }>,
): number {
  return Object.values(event.native?.mutation_constraint_rejections || {})
    .reduce((total, count) => total + Math.max(0, count), 0);
}

function InlineRichText({ content }: { content: string }) {
  return (
    <>
      {parseInlineRichText(content).map((part, index) => {
        switch (part.type) {
          case "code":
            return (
              <code className="rich-text-inline-code" key={index}>
                {part.text}
              </code>
            );
          case "strong":
            return <strong key={index}>{part.text}</strong>;
          case "emphasis":
            return <em key={index}>{part.text}</em>;
          case "text":
            return <Fragment key={index}>{part.text}</Fragment>;
        }
      })}
    </>
  );
}

function MessageTime({ timestampMs }: { timestampMs?: number }) {
  if (!timestampMs) return null;
  return (
    <time
      className="message-time"
      dateTime={new Date(timestampMs).toISOString()}
    >
      {formatTranscriptTime(timestampMs)}
    </time>
  );
}

function RichText({ content }: { content: string }) {
  const blocks = parseRichText(content);

  return (
    <div className="rich-text">
      {blocks.map((block, index) => {
        switch (block.type) {
          case "heading": {
            const Heading = `h${
              block.level + 2
            }` as keyof React.JSX.IntrinsicElements;
            return (
              <Heading key={index}>
                <InlineRichText content={block.text} />
              </Heading>
            );
          }
          case "unordered_list":
            return (
              <ul key={index}>
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>
                    <InlineRichText content={item} />
                  </li>
                ))}
              </ul>
            );
          case "ordered_list":
            return (
              <ol key={index}>
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>
                    <InlineRichText content={item} />
                  </li>
                ))}
              </ol>
            );
          case "code":
            return (
              <pre
                key={index}
                className="rich-text-code"
              ><code>{block.code}</code></pre>
            );
          case "paragraph":
            return (
              <p key={index}>
                {block.lines.map((line, lineIndex) => (
                  <Fragment key={lineIndex}>
                    <InlineRichText content={line} />
                    {lineIndex < block.lines.length - 1 ? <br /> : null}
                  </Fragment>
                ))}
              </p>
            );
        }
      })}
    </div>
  );
}

export function DiffView({ diff }: { diff: string }) {
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

/* ─── message bubble component for grouping ────────────────── */

function controllerActionPresentation(event: AgentEvent): {
  label: string;
  detail: string;
  icon: string;
  anticipated?: boolean;
} | null {
  switch (event.type) {
    case "controller_observation":
      return {
        label: `${
          event.receipt.operation === "read_file" ? "Read" : "Inspected"
        } ${event.receipt.path}`,
        detail: `Done early · ${
          event.receipt.coverage === "full"
            ? `${event.receipt.observed_bytes.toLocaleString()} bytes · full coverage`
            : `${event.receipt.observed_bytes.toLocaleString()} bytes · bounded ranges`
        }`,
        icon: event.receipt.operation === "read_file"
          ? "bi bi-file-earmark-text"
          : "bi bi-search",
        anticipated: true,
      };
    case "controller_closure":
      return {
        label: "Closed no-change work",
        detail: event.reason,
        icon: "bi bi-check2-circle",
      };
    case "controller_mutation":
      return {
        label: `Deleted ${event.receipt.path}`,
        detail: "Tracked · Git-recoverable",
        icon: "bi bi-trash3",
      };
    default:
      return null;
  }
}

export function ActionGroupBubble({
  actor,
  assistingProfile,
  inferenceEvents,
  toolCalls,
  toolResults,
  controllerActions,
  showIdentity = true,
}: {
  actor?: import("../types").TeamActor;
  assistingProfile?: string;
  inferenceEvents: EventEnvelope[];
  toolCalls: EventEnvelope[];
  toolResults: EventEnvelope[];
  controllerActions: EventEnvelope[];
  showIdentity?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const collapseId = useId();

  if (toolCalls.length === 0 && controllerActions.length === 0) return null;

  const teammate = teamActorPresentation(actor);
  const isTrinity = actor?.kind === "automation";
  const accentClass = teamActorAccentClass(actor);

  const toolItems = toolCalls
    .map((e, i) => {
      if (e.event.type !== "tool_call") return null;

      const toolName = e.event.tool;
      const friendlyName = TOOL_FRIENDLY_NAMES[toolName] || toolName;
      const iconClass = TOOL_ICONS[toolName] || "bi bi-file-earmark-text";

      let statusClass = "pending";
      let detailText: string | null = null;

      const result = toolResultForCall(e, toolResults);
      if (result?.event.type === "tool_result") {
        statusClass = result.event.outcome || "unknown";
      }
      detailText = getToolDetail(e, result);

      return (
        <div key={`model-${i}`} className={`tool-item ${statusClass}`}>
          <i className={iconClass}></i>
          <span className="action-label">
            <span>{friendlyName}</span>
          </span>
          {detailText && <small>{detailText}</small>}
        </div>
      );
    })
    .filter(Boolean);

  const controllerItems = controllerActions.map((envelope, index) => {
    const presentation = controllerActionPresentation(envelope.event);
    if (!presentation) return null;
    return (
      <div
        key={`automatic-${index}`}
        className={`tool-item success automatic-action${
          presentation.anticipated ? " anticipated-action" : ""
        }`}
      >
        <i className={presentation.icon}></i>
        <span className="action-label">
          <span>{presentation.label}</span>
        </span>
        <small>{presentation.detail}</small>
      </div>
    );
  }).filter(Boolean);

  const actionNameList = toolCalls
    .map((e, i) => {
      if (e.event.type === "tool_call") {
        return TOOL_FRIENDLY_NAMES[e.event.tool] || e.event.tool;
      }
      return "";
    })
    .concat(
      controllerActions.map((envelope) =>
        controllerActionPresentation(envelope.event)?.label || ""
      ),
    )
    .filter(Boolean);
  const actionNameCounts = actionNameList.reduce((counts, name) => {
    counts.set(name, (counts.get(name) || 0) + 1);
    return counts;
  }, new Map<string, number>());
  const actionNames = [...actionNameCounts.entries()]
    .slice(0, 3)
    .map(([name, count]) => count > 1 ? `${name} ×${count}` : name)
    .concat(
      actionNameCounts.size > 3 ? [`+${actionNameCounts.size - 3} more`] : [],
    )
    .join(" · ");
  const actionCount = toolCalls.length + controllerActions.length;
  const assisting = assistingProfile
    ? profileName(assistingProfile)
    : undefined;
  const proactiveObservations = controllerActions.filter((envelope) =>
    envelope.event.type === "controller_observation"
  );
  const isProactiveRun = toolCalls.length === 0 &&
    proactiveObservations.length === controllerActions.length &&
    proactiveObservations.length > 0;
  const proactiveReads =
    proactiveObservations.filter((envelope) =>
      envelope.event.type === "controller_observation" &&
      envelope.event.receipt.operation === "read_file"
    ).length;
  const proactiveActionLabel = proactiveReads === proactiveObservations.length
    ? proactiveObservations.length === 1
      ? "repository read"
      : "repository reads"
    : proactiveReads === 0
    ? proactiveObservations.length === 1
      ? "change inspection"
      : "change inspections"
    : "repository checks";
  const actionSummary = isProactiveRun
    ? `Predicted ${
      proactiveObservations.length === 1 ? "a" : proactiveObservations.length
    } ${proactiveActionLabel}${
      assisting ? ` for ${assisting.split(/\s+/, 1)[0]}` : ""
    }`
    : actionCount === 1
    ? actionNames
    : actor?.kind === "automation"
    ? `${actionCount} harness actions${assisting ? ` for ${assisting}` : ""}`
    : `${actionCount} actions`;
  const firstEvent = toolCalls[0] || controllerActions[0];
  const timestampMs = firstEvent && "timestamp_ms" in firstEvent.event
    ? firstEvent.event.timestamp_ms
    : undefined;

  return (
    <article
      className={`bot message-row assistant-message compact tool-message${
        accentClass ? ` ${accentClass}` : ""
      }${showIdentity ? "" : " speaker-continuation"}`}
    >
      <div className="bot-avatar action-avatar">
        <img src={teammate.avatar} alt={teammate.name} />
      </div>
      <div className="message-container">
        <div className="author-line action-author-line">
          <strong>{teammate.name}</strong>
          <span>{teammate.role}</span>
          <span className="action-origin">{teammate.provenance}</span>
          {timestampMs ? <time>{formatEventTime(timestampMs)}</time> : null}
        </div>
        <div className="bubble thought-bubble action-bubble">
          <button
            className={`tool-strip${isOpen ? "" : " collapsed"}`}
            onClick={() => setIsOpen(!isOpen)}
            aria-expanded={isOpen}
            aria-controls={collapseId}
            type="button"
          >
            <span className="tool-strip-copy">
              <i
                className={isProactiveRun
                  ? "bi bi-stars trinity-prediction-glyph"
                  : "bi bi-lightning-charge"}
                aria-hidden="true"
              >
              </i>
              <span>
                <strong>{actionSummary}</strong>
                {actionCount > 1
                  ? <small className="tool-names">{actionNames}</small>
                  : null}
              </span>
            </span>
            <i
              className={`bi bi-chevron-down${isOpen ? "" : " collapsed"}`}
            >
            </i>
          </button>
          <div className={`collapse${isOpen ? " show" : ""}`} id={collapseId}>
            <div className="tool-list">
              {controllerItems}
              {toolItems}
            </div>
          </div>
        </div>
        <ActionInferenceDetails
          events={inferenceEvents}
          teammate={teammate.name}
        />
      </div>
      <MessageTime timestampMs={timestampMs} />
    </article>
  );
}

export function ActionDrawerItem({ item }: { item: ActionTimelineItem }) {
  const teammate = teamActorPresentation(item.actor);
  const controller = controllerActionPresentation(item.envelope.event);
  const tool = item.envelope.event.type === "tool_call"
    ? item.envelope.event
    : undefined;
  if (!controller && !tool) return null;
  const label = controller?.label ||
    (tool ? TOOL_FRIENDLY_NAMES[tool.tool] || tool.tool : "Action");
  const detail = controller?.detail ||
    (tool ? getToolDetail(item.envelope, item.result) : null);
  const icon = controller?.icon ||
    (tool
      ? TOOL_ICONS[tool.tool] || "bi bi-file-earmark-text"
      : "bi bi-lightning-charge");
  const timestampMs = "timestamp_ms" in item.envelope.event
    ? item.envelope.event.timestamp_ms
    : undefined;
  return (
    <div className="drawer-item action-drawer-item">
      <i className={icon} aria-hidden="true"></i>
      <span className="drawer-action-copy">
        <span className="drawer-action-title">
          <strong>{label}</strong>
          <small className="action-origin">{teammate.provenance}</small>
        </span>
        {detail
          ? <small className="drawer-action-detail">{detail}</small>
          : null}
        {timestampMs && <time>{formatEventTime(timestampMs)}</time>}
      </span>
    </div>
  );
}

export function DrawerPanel({
  title,
  icon,
  count,
  children,
  defaultOpen = true,
}: {
  title: string;
  icon: string;
  count: number;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <section className="drawer-panel">
      <button
        className="drawer-panel-header"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        type="button"
      >
        <span>
          <i className={icon}></i>
          <h2>{title}</h2>
        </span>
        <span className="drawer-count">
          <strong>{count}</strong>
          <i className={`bi bi-chevron-down${isOpen ? "" : " collapsed"}`}></i>
        </span>
      </button>
      {isOpen && <div className="drawer-panel-body">{children}</div>}
    </section>
  );
}

export function TodoDrawer({ tasks }: { tasks: TodoTask[] }) {
  if (tasks.length === 0) {
    return (
      <div className="empty-detail compact">
        <i className="bi bi-check2-square"></i>
        <h3>No managed tasks</h3>
        <p>
          Todo tool activity will appear here as the agent plans and updates
          work.
        </p>
      </div>
    );
  }

  return (
    <ol className="todo-list">
      {tasks.map((task) => (
        <li key={task.id} className={`todo-item ${task.status}`}>
          <div className="todo-title-row">
            <span className="todo-id">#{task.id}</span>
            <span className="todo-status">
              {TODO_STATUS_LABELS[task.status] || task.status}
            </span>
          </div>
          <strong>{task.title}</strong>
          {task.description && <p>{task.description}</p>}
          {task.parent_id ? <small>Parent #{task.parent_id}</small> : null}
          {task.notes?.length
            ? (
              <ul className="todo-notes">
                {task.notes.map((note, index) => <li key={index}>{note}</li>)}
              </ul>
            )
            : null}
          {task.timestampMs && <time>{formatEventTime(task.timestampMs)}</time>}
        </li>
      ))}
    </ol>
  );
}

function ErrorEventBubble({
  event,
}: {
  event: Extract<AgentEvent, { type: "error" }>;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const summary = errorSummary(event);
  const rawDetail = String(event.message || "").trim();
  const detail = rawDetail.startsWith(`${summary}:`)
    ? rawDetail.slice(summary.length + 1).trim()
    : rawDetail === summary
    ? ""
    : rawDetail;
  const hasDetail = detail.length > 0;

  return (
    <article className="message-row compact tool-message error-message">
      <div className="bubble thought-bubble error-tool-bubble">
        <button
          className={`tool-strip error-strip${isOpen ? "" : " collapsed"}`}
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          type="button"
        >
          <span>
            <i className="bi bi-exclamation-triangle-fill"></i> Error
          </span>
          <span className="tool-names">{summary}</span>
          <i
            className={`bi bi-chevron-down${isOpen ? "" : " collapsed"}`}
            aria-hidden="true"
          >
          </i>
        </button>
        <div className={`collapse${isOpen ? " show" : ""}`}>
          <div className="error-detail">
            {hasDetail
              ? <pre className="mb-0 small result-pre">{detail}</pre>
              : null}
            {!hasDetail
              ? (
                <p className="mb-0">
                  {summary || "No error details provided."}
                </p>
              )
              : null}
          </div>
        </div>
      </div>
    </article>
  );
}

function CorrectionNotice({
  envelope,
  showIdentity = true,
}: {
  envelope: EventEnvelope;
  showIdentity?: boolean;
}) {
  const event = envelope.event;
  if (event.type !== "correction") return null;
  const copy = envelope.chatter?.find(({ audience }) => audience === "team");
  const actor = copy?.actor || event.actor || workflowStewardActor();
  const teammate = teamActorPresentation(actor);
  const message = copy?.message || "Trinity has an update for this task.";
  const technicalDetail = prettyTechnicalDetail(copy?.detail || event.message);

  return (
    <article
      className={`bot message-row assistant-message compact correction-message chat-event-message teammate-message trinity-message${
        showIdentity ? "" : " speaker-continuation"
      }`}
      aria-label={`Correction from ${teammate.name}`}
    >
      <div className="bot-avatar team-avatar">
        <img src={teammate.avatar} alt={teammate.name} />
      </div>
      <div className="message-container">
        <div className="author-line">
          <strong>{teammate.name}</strong>
          <span>{teammate.role}</span>
          <span className="action-origin">{teammate.provenance}</span>
          {event.timestamp_ms
            ? <time>{formatEventTime(event.timestamp_ms)}</time>
            : null}
        </div>
        <TechnicalDetailsBubble
          className="correction-bubble"
          detail={technicalDetail}
          title={`${teammate.name} correction evidence`}
          teammateName={teammate.name}
        >
          {copy?.headline
            ? <strong className="feedback-heading">{copy.headline}</strong>
            : null}
          <RichText content={message} />
        </TechnicalDetailsBubble>
      </div>
      <MessageTime timestampMs={event.timestamp_ms} />
    </article>
  );
}

function WorkflowBlockedNotice({
  event,
  envelope,
  showIdentity = true,
}: {
  event: Extract<AgentEvent, { type: "workflow_blocked" }>;
  envelope: EventEnvelope;
  showIdentity?: boolean;
}) {
  const teammateFeedback = envelope.chatter?.find(
    ({ audience }) => audience === "team",
  );
  const userFeedback = envelope.chatter?.find(
    ({ audience }) => audience === "current_user",
  );
  const teammate = teamActorPresentation(
    teammateFeedback?.actor || workflowStewardActor(),
  );
  const [currentUsername, setCurrentUsername] = useState<string>();

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/current-user", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((currentUser: { username?: string } | null) => {
        const username = currentUser?.username?.trim();
        if (username) setCurrentUsername(username);
      })
      .catch((error) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          console.debug("Could not address the current user by name", error);
        }
      });
    return () => controller.abort();
  }, []);

  const teammateMessage = teammateFeedback?.message ||
    "Team, this delivery is on hold at a safe boundary because the reported problem needs a different approach.";
  const requestMessage = userFeedback?.message ||
    "Can you start a follow-up task here and add any context that could help the team find a different way forward?";
  const userMessage = currentUsername && userFeedback
    ? `@${currentUsername}, ${requestMessage.charAt(0).toLocaleLowerCase()}${
      requestMessage.slice(1)
    }`
    : requestMessage;

  return (
    <>
      <article
        className={`bot message-row assistant-message workflow-feedback chat-event-message teammate-message trinity-message${
          showIdentity ? "" : " speaker-continuation"
        }`}
        aria-label={`Task hold message from ${teammate.name}`}
      >
        <div className="bot-avatar team-avatar">
          <img src={teammate.avatar} alt={teammate.name} />
        </div>
        <div className="message-container">
          <div className="author-line">
            <strong>{teammate.name}</strong>
            <span>{teammate.role}</span>
            <span className="action-origin">{teammate.provenance}</span>
            {event.timestamp_ms
              ? <time>{formatEventTime(event.timestamp_ms)}</time>
              : null}
          </div>
          <TechnicalDetailsBubble
            className="correction-bubble"
            detail={prettyTechnicalDetail(
              teammateFeedback?.detail || event.reason,
            )}
            title="Why this task is on hold"
            teammateName={teammate.name}
          >
            <RichText content={teammateMessage} />
          </TechnicalDetailsBubble>
        </div>
        <MessageTime timestampMs={event.timestamp_ms} />
      </article>
      <article
        className="bot message-row assistant-message workflow-feedback chat-event-message teammate-message trinity-message speaker-continuation"
        aria-label={`Request from ${teammate.name} to the current user`}
      >
        <div className="bot-avatar team-avatar" aria-hidden="true">
          <img src={teammate.avatar} alt="" />
        </div>
        <div className="message-container">
          <div className="author-line">
            <strong>{teammate.name}</strong>
            <span>{teammate.role}</span>
            <span className="action-origin">{teammate.provenance}</span>
          </div>
          <div className="bubble thought-bubble correction-bubble">
            <RichText content={userMessage} />
          </div>
        </div>
        <MessageTime timestampMs={event.timestamp_ms} />
      </article>
    </>
  );
}

export function InitialUserMessage(
  { task, timestampMs }: { task: string; timestampMs?: number },
) {
  return (
    <article className="user message-row user-message">
      <div className="message-container">
        <div className="author-line">
          <strong>You</strong>
          {timestampMs ? <time>{formatEventTime(timestampMs)}</time> : null}
        </div>
        <div className="bubble user-bubble">
          <p>{task}</p>
        </div>
      </div>
      <div className="user-avatar">
        <img src="/api/current-user.png" alt="Current user" />
      </div>
      <MessageTime timestampMs={timestampMs} />
    </article>
  );
}

type AssistantMessageRowProps = {
  profile: string;
  timestampMs?: number;
  compact?: boolean;
  showIdentity?: boolean;
  children: React.ReactNode;
};

function AssistantMessageRow({
  profile,
  timestampMs,
  compact = false,
  showIdentity = true,
  children,
}: AssistantMessageRowProps) {
  return (
    <article
      className={`bot message-row assistant-message assistant-transcript chat-event-message teammate-message ${
        profileAccentClass(profile)
      }${compact ? " compact" : ""}${
        showIdentity ? "" : " speaker-continuation"
      }`}
    >
      <div className="bot-avatar">
        <img src={getAvatarForProfile(profile)} alt={profileName(profile)} />
      </div>
      <div className="message-container">
        <div className="author-line">
          <strong>{profileName(profile)}</strong>
          <span>{profileJobTitle(profile)}</span>
          {timestampMs ? <time>{formatEventTime(timestampMs)}</time> : null}
        </div>
        <div className="bubble thought-bubble">
          {children}
        </div>
      </div>
      <MessageTime timestampMs={timestampMs} />
    </article>
  );
}

function DeliveryPlanCard({
  workflow,
  timestampMs,
  showIdentity = true,
}: {
  workflow: WorkflowSummary;
  timestampMs?: number;
  showIdentity?: boolean;
}) {
  const envelope = workflow.plan;
  if (!envelope) return null;
  const plan = envelope.artifact;
  const review = workflow.plan_review?.artifact;
  const reviewMatchesPlan = review?.plan_sha256 === envelope.sha256;
  const reviewWasInvalidated = !reviewMatchesPlan &&
    workflow.stage === "blocked" &&
    workflow.paused_stage === "plan_review" &&
    workflow.blocked_reason?.includes("repository content changed");
  const reviewEndedIncomplete = !reviewMatchesPlan &&
    ["failed", "blocked", "cancelled"].includes(workflow.stage);
  const reviewLabel = reviewMatchesPlan
    ? review?.verdict === "pass"
      ? `${profileName("review")} reviewed`
      : "Changes requested"
    : reviewWasInvalidated
    ? "Review invalidated"
    : reviewEndedIncomplete
    ? workflow.stage === "cancelled" ? "Review cancelled" : "Review incomplete"
    : "Awaiting review";

  return (
    <AssistantMessageRow
      profile="plan"
      timestampMs={timestampMs}
      showIdentity={showIdentity}
    >
      <section className="delivery-plan" aria-label="Delivery plan">
        <div className="delivery-plan-heading">
          <span className="handoff-state">Delivery plan</span>
          <span className="delivery-plan-review">
            <i
              className={reviewMatchesPlan && review?.verdict === "pass"
                ? "bi bi-check2-circle"
                : reviewWasInvalidated
                ? "bi bi-exclamation-circle"
                : reviewEndedIncomplete
                ? "bi bi-dash-circle"
                : "bi bi-hourglass-split"}
              aria-hidden="true"
            >
            </i>
            {reviewLabel}
          </span>
        </div>
        <strong className="feedback-heading">
          <InlineRichText content={plan.summary} />
        </strong>
        <section>
          <h3>What it must achieve</h3>
          <ul>
            {plan.requirements.map((requirement) => (
              <li key={requirement.id}>
                <InlineRichText content={requirement.description} />
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h3>Implementation</h3>
          <ol>
            {plan.steps.map((step) => (
              <li key={step.id}>
                <span>
                  <InlineRichText content={step.description} />
                </span>
                {step.paths.length > 0
                  ? (
                    <div className="delivery-plan-paths">
                      {step.paths.map((path) => (
                        <code key={`${step.id}-${path.change}-${path.path}`}>
                          <span>{path.change}</span> {path.path}
                        </code>
                      ))}
                    </div>
                  )
                  : null}
              </li>
            ))}
          </ol>
        </section>
        <section>
          <h3>Done when</h3>
          <ul>
            {plan.acceptance.map((acceptance) => (
              <li key={acceptance.id}>
                <InlineRichText content={acceptance.description} />
              </li>
            ))}
          </ul>
        </section>
      </section>
    </AssistantMessageRow>
  );
}

function handoffOutcomeLabel(outcome?: string): string {
  switch (outcome) {
    case "ready":
      return "Ready to hand back";
    case "no_change":
      return "No code changes";
    case "checks_failed":
    case "repair_exhausted":
      return "Needs another pass";
    case "executor_unavailable":
    case "commit_blocked":
      return "Needs help";
    default:
      return "Checking the handoff";
  }
}

function TeamMessageBubble({
  envelope,
  events,
  showIdentity = true,
}: {
  envelope: EventEnvelope;
  events: EventEnvelope[];
  showIdentity?: boolean;
}) {
  const event = envelope.event;
  if (event.type !== "team_message") return null;
  const teammate = teamActorPresentation(event.actor);
  const accentClass = teamActorAccentClass(event.actor);

  const index = events.indexOf(envelope);
  const priorEvents = events.slice(0, index < 0 ? events.length : index);
  const recentPriorEvents = [...priorEvents].reverse();
  const followingEvents = events.slice(index < 0 ? 0 : index + 1);
  const evidenceIds = new Set(event.evidence_ids || []);
  const checkEvidence = Array.from(evidenceIds)
    .filter((id) => id.startsWith("check:"))
    .map((id) => id.slice("check:".length))
    .map((checkId) =>
      recentPriorEvents.find((candidate) =>
        candidate.event.type === "check_result" &&
        candidate.event.check_id === checkId
      )
    )
    .filter((candidate): candidate is EventEnvelope => Boolean(candidate));
  const commitEvidence = Array.from(evidenceIds)
    .filter((id) => id.startsWith("commit:"))
    .map((id) => id.slice("commit:".length))
    .map((oid) =>
      recentPriorEvents.find((candidate) =>
        candidate.event.type === "commit_result" && candidate.event.oid === oid
      )
    )
    .filter((candidate): candidate is EventEnvelope => Boolean(candidate));
  const handoff = followingEvents.find((candidate) =>
    candidate.event.type === "handoff_summary"
  );
  const summary = handoff?.event.type === "handoff_summary"
    ? handoff.event.summary
    : undefined;
  const start = events.find((candidate) => candidate.event.type === "started");
  const focusRoot = start?.event.type === "started"
    ? start.event.focus_root
    : undefined;
  const hasEvidence = checkEvidence.length > 0 || commitEvidence.length > 0 ||
    Boolean(summary) || Boolean(event.detail);

  return (
    <article
      className={`bot message-row assistant-message team-message chat-event-message tone-${event.tone}${
        accentClass ? ` ${accentClass}` : ""
      }${showIdentity ? "" : " speaker-continuation"}`}
    >
      <div className="bot-avatar team-avatar">
        <img src={teammate.avatar} alt={teammate.name} />
      </div>
      <div className="message-container">
        <div className="author-line">
          <strong>{teammate.name}</strong>
          <span>{teammate.role}</span>
          <span className="action-origin">{teammate.provenance}</span>
          {event.timestamp_ms
            ? <time>{formatEventTime(event.timestamp_ms)}</time>
            : null}
        </div>
        <div className="bubble thought-bubble team-bubble">
          <span className="handoff-state">
            {handoffOutcomeLabel(summary?.outcome)}
          </span>
          <RichText content={event.message} />
          {hasEvidence
            ? (
              <details className="handoff-evidence">
                <summary>What I ran</summary>
                {summary?.affected_components.length
                  ? (
                    <p className="handoff-scope">
                      <strong>Affected:</strong>{" "}
                      {summary.affected_components.join(", ")}
                    </p>
                  )
                  : null}
                {focusRoot
                  ? (
                    <p className="handoff-scope">
                      <strong>Focus:</strong> <code>{focusRoot}</code>
                    </p>
                  )
                  : null}
                {event.detail
                  ? <pre className="handoff-detail">{event.detail}</pre>
                  : null}
                {checkEvidence.map((candidate) => {
                  if (candidate.event.type !== "check_result") return null;
                  const check = candidate.event;
                  const status = check.skip_reason
                    ? "skipped"
                    : check.reused
                    ? "reused"
                    : check.success
                    ? "passed"
                    : "failed";
                  return (
                    <section
                      className={`handoff-check ${status}`}
                      key={`check-${check.check_id}`}
                    >
                      <div>
                        <strong>{check.check_id}</strong>
                        <span>{status}</span>
                      </div>
                      {check.command ? <code>{check.command}</code> : null}
                      <small>
                        {check.executor ? `Executor: ${check.executor}` : ""}
                        {check.cwd ? ` · From: ${check.cwd}` : ""}
                        {` · Exit: ${check.exit_status}`}
                      </small>
                      {check.skip_reason ? <p>{check.skip_reason}</p> : null}
                      {check.output?.trim() ? <pre>{check.output}</pre> : null}
                    </section>
                  );
                })}
                {commitEvidence.map((candidate) => {
                  if (candidate.event.type !== "commit_result") return null;
                  const commit = candidate.event;
                  return (
                    <section
                      className="handoff-commit"
                      key={`commit-${commit.oid}`}
                    >
                      <strong>
                        {commit.reused ? "Existing commit" : "Commit"}
                      </strong>
                      <code>{commit.oid?.slice(0, 12)} {commit.subject}</code>
                    </section>
                  );
                })}
              </details>
            )
            : null}
        </div>
      </div>
      <MessageTime timestampMs={event.timestamp_ms} />
    </article>
  );
}

function MetricField(
  { label, value }: { label: string; value: React.ReactNode },
) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function MetricsDialog({
  eyebrow,
  title,
  closeLabel,
  onClose,
  children,
}: {
  eyebrow: string;
  title: string;
  closeLabel: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const dialogTitleId = useId();

  useEffect(() => {
    const closeOnEscape = (keyboardEvent: KeyboardEvent) => {
      if (keyboardEvent.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="metrics-dialog-backdrop" onMouseDown={onClose}>
      <section
        className="metrics-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={dialogTitleId}
        onMouseDown={(mouseEvent) => mouseEvent.stopPropagation()}
      >
        <header>
          <div>
            <span>{eyebrow}</span>
            <h2 id={dialogTitleId}>{title}</h2>
          </div>
          <button
            className="btn btn-light btn-icon"
            type="button"
            autoFocus
            aria-label={closeLabel}
            onClick={onClose}
          >
            <i className="bi bi-x-lg" aria-hidden="true"></i>
          </button>
        </header>
        {children}
      </section>
    </div>
  );
}

function TechnicalDetailsBubble({
  className,
  detail,
  title,
  teammateName,
  children,
}: {
  className?: string;
  detail: string;
  title: string;
  teammateName: string;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const longPressTimer = useRef<number | undefined>(undefined);
  const hasDetail = Boolean(detail.trim());

  const cancelLongPress = () => {
    if (longPressTimer.current !== undefined) {
      window.clearTimeout(longPressTimer.current);
      longPressTimer.current = undefined;
    }
  };

  useEffect(() => cancelLongPress, []);

  return (
    <>
      <div
        className={`bubble thought-bubble ${className || ""}${
          hasDetail ? " technical-detail-surface" : ""
        }`.trim()}
        onPointerDown={hasDetail
          ? (pointerEvent) => {
            if (pointerEvent.pointerType === "mouse") return;
            cancelLongPress();
            longPressTimer.current = window.setTimeout(() => {
              setIsOpen(true);
              longPressTimer.current = undefined;
            }, 550);
          }
          : undefined}
        onPointerUp={hasDetail ? cancelLongPress : undefined}
        onPointerCancel={hasDetail ? cancelLongPress : undefined}
        onPointerLeave={hasDetail ? cancelLongPress : undefined}
        onContextMenu={hasDetail
          ? (contextMenuEvent) => contextMenuEvent.preventDefault()
          : undefined}
      >
        {children}
        {hasDetail
          ? (
            <button
              className="inference-info-button technical-detail-button"
              type="button"
              aria-label={`View technical details from ${teammateName}`}
              onPointerDown={(pointerEvent) => pointerEvent.stopPropagation()}
              onClick={() => setIsOpen(true)}
            >
              <i className="bi bi-info-circle" aria-hidden="true"></i>
            </button>
          )
          : null}
      </div>

      {hasDetail && isOpen
        ? (
          <MetricsDialog
            eyebrow="Technical details"
            title={title}
            closeLabel="Close technical details"
            onClose={() => setIsOpen(false)}
          >
            <pre className="technical-detail-content">{detail}</pre>
          </MetricsDialog>
        )
        : null}
    </>
  );
}

function ActionInferenceDetails({
  events,
  teammate,
}: {
  events: EventEnvelope[];
  teammate: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const inferences = events.flatMap((envelope) =>
    envelope.event.type === "llm_invocation" ? [envelope.event] : []
  );
  if (inferences.length === 0) return null;

  const durationMs = inferences.reduce(
    (total, event) => total + event.duration_ms,
    0,
  );
  const tokens = inferences.reduce(
    (total, event) => total + event.prompt_tokens + event.generated_tokens,
    0,
  );
  const teammateFirstName = teammate.split(/\s+/, 1)[0];

  return (
    <>
      <div
        className="inference-marker action-inference-marker"
        aria-label={`${teammate} model work summary`}
      >
        <span>
          {teammateFirstName} worked for {formatHumanDurationMs(durationMs)}
          {inferences.length > 1
            ? ` across ${inferences.length} model calls`
            : ""}
        </span>
        <button
          className="inference-info-button action-run-info"
          type="button"
          aria-label={`View ${inferences.length} model call detail${
            inferences.length === 1 ? "" : "s"
          } for ${teammate}`}
          onClick={() => setIsOpen(true)}
        >
          <i className="bi bi-info-circle" aria-hidden="true"></i>
        </button>
      </div>
      {isOpen
        ? (
          <MetricsDialog
            eyebrow="Model call details"
            title={`${teammate} · ${inferences.length} call${
              inferences.length === 1 ? "" : "s"
            }`}
            closeLabel="Close model call details"
            onClose={() => setIsOpen(false)}
          >
            <dl className="metrics-grid">
              <MetricField
                label="Total duration"
                value={formatHumanDurationMs(durationMs)}
              />
              <MetricField label="Total tokens" value={formatNumber(tokens)} />
            </dl>
            <div className="run-inference-list">
              {inferences.map((event, index) => {
                const totalTokens = event.prompt_tokens +
                  event.generated_tokens;
                return (
                  <details
                    className="metrics-section run-inference-step"
                    key={`${event.step}-${index}`}
                  >
                    <summary>
                      <span>Step {event.step}</span>
                      <small>
                        {sentenceCaseIdentifier(
                          event.purpose || "unclassified",
                        )} · {formatHumanDurationMs(event.duration_ms)}
                      </small>
                    </summary>
                    <dl className="metrics-grid">
                      <MetricField
                        label="Workflow stage"
                        value={sentenceCaseIdentifier(
                          event.workflow_stage || "none",
                        )}
                      />
                      <MetricField
                        label="Tokens"
                        value={formatNumber(totalTokens)}
                      />
                      <MetricField
                        label="Prompt"
                        value={formatNumber(event.prompt_tokens)}
                      />
                      <MetricField
                        label="Generated"
                        value={formatNumber(event.generated_tokens)}
                      />
                      {event.prompt_cache
                        ? (
                          <>
                            <MetricField
                              label="Cache reused"
                              value={formatNumber(
                                event.prompt_cache.cached_tokens,
                              )}
                            />
                            <MetricField
                              label="Fresh prompt"
                              value={formatNumber(
                                event.prompt_cache.prefilled_tokens,
                              )}
                            />
                            {event.prompt_cache.miss_reason
                              ? (
                                <MetricField
                                  label="Cache miss"
                                  value={sentenceCaseIdentifier(
                                    event.prompt_cache.miss_reason,
                                  )}
                                />
                              )
                              : null}
                          </>
                        )
                        : null}
                      {event.energy_joules !== undefined
                        ? (
                          <MetricField
                            label="Energy"
                            value={formatEnergy(event.energy_joules)}
                          />
                        )
                        : null}
                      {event.native
                        ? (
                          <>
                            <MetricField
                              label="Model"
                              value={event.native.model_family}
                            />
                            <MetricField
                              label="Prefill"
                              value={`${
                                event.native.prefill_tokens_per_second.toFixed(
                                  1,
                                )
                              } tok/s`}
                            />
                            <MetricField
                              label="Decode"
                              value={`${
                                event.native.decode_tokens_per_second.toFixed(1)
                              } tok/s`}
                            />
                            {event.native.constraint_terminal_state
                              ? (
                                <MetricField
                                  label="Control collar result"
                                  value={sentenceCaseIdentifier(
                                    event.native.constraint_terminal_state,
                                  )}
                                />
                              )
                              : null}
                            {event.native.rejected_constraint_candidates > 0
                              ? (
                                <MetricField
                                  label="Control collar filtered"
                                  value={`${
                                    formatNumber(
                                      event.native
                                        .rejected_constraint_candidates,
                                    )
                                  } invalid output candidates`}
                                />
                              )
                              : null}
                            {mutationCandidateCount(event) > 0
                              ? (
                                <MetricField
                                  label="Mutation candidates"
                                  value={`${
                                    formatNumber(mutationCandidateCount(event))
                                  } stopped before sampling`}
                                />
                              )
                              : null}
                          </>
                        )
                        : null}
                    </dl>
                  </details>
                );
              })}
            </div>
          </MetricsDialog>
        )
        : null}
    </>
  );
}

function InferenceDetails({
  event,
  activityProfile,
}: {
  event: Extract<AgentEvent, { type: "llm_invocation" }>;
  activityProfile?: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const longPressTimer = useRef<number | undefined>(undefined);
  const profile = event.profile || activityProfile || "build";
  const teammate = profileName(profile);
  const teammateFirstName = teammate.split(/\s+/, 1)[0];
  const totalTokens = event.prompt_tokens + event.generated_tokens;

  const cancelLongPress = () => {
    if (longPressTimer.current !== undefined) {
      window.clearTimeout(longPressTimer.current);
      longPressTimer.current = undefined;
    }
  };

  useEffect(() => cancelLongPress, []);

  return (
    <>
      <article
        className="inference-marker"
        aria-label={`${teammate} model inference ${event.step}`}
        onPointerDown={(pointerEvent) => {
          if (pointerEvent.pointerType === "mouse") return;
          cancelLongPress();
          longPressTimer.current = window.setTimeout(() => {
            setIsOpen(true);
            longPressTimer.current = undefined;
          }, 550);
        }}
        onPointerUp={cancelLongPress}
        onPointerCancel={cancelLongPress}
        onPointerLeave={cancelLongPress}
        onContextMenu={(contextMenuEvent) => contextMenuEvent.preventDefault()}
      >
        <span>
          {teammateFirstName} worked for{" "}
          {formatHumanDurationMs(event.duration_ms)}
        </span>
        <button
          className="inference-info-button"
          type="button"
          aria-label={`View inference ${event.step} details`}
          onPointerDown={(pointerEvent) => pointerEvent.stopPropagation()}
          onClick={() => setIsOpen(true)}
        >
          <i className="bi bi-info-circle" aria-hidden="true"></i>
        </button>
      </article>

      {isOpen
        ? (
          <MetricsDialog
            eyebrow="Inference details"
            title={`${teammate} · step ${event.step}`}
            closeLabel="Close inference details"
            onClose={() => setIsOpen(false)}
          >
            <dl className="metrics-grid">
              <MetricField
                label="Purpose"
                value={sentenceCaseIdentifier(
                  event.purpose || "unclassified",
                )}
              />
              <MetricField
                label="Workflow stage"
                value={sentenceCaseIdentifier(event.workflow_stage || "none")}
              />
              <MetricField
                label="Duration"
                value={formatHumanDurationMs(event.duration_ms)}
              />
              <MetricField
                label="Tokens"
                value={formatNumber(totalTokens)}
              />
              <MetricField
                label="Prompt"
                value={formatNumber(event.prompt_tokens)}
              />
              <MetricField
                label="Generated"
                value={formatNumber(event.generated_tokens)}
              />
              {event.energy_joules !== undefined
                ? (
                  <MetricField
                    label="Energy"
                    value={`${formatEnergy(event.energy_joules)}${
                      event.average_power_watts === undefined
                        ? ""
                        : ` at ${formatPower(event.average_power_watts)}`
                    }`}
                  />
                )
                : null}
            </dl>

            {event.prompt_cache
              ? (
                <section className="metrics-section">
                  <h3>Prompt cache</h3>
                  <dl className="metrics-grid">
                    <MetricField
                      label="Source"
                      value={sentenceCaseIdentifier(
                        event.prompt_cache.source,
                      )}
                    />
                    <MetricField
                      label="Reused"
                      value={formatNumber(event.prompt_cache.cached_tokens)}
                    />
                    <MetricField
                      label="Fresh"
                      value={formatNumber(
                        event.prompt_cache.prefilled_tokens,
                      )}
                    />
                    {event.prompt_cache.miss_reason
                      ? (
                        <MetricField
                          label="Miss reason"
                          value={sentenceCaseIdentifier(
                            event.prompt_cache.miss_reason,
                          )}
                        />
                      )
                      : null}
                    {event.prompt_cache.lookup_detail
                      ? (
                        <MetricField
                          label="Lookup"
                          value={sentenceCaseIdentifier(
                            event.prompt_cache.lookup_detail,
                          )}
                        />
                      )
                      : null}
                    {event.prompt_cache.root
                      ? (
                        <MetricField
                          label="Stable root"
                          value={`${
                            formatNumber(
                              event.prompt_cache.root.reused_tokens,
                            )
                          } of ${
                            formatNumber(event.prompt_cache.root.tokens)
                          } tokens · ${
                            sentenceCaseIdentifier(
                              event.prompt_cache.root.authority_class,
                            )
                          }`}
                        />
                      )
                      : null}
                  </dl>
                </section>
              )
              : null}

            {event.native
              ? (
                <section className="metrics-section">
                  <h3>Local runtime</h3>
                  <dl className="metrics-grid">
                    <MetricField
                      label="Model family"
                      value={event.native.model_family}
                    />
                    <MetricField
                      label="Prefill"
                      value={`${
                        formatNumber(event.native.fresh_prefill_tokens)
                      } tokens · ${
                        event.native.prefill_tokens_per_second.toFixed(1)
                      } tok/s`}
                    />
                    <MetricField
                      label="Decode"
                      value={`${
                        formatNumber(event.native.decode_tokens)
                      } tokens · ${
                        event.native.decode_tokens_per_second.toFixed(1)
                      } tok/s`}
                    />
                    <MetricField
                      label="Strategy"
                      value={sentenceCaseIdentifier(
                        event.native.expert_strategy,
                      )}
                    />
                    {event.native.tool_constraint_mode
                      ? (
                        <MetricField
                          label="Control collar"
                          value={sentenceCaseIdentifier(
                            event.native.tool_constraint_mode,
                          )}
                        />
                      )
                      : null}
                    {event.native.rejected_constraint_candidates > 0
                      ? (
                        <MetricField
                          label="Candidates filtered"
                          value={`${
                            formatNumber(
                              event.native.rejected_constraint_candidates,
                            )
                          } invalid output candidates`}
                        />
                      )
                      : null}
                    {mutationCandidateCount(event) > 0
                      ? (
                        <MetricField
                          label="Mutation candidates"
                          value={`${
                            formatNumber(mutationCandidateCount(event))
                          } stopped before sampling`}
                        />
                      )
                      : null}
                    {event.native.constraint_terminal_state
                      ? (
                        <MetricField
                          label="Collar result"
                          value={sentenceCaseIdentifier(
                            event.native.constraint_terminal_state,
                          )}
                        />
                      )
                      : null}
                    {event.native.refill
                      ? (
                        <MetricField
                          label="Refill timing"
                          value={`lookup ${event.native.refill.cache_lookup_wall_ms} ms · disk ${event.native.refill.disk_read_decode_wall_ms} ms · validate ${event.native.refill.cpu_state_validation_allocation_wall_ms} ms · hydrate ${event.native.refill.state_hydration_wall_ms} ms · suffix ${event.native.refill.fresh_suffix_prefill_wall_ms} ms · snapshot ${event.native.refill.snapshot_capture_wall_ms} ms · queue ${event.native.refill.persistence_queue_wall_ms} ms`}
                        />
                      )
                      : null}
                  </dl>
                </section>
              )
              : null}
          </MetricsDialog>
        )
        : null}
    </>
  );
}

function SessionMetricsDetails({
  event,
  evidenceEvents,
}: {
  event: Extract<AgentEvent, { type: "session_metrics" }>;
  evidenceEvents: EventEnvelope[];
}) {
  const [isOpen, setIsOpen] = useState(false);
  const longPressTimer = useRef<number | undefined>(undefined);
  const totalTokens = event.prompt_tokens + event.generated_tokens;
  const totalEnergyJoules = metricEnergyJoules(event);
  const totalRuntimeMs = metricRuntimeMs(event);
  const coverage = event.energy_coverage === undefined
    ? "Unknown"
    : `${Math.round(event.energy_coverage * 100)}%`;
  const hasMeasurementMetadata = (event.wall_runtime_ms ?? 0) > 0 ||
    event.total_energy_joules !== undefined ||
    event.energy_source !== undefined;
  const hasCachePersistence =
    (event.cache_persistence_queued_checkpoints ?? 0) > 0 ||
    (event.cache_persistence_failures ?? 0) > 0;
  const efficiency = harnessEfficiencyStats(evidenceEvents);
  const assistanceSummary = trinityAssistanceSummary(efficiency);
  const preventedActions = efficiency.duplicateActionsPrevented +
    efficiency.dependentBatchesPrevented;
  const hasHarnessEfficiency = efficiency.proactiveActions > 0 ||
    efficiency.collarCandidatesFiltered > 0 || preventedActions > 0 ||
    efficiency.noProgressLoopsStopped > 0;

  const cancelLongPress = () => {
    if (longPressTimer.current !== undefined) {
      window.clearTimeout(longPressTimer.current);
      longPressTimer.current = undefined;
    }
  };

  useEffect(() => cancelLongPress, []);

  return (
    <>
      <article
        className="session-correction session-metrics-summary"
        aria-label="Session runtime summary"
        onPointerDown={(pointerEvent) => {
          if (pointerEvent.pointerType === "mouse") return;
          cancelLongPress();
          longPressTimer.current = window.setTimeout(() => {
            setIsOpen(true);
            longPressTimer.current = undefined;
          }, 550);
        }}
        onPointerUp={cancelLongPress}
        onPointerCancel={cancelLongPress}
        onPointerLeave={cancelLongPress}
        onContextMenu={(contextMenuEvent) => contextMenuEvent.preventDefault()}
      >
        <span className="session-metrics-copy">
          {assistanceSummary
            ? (
              <span className="trinity-efficiency-summary">
                <i className="bi bi-stars" aria-hidden="true"></i>
                <span>{assistanceSummary}</span>
              </span>
            )
            : null}
          <span>
            {funEnergySummary(totalRuntimeMs, totalTokens, totalEnergyJoules)}
            {totalEnergyJoules === undefined && event.energy_exclusive === false
              ? " Power estimate unavailable."
              : null}
          </span>
        </span>
        <button
          className="inference-info-button"
          type="button"
          aria-label="View session runtime details"
          onPointerDown={(pointerEvent) => pointerEvent.stopPropagation()}
          onClick={() => setIsOpen(true)}
        >
          <i className="bi bi-info-circle" aria-hidden="true"></i>
        </button>
        {event.timestamp_ms
          ? <time>{formatEventTime(event.timestamp_ms)}</time>
          : null}
      </article>

      {isOpen
        ? (
          <MetricsDialog
            eyebrow="Runtime details"
            title="Session totals"
            closeLabel="Close session runtime details"
            onClose={() => setIsOpen(false)}
          >
            <dl className="metrics-grid">
              <MetricField
                label="Duration"
                value={formatHumanDurationMs(totalRuntimeMs)}
              />
              <MetricField label="Tokens" value={formatNumber(totalTokens)} />
              <MetricField
                label="Model calls"
                value={formatNumber(event.llm_invocations)}
              />
              <MetricField
                label="Tool calls"
                value={formatNumber(event.tool_calls)}
              />
              <MetricField
                label="Prompt"
                value={formatNumber(event.prompt_tokens)}
              />
              <MetricField
                label="Generated"
                value={formatNumber(event.generated_tokens)}
              />
            </dl>

            {hasHarnessEfficiency
              ? (
                <section className="metrics-section trinity-efficiency-section">
                  <h3>
                    <i className="bi bi-stars" aria-hidden="true"></i>
                    How pb helped
                  </h3>
                  <dl className="metrics-grid">
                    {efficiency.proactiveActions > 0
                      ? (
                        <>
                          <MetricField
                            label="Trinity acted early"
                            value={`${
                              formatNumber(efficiency.proactiveActions)
                            } repository ${
                              efficiency.proactiveActions === 1
                                ? "action"
                                : "actions"
                            }`}
                          />
                          <MetricField
                            label="Potential turns avoided"
                            value={`Up to ${
                              formatNumber(efficiency.proactiveActions)
                            }`}
                          />
                        </>
                      )
                      : null}
                    {preventedActions > 0
                      ? (
                        <MetricField
                          label="Actions prevented"
                          value={`${
                            formatNumber(preventedActions)
                          } before execution`}
                        />
                      )
                      : null}
                    {efficiency.noProgressLoopsStopped > 0
                      ? (
                        <MetricField
                          label="Loops stopped"
                          value={formatNumber(
                            efficiency.noProgressLoopsStopped,
                          )}
                        />
                      )
                      : null}
                    {efficiency.collarCandidatesFiltered > 0
                      ? (
                        <MetricField
                          label="Control collar"
                          value={`${
                            formatNumber(
                              efficiency.collarCandidatesFiltered,
                            )
                          } invalid output candidates filtered${
                            efficiency.mutationCandidatesFiltered > 0
                              ? `, including ${
                                formatNumber(
                                  efficiency.mutationCandidatesFiltered,
                                )
                              } mutation candidates`
                              : ""
                          }`}
                        />
                      )
                      : null}
                  </dl>
                  {efficiency.proactiveActions > 0
                    ? (
                      <p className="metrics-note">
                        Potential turns avoided is an upper bound based on work
                        Trinity completed before the next model call. Energy is
                        measured session use, not a counterfactual saving.
                      </p>
                    )
                    : null}
                </section>
              )
              : null}

            {totalEnergyJoules !== undefined
              ? (
                <section className="metrics-section">
                  <h3>Energy</h3>
                  <dl className="metrics-grid">
                    <MetricField
                      label="Total"
                      value={formatEnergy(totalEnergyJoules)}
                    />
                    <MetricField
                      label="Average power"
                      value={formatPower(event.average_power_watts)}
                    />
                    <MetricField
                      label="Model inference"
                      value={formatEnergy(event.llm_energy_joules)}
                    />
                    <MetricField
                      label="Tools"
                      value={formatEnergy(event.tool_energy_joules)}
                    />
                    {hasMeasurementMetadata
                      ? (
                        <>
                          <MetricField
                            label="Gross device energy"
                            value={formatEnergy(event.gross_energy_joules)}
                          />
                          <MetricField
                            label="After adjustment"
                            value={formatEnergy(event.adjusted_energy_joules)}
                          />
                          <MetricField label="Coverage" value={coverage} />
                          <MetricField
                            label="Source"
                            value={sentenceCaseIdentifier(
                              event.energy_source || "unknown",
                            )}
                          />
                          <MetricField
                            label="Excluded"
                            value={[
                              event.display_energy_excluded
                                ? "Measured display"
                                : null,
                              event.idle_baseline_applied
                                ? "Idle device baseline"
                                : null,
                            ].filter(Boolean).join(", ") || "None available"}
                          />
                          <MetricField
                            label="Estimate"
                            value={event.energy_complete === false
                              ? "Partial or changed source"
                              : "Complete"}
                          />
                          {event.energy_exclusive === false
                            ? (
                              <MetricField
                                label="Attribution"
                                value="Shared system meter; task-only attribution unavailable"
                              />
                            )
                            : null}
                        </>
                      )
                      : null}
                  </dl>
                </section>
              )
              : null}

            {hasCachePersistence
              ? (
                <section className="metrics-section">
                  <h3>Cache persistence</h3>
                  <dl className="metrics-grid">
                    <MetricField
                      label="Checkpoints"
                      value={`${
                        formatNumber(
                          event.cache_persistence_completed_checkpoints ?? 0,
                        )
                      } of ${
                        formatNumber(
                          event.cache_persistence_queued_checkpoints ?? 0,
                        )
                      }`}
                    />
                    <MetricField
                      label="Duration"
                      value={`${
                        formatNumber(
                          event.cache_persistence_wall_ms ?? 0,
                        )
                      } ms`}
                    />
                    <MetricField
                      label="Failures"
                      value={formatNumber(
                        event.cache_persistence_failures ?? 0,
                      )}
                    />
                  </dl>
                </section>
              )
              : null}
          </MetricsDialog>
        )
        : null}
    </>
  );
}

export function MessageBubble({
  envelope,
  activityProfile,
  evidenceEvents = [],
  workflow,
  showIdentity = true,
}: {
  envelope: EventEnvelope;
  activityProfile?: string;
  evidenceEvents?: EventEnvelope[];
  workflow?: WorkflowSummary;
  showIdentity?: boolean;
}) {
  const e = envelope.event;

  switch (e.type) {
    case "started":
      return (
        <article
          className={`user message-row user-message${
            showIdentity ? "" : " speaker-continuation"
          }`}
        >
          <div className="message-container">
            <div className="author-line">
              <strong>You</strong>
              <time>{formatEventTime(e.timestamp_ms)}</time>
            </div>
            <div className="bubble user-bubble">
              <p>{e.task}</p>
            </div>
          </div>
          <div className="user-avatar">
            <img src="/api/current-user.png" alt="Current user" />
          </div>
          <MessageTime timestampMs={e.timestamp_ms} />
        </article>
      );

    case "model_loading":
    case "step_started": {
      const label = e.type === "model_loading"
        ? "Loading model"
        : `Working step ${e.step} of ${e.max_steps}`;
      return (
        <article
          className="message-row assistant-message compact typing-row"
          aria-label={label}
        >
          <div className="bot-avatar typing-avatar">
            <img
              src={getAvatarForProfile(e.profile)}
              alt=""
              aria-hidden="true"
            />
          </div>
          <div className="typing-indicator" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </article>
      );
    }

    case "reasoning":
      return (
        <AssistantMessageRow
          profile={e.profile}
          timestampMs={e.timestamp_ms}
          showIdentity={showIdentity}
        >
          <RichText content={e.content} />
        </AssistantMessageRow>
      );

    case "user_question":
      return (
        <AssistantMessageRow
          profile={e.profile}
          timestampMs={e.timestamp_ms}
          showIdentity={showIdentity}
        >
          <div className="assistant-question">
            <RichText content={e.question} />
            {e.choices?.length
              ? (
                <div className="question-choices">
                  {e.choices.map((choice) => (
                    <span key={choice}>
                      <InlineRichText content={choice} />
                    </span>
                  ))}
                </div>
              )
              : null}
          </div>
        </AssistantMessageRow>
      );

    case "workflow_challenge_raised":
      return (
        <article className="session-correction" aria-label="Workflow challenge">
          <strong>{e.severity.toUpperCase()} challenge</strong>
          <span>{e.summary}</span>
          {e.timestamp_ms
            ? <time>{formatEventTime(e.timestamp_ms)}</time>
            : null}
        </article>
      );

    case "workflow_blocked":
      return (
        <WorkflowBlockedNotice
          event={e}
          envelope={envelope}
          showIdentity={showIdentity}
        />
      );

    case "workflow_evidence_invalidated":
      return (
        <article
          className="session-correction"
          aria-label="Workflow evidence invalidated"
        >
          <span>{e.reason}</span>
          {e.timestamp_ms
            ? <time>{formatEventTime(e.timestamp_ms)}</time>
            : null}
        </article>
      );

    case "correction":
      return (
        <CorrectionNotice
          envelope={envelope}
          showIdentity={showIdentity}
        />
      );

    case "team_message":
      return (
        <TeamMessageBubble
          envelope={envelope}
          events={evidenceEvents}
          showIdentity={showIdentity}
        />
      );

    case "user_answer":
      return (
        <article
          className={`user message-row user-message compact${
            showIdentity ? "" : " speaker-continuation"
          }`}
        >
          <div className="message-container">
            <div className="author-line">
              <strong>You</strong>
              <span>Answer</span>
              <time>{formatEventTime(e.timestamp_ms)}</time>
            </div>
            <div className="bubble user-bubble">
              <p>{e.answer}</p>
            </div>
          </div>
          <div className="user-avatar">
            <img src="/api/current-user.png" alt="Current user" />
          </div>
          <MessageTime timestampMs={e.timestamp_ms} />
        </article>
      );

    case "user_message":
      return (
        <article
          className={`user message-row user-message compact${
            showIdentity ? "" : " speaker-continuation"
          }`}
        >
          <div className="message-container">
            <div className="author-line">
              <strong>You</strong>
              <span>Message to running agent</span>
              {e.timestamp_ms
                ? <time>{formatEventTime(e.timestamp_ms)}</time>
                : null}
            </div>
            <div className="bubble user-bubble">
              <p>{e.message}</p>
            </div>
          </div>
          <div className="user-avatar">
            <img src="/api/current-user.png" alt="Current user" />
          </div>
          <MessageTime timestampMs={e.timestamp_ms} />
        </article>
      );

    case "user_message_applied":
      return null;

    case "sub_agent_started":
      return (
        <article className="message-row assistant-message compact">
          <div className="bot-avatar">
            <img src={getAvatarForProfile(e.profile)} alt={e.profile} />
          </div>
          <div className="bubble thought-bubble">
            <p>
              <span className="badge bg-primary me-2">{e.profile}</span>
              {e.task}
            </p>
          </div>
        </article>
      );

    case "sub_agent_finished":
      return (
        <article className="message-row assistant-message compact">
          <div className="bot-avatar">
            <i className="bi bi-stars"></i>
          </div>
          <details open className="bubble thought-bubble">
            <summary style={{ display: "none" }} />
            <p>Sub-agent {e.profile} completed</p>
            <pre className="mb-0 small result-pre">{e.result}</pre>
          </details>
        </article>
      );

    case "diff":
      return (
        <article className="message-row assistant-message diff-message">
          <div className="bot-avatar">
            <i className="bi bi-stars"></i>
          </div>
          <details open className="bubble thought-bubble">
            <summary style={{ display: "none" }} />
            <p>
              <code>{e.path}</code> changed
            </p>
            <details open className="transcript-diff">
              <summary className="transcript-diff-header">
                <span>Diff</span>
              </summary>
              <div className="transcript-diff-body">
                <DiffView diff={e.diff} />
              </div>
            </details>
          </details>
        </article>
      );

    case "final":
      return (
        <AssistantMessageRow
          profile={e.profile}
          timestampMs={e.timestamp_ms}
          showIdentity={showIdentity}
        >
          <RichText content={e.content} />
        </AssistantMessageRow>
      );

    case "llm_invocation":
      return <InferenceDetails event={e} activityProfile={activityProfile} />;

    case "workflow_artifact_accepted":
      return e.artifact_kind === "plan" &&
          workflow?.plan?.sha256 === e.sha256
        ? (
          <DeliveryPlanCard
            workflow={workflow}
            timestampMs={e.timestamp_ms}
            showIdentity={showIdentity}
          />
        )
        : null;

    case "executor_started":
    case "task_plan_accepted":
    case "task_plan_rejected":
    case "tasks_changed":
    case "workflow_started":
    case "workflow_resumed":
    case "workflow_stage_started":
    case "workflow_stage_completed":
    case "workflow_completed":
    case "check_result":
    case "commit_result":
    case "handoff_summary":
    case "final_grace":
      return null;

    case "session_metrics":
      return (
        <SessionMetricsDetails
          event={e}
          evidenceEvents={evidenceEvents}
        />
      );

    case "session_summary": {
      const hasSummary = Boolean(e.summary?.trim());
      const commitLines = trustedSessionSummaryCommitLines(
        e.commits,
        evidenceEvents,
      );
      const hasChanges = Boolean(e.diff_stat?.trim() || e.diff?.trim());
      if (!hasSummary && commitLines.length === 0 && !hasChanges) return null;
      return (
        <article
          className={`message-row assistant-message compact delivery-summary chat-event-message${
            showIdentity ? "" : " speaker-continuation"
          }`}
        >
          <div className="bot-avatar delivery-avatar">
            <i className="bi bi-box-seam" aria-hidden="true"></i>
          </div>
          <div className="message-container">
            <div className="author-line">
              <strong>Delivery summary</strong>
              {e.timestamp_ms
                ? <time>{formatEventTime(e.timestamp_ms)}</time>
                : null}
            </div>
            <div className="bubble thought-bubble">
              {hasSummary ? <RichText content={e.summary!.trim()} /> : null}
              {commitLines.length > 0
                ? (
                  <section className="delivery-commits">
                    <strong>Commits from this delivery</strong>
                    <ul>
                      {commitLines.map((line) => (
                        <li key={line}>
                          <code>{line}</code>
                        </li>
                      ))}
                    </ul>
                  </section>
                )
                : null}
              {e.diff_stat?.trim()
                ? (
                  <section className="delivery-changes">
                    <strong>Changes</strong>
                    <pre className="small result-pre">{e.diff_stat}</pre>
                  </section>
                )
                : null}
              {e.diff?.trim()
                ? (
                  <details className="transcript-diff">
                    <summary className="transcript-diff-header">
                      <span>View changes</span>
                    </summary>
                    <div className="transcript-diff-body">
                      <DiffView diff={e.diff} />
                    </div>
                  </details>
                )
                : null}
            </div>
          </div>
          <MessageTime timestampMs={e.timestamp_ms} />
        </article>
      );
    }

    case "error":
      return <ErrorEventBubble event={e} />;

    default:
      return null;
  }
}

export function SessionCard({
  session,
  onClick,
}: {
  session: SessionItem;
  onClick: () => void;
}) {
  let badge: React.ReactNode;
  if (session.goal) {
    const goal = session.goal;
    const text = goal.active
      ? goal.stage === "awaiting_user_review"
        ? "Goal ready for review"
        : goal.stage === "blocked"
        ? "Goal needs help"
        : goal.stage === "paused"
        ? "Goal paused"
        : `Goal · ${goal.completed_milestones}/${goal.total_milestones}`
      : goal.stage === "completed"
      ? "Goal complete"
      : goal.stage === "cancelled"
      ? "Goal cancelled"
      : goal.outcome === "budget_exhausted"
      ? "Goal budget reached"
      : "Goal stopped";
    const style = goal.active
      ? goal.stage === "blocked" || goal.stage === "awaiting_user_review" ||
          goal.stage === "paused"
        ? "bg-warning text-dark"
        : "bg-primary"
      : goal.stage === "completed"
      ? "bg-success"
      : "bg-secondary";
    badge = <span className={`badge ${style}`}>{text}</span>;
  } else if (session.status === "running") {
    badge = (
      <span className="badge bg-primary d-flex align-items-center gap-1">
        <span
          className="spinner-border spinner-border-sm"
          style={{ width: "0.6rem", height: "0.6rem" }}
        />
        Running
      </span>
    );
  } else if (session.status === "queued") {
    badge = <span className="badge bg-info text-dark">Queued</span>;
  } else if (session.status === "paused") {
    badge = session.pending_question
      ? <span className="badge bg-warning text-dark">Needs answer</span>
      : (
        <span className="badge bg-warning text-dark">Paused after restart</span>
      );
  } else if (
    session.workflow_outcome === "no_change" ||
    session.handoff_outcome === "no_change"
  ) {
    badge = <span className="badge bg-secondary">No code changes</span>;
  } else if (
    session.workflow_outcome === "ready" || session.handoff_outcome === "ready"
  ) {
    badge = <span className="badge bg-success">Ready</span>;
  } else if (
    session.workflow_outcome === "checks_failed" ||
    session.workflow_outcome === "review_failed" ||
    session.workflow_outcome === "repair_cycles_exhausted" ||
    session.workflow_outcome === "contract_unsatisfied" ||
    session.handoff_outcome === "checks_failed" ||
    session.handoff_outcome === "repair_exhausted"
  ) {
    badge = (
      <span className="badge bg-warning text-dark">Needs another pass</span>
    );
  } else if (
    session.workflow_outcome === "executor_unavailable" ||
    session.workflow_outcome === "commit_blocked" ||
    session.handoff_outcome === "executor_unavailable" ||
    session.handoff_outcome === "commit_blocked"
  ) {
    badge = <span className="badge bg-danger">Needs help</span>;
  } else if (session.workflow_outcome === "cancelled") {
    badge = <span className="badge bg-secondary">Cancelled</span>;
  } else if (session.status === "failed") {
    badge = <span className="badge bg-danger">Stopped</span>;
  } else if (session.branch) {
    badge = <span className="badge bg-success">{session.branch}</span>;
  } else {
    badge = <span className="badge bg-secondary">Completed</span>;
  }

  return (
    <button
      type="button"
      className="list-group-item list-group-item-action py-2"
      onClick={onClick}
    >
      <div className="d-flex justify-content-between align-items-start gap-2">
        <div className="fw-semibold text-truncate flex-grow-1 small">
          {sessionTitle(session)}
        </div>
        {badge}
      </div>
      <div className="small mt-1 text-body-secondary">
        <span className="me-2">{projectName(session.workdir)}</span>
        <span>{relativeTime(session.updated_at_ms)}</span>
      </div>
    </button>
  );
}
