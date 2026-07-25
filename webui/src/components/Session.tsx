import type React from "react";
import { Fragment, useId, useState } from "react";
import type { AgentEvent, EventEnvelope, SessionItem } from "../types";
import { TOOL_FRIENDLY_NAMES, TOOL_ICONS } from "../lib/constants";
import {
  formatEventTime,
  getAvatarForProfile,
  projectName,
  relativeTime,
  sessionTitle,
  toolResultForCall,
} from "../lib/helpers";
import {
  errorSummary,
  getToolDetail,
  profileJobTitle,
  profileName,
  TODO_STATUS_LABELS,
} from "../lib/sessionUtils";
import type { ActionTimelineItem, TodoTask } from "../lib/sessionUtils";
import { parseRichText } from "../lib/richText";
import {
  formatEnergy,
  formatPower,
  ledEquivalent,
  metricEnergyJoules,
  metricRuntimeMs,
} from "../lib/energy";
import { teamActorPresentation, workflowStewardActor } from "../lib/team";

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
            return <Heading key={index}>{block.text}</Heading>;
          }
          case "unordered_list":
            return (
              <ul key={index}>
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{item}</li>
                ))}
              </ul>
            );
          case "ordered_list":
            return (
              <ol key={index}>
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{item}</li>
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
                    {line}
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
} | null {
  switch (event.type) {
    case "controller_observation":
      return {
        label: `${
          event.receipt.operation === "read_file" ? "Read" : "Inspected"
        } ${event.receipt.path}`,
        detail: event.receipt.coverage === "full"
          ? `${event.receipt.observed_bytes.toLocaleString()} bytes · full coverage`
          : `${event.receipt.observed_bytes.toLocaleString()} bytes · bounded ranges`,
        icon: event.receipt.operation === "read_file"
          ? "bi bi-file-earmark-text"
          : "bi bi-search",
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
  toolCalls,
  toolResults,
  controllerActions,
}: {
  actor?: import("../types").TeamActor;
  assistingProfile?: string;
  toolCalls: EventEnvelope[];
  toolResults: EventEnvelope[];
  controllerActions: EventEnvelope[];
}) {
  const [isOpen, setIsOpen] = useState(false);
  const collapseId = useId();

  if (toolCalls.length === 0 && controllerActions.length === 0) return null;

  const teammate = teamActorPresentation(actor);

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
        className="tool-item success automatic-action"
      >
        <i className={presentation.icon}></i>
        <span className="action-label">
          <span>{presentation.label}</span>
        </span>
        <small>{presentation.detail}</small>
      </div>
    );
  }).filter(Boolean);

  const actionNames = toolCalls
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
    .filter(Boolean)
    .join(" · ");
  const actionCount = toolCalls.length + controllerActions.length;
  const assisting = assistingProfile
    ? profileName(assistingProfile)
    : undefined;
  const actionSummary = actor?.kind === "automation"
    ? `${actionCount} routine ${actionCount === 1 ? "action" : "actions"}${
      assisting ? ` while assisting ${assisting}` : ""
    }`
    : `${actionCount} ${actionCount === 1 ? "action" : "actions"}`;
  const firstEvent = toolCalls[0] || controllerActions[0];
  const timestampMs = firstEvent && "timestamp_ms" in firstEvent.event
    ? firstEvent.event.timestamp_ms
    : undefined;

  return (
    <article className="bot message-row assistant-message compact tool-message">
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
            <span>
              <i className="bi bi-lightning-charge"></i> {actionSummary}
            </span>
            <span className="tool-names">{actionNames}</span>
            <i
              className={`bi bi-chevron-down${isOpen ? "" : " collapsed"}`}
            >
            </i>
          </button>
          <div className={`collapse${isOpen ? " show" : ""}`} id={collapseId}>
            <div className="tool-list">{controllerItems}{toolItems}</div>
          </div>
        </div>
      </div>
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
      <img className="drawer-action-avatar" src={teammate.avatar} alt="" />
      <span className="drawer-action-copy">
        <span className="drawer-action-author">
          <strong>{teammate.name}</strong>
          <small>{teammate.role} · {teammate.provenance}</small>
        </span>
        <span className="drawer-action-detail">
          <i className={icon}></i>
          <span>
            <strong>{label}</strong>
            {detail ? <small>{detail}</small> : null}
          </span>
        </span>
      </span>
      {timestampMs && <time>{formatEventTime(timestampMs)}</time>}
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
  event,
}: {
  event: Extract<AgentEvent, { type: "correction" }>;
}) {
  const text = (event.summary || event.message || "Agent framework correction")
    .trim();
  const teammate = teamActorPresentation(event.actor || workflowStewardActor());
  const detail = event.message.trim() === text ? "" : event.message.trim();

  return (
    <article
      className="bot message-row assistant-message compact correction-message"
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
        <div className="bubble thought-bubble correction-bubble">
          <p>{text}</p>
          {detail
            ? (
              <details>
                <summary>Details</summary>
                <p>{detail}</p>
              </details>
            )
            : null}
        </div>
      </div>
    </article>
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
          <span>Session request</span>
          {timestampMs ? <time>{formatEventTime(timestampMs)}</time> : null}
        </div>
        <div className="bubble user-bubble">
          <p>{task}</p>
        </div>
      </div>
      <div className="user-avatar">
        <img src="/api/current-user.png" alt="Current user" />
      </div>
    </article>
  );
}

type AssistantMessageRowProps = {
  profile: string;
  timestampMs?: number;
  nestingDepth?: number;
  compact?: boolean;
  children: React.ReactNode;
};

function AssistantMessageRow({
  profile,
  timestampMs,
  nestingDepth = 0,
  compact = false,
  children,
}: AssistantMessageRowProps) {
  return (
    <article
      className={`bot message-row assistant-message assistant-transcript${
        compact ? " compact" : ""
      }`}
      style={{ marginLeft: `${nestingDepth}rem` }}
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
    </article>
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
}: {
  envelope: EventEnvelope;
  events: EventEnvelope[];
}) {
  const event = envelope.event;
  if (event.type !== "team_message") return null;
  const teammate = teamActorPresentation(event.actor);

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
      className={`bot message-row assistant-message team-message tone-${event.tone}`}
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
          <p>{event.message}</p>
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
    </article>
  );
}

export function MessageBubble({
  envelope,
  activityProfile,
  evidenceEvents = [],
}: {
  envelope: EventEnvelope;
  activityProfile?: string;
  evidenceEvents?: EventEnvelope[];
}) {
  const e = envelope.event;

  switch (e.type) {
    case "started":
      return (
        <article className="user message-row user-message">
          <div className="message-container">
            <div className="author-line">
              <strong>You</strong>
              <span>Session request</span>
              <time>{formatEventTime(e.timestamp_ms)}</time>
            </div>
            <div className="bubble user-bubble">
              <p>{e.task}</p>
            </div>
          </div>
          <div className="user-avatar">
            <img src="/api/current-user.png" alt="Current user" />
          </div>
        </article>
      );

    case "model_loading":
    case "step_started": {
      const sd = e.nesting_depth || 0;
      const label = e.type === "model_loading"
        ? "Loading model"
        : `Working step ${e.step} of ${e.max_steps}`;
      const profile = activityProfile || "build";
      return (
        <article
          className="message-row assistant-message compact typing-row"
          style={{ marginLeft: `${sd}rem` }}
          aria-label={label}
        >
          <div className="bot-avatar typing-avatar">
            <img src={getAvatarForProfile(profile)} alt="" aria-hidden="true" />
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
          nestingDepth={e.nesting_depth || 0}
        >
          <RichText content={e.content} />
        </AssistantMessageRow>
      );

    case "user_question":
      return (
        <AssistantMessageRow profile={e.profile} timestampMs={e.timestamp_ms}>
          <div className="assistant-question">
            <p>{e.question}</p>
            {e.choices?.length
              ? (
                <div className="question-choices">
                  {e.choices.map((choice) => <span key={choice}>{choice}
                  </span>)}
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
        <article className="session-error" aria-label="Workflow blocked">
          <strong>Delivery needs help</strong>
          <span>{e.reason}</span>
          {e.timestamp_ms
            ? <time>{formatEventTime(e.timestamp_ms)}</time>
            : null}
        </article>
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
      return <CorrectionNotice event={e} />;

    case "team_message":
      return <TeamMessageBubble envelope={envelope} events={evidenceEvents} />;

    case "user_answer":
      return (
        <article className="user message-row user-message compact">
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
        </article>
      );

    case "user_message":
      return (
        <article className="user message-row user-message compact">
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
        </article>
      );

    case "user_message_applied":
      return null;

    case "sub_agent_started":
      const saDepth = e.nesting_depth || 0;
      return (
        <article
          className="message-row assistant-message compact"
          style={{ marginLeft: `${saDepth}rem` }}
        >
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
      const sfDepth = e.nesting_depth || 0;
      return (
        <article
          className="message-row assistant-message compact"
          style={{ marginLeft: `${sfDepth}rem` }}
        >
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
      const dd = e.nesting_depth || 0;
      return (
        <article
          className="message-row assistant-message diff-message"
          style={{ marginLeft: `${dd}rem` }}
        >
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
          nestingDepth={e.nesting_depth || 0}
        >
          <RichText content={e.content} />
        </AssistantMessageRow>
      );

    case "llm_invocation":
      return (
        <article
          className="session-correction"
          aria-label={`Model inference step ${e.step}`}
        >
          <span>
            Model inference {e.step}
            {e.purpose ? ` (${e.purpose.replaceAll("_", " ")})` : ""} ·{" "}
            {formatHumanDurationMs(e.duration_ms)} ·{" "}
            {formatNumber(e.prompt_tokens + e.generated_tokens)} tokens
            {e.prompt_cache
              ? ` · ${formatNumber(e.prompt_cache.cached_tokens)} cached, ${
                formatNumber(e.prompt_cache.prefilled_tokens)
              } prefilled (${e.prompt_cache.source.replaceAll("_", " ")})` +
                (e.prompt_cache.miss_reason
                  ? ` · miss: ${
                    e.prompt_cache.miss_reason.replaceAll("_", " ")
                  }`
                  : "") +
                (e.prompt_cache.root
                  ? ` · root ${
                    formatNumber(e.prompt_cache.root.reused_tokens)
                  }/${formatNumber(e.prompt_cache.root.tokens)} (${
                    e.prompt_cache.root.authority_class.replaceAll("_", " ")
                  })`
                  : "")
              : ""}
            {e.native?.refill
              ? ` · refill lookup ${e.native.refill.cache_lookup_wall_ms}, hydrate ${e.native.refill.state_hydration_wall_ms}, suffix ${e.native.refill.fresh_suffix_prefill_wall_ms}, snapshot ${e.native.refill.snapshot_capture_wall_ms} ms`
              : ""}
            {e.energy_joules !== undefined
              ? ` · ${formatEnergy(e.energy_joules)}`
              : ""}
            {e.average_power_watts !== undefined
              ? ` at ${formatPower(e.average_power_watts)}`
              : ""}
          </span>
          {e.timestamp_ms
            ? <time>{formatEventTime(e.timestamp_ms)}</time>
            : null}
        </article>
      );

    case "executor_started":
    case "task_plan_accepted":
    case "task_plan_rejected":
    case "tasks_changed":
    case "workflow_started":
    case "workflow_resumed":
    case "workflow_stage_started":
    case "workflow_stage_completed":
    case "workflow_artifact_accepted":
    case "workflow_completed":
    case "check_result":
    case "commit_result":
    case "handoff_summary":
    case "final_grace":
      return null;

    case "session_metrics": {
      const totalTokens = e.prompt_tokens + e.generated_tokens;
      const totalEnergyJoules = metricEnergyJoules(e);
      const totalRuntimeMs = metricRuntimeMs(e);
      const coverage = e.energy_coverage === undefined
        ? undefined
        : `${Math.round(e.energy_coverage * 100)}%`;
      const hasMeasurementMetadata = (e.wall_runtime_ms ?? 0) > 0 ||
        e.total_energy_joules !== undefined || e.energy_source !== undefined;
      return (
        <article className="session-correction" aria-label="Session metrics">
          <span>
            {funEnergySummary(totalRuntimeMs, totalTokens, totalEnergyJoules)}
            {totalEnergyJoules !== undefined
              ? (
                <details>
                  <summary>Power-estimate details</summary>
                  <div>
                    Average incremental power:{" "}
                    {formatPower(e.average_power_watts)}
                  </div>
                  {hasMeasurementMetadata
                    ? (
                      <>
                        <div>
                          Gross device energy:{" "}
                          {formatEnergy(e.gross_energy_joules)}
                        </div>
                        <div>
                          After display adjustment:{" "}
                          {formatEnergy(e.adjusted_energy_joules)}
                        </div>
                        <div>Measurement coverage: {coverage ?? "Unknown"}</div>
                        <div>
                          Source:{" "}
                          {e.energy_source?.replaceAll("_", " ") ?? "Unknown"}
                        </div>
                        <div>
                          Exclusions: {[
                            e.display_energy_excluded
                              ? "measured display"
                              : null,
                            e.idle_baseline_applied
                              ? "idle device baseline"
                              : null,
                          ].filter(Boolean).join(", ") || "none available"}
                        </div>
                      </>
                    )
                    : null}
                  <div>
                    Model inference: {formatEnergy(e.llm_energy_joules)}
                  </div>
                  <div>Tools: {formatEnergy(e.tool_energy_joules)}</div>
                  {hasMeasurementMetadata && e.energy_complete === false
                    ? (
                      <div>
                        Estimate is partial or changed source during the task.
                      </div>
                    )
                    : null}
                  {hasMeasurementMetadata && e.energy_exclusive === false
                    ? (
                      <div>
                        Another pb process held the system meter; task
                        attribution is unavailable.
                      </div>
                    )
                    : null}
                </details>
              )
              : null}
            {totalEnergyJoules === undefined && e.energy_exclusive === false
              ? " Power estimate unavailable: the system meter is unsupported or already in use."
              : null}
          </span>
          {e.timestamp_ms
            ? <time>{formatEventTime(e.timestamp_ms)}</time>
            : null}
        </article>
      );
    }

    case "session_summary":
      const ssd = e.nesting_depth || 0;
      return (
        <article
          className="message-row assistant-message compact"
          style={{ marginLeft: `${ssd}rem` }}
        >
          <div className="bot-avatar">
            <i className="bi bi-stars"></i>
          </div>
          <div className="bubble thought-bubble">
            <p>
              Session complete <code>{e.branch}</code>
            </p>
            {e.summary?.trim()
              ? (
                <>
                  <strong>Summary</strong>
                  <pre className="small result-pre">{e.summary}</pre>
                </>
              )
              : null}
            {e.commits?.trim()
              ? (
                <>
                  <strong>Commits</strong>
                  <pre className="small result-pre">{e.commits}</pre>
                </>
              )
              : null}
            {e.diff_stat?.trim()
              ? (
                <>
                  <strong>Diff stat from main</strong>
                  <pre className="small result-pre">{e.diff_stat}</pre>
                </>
              )
              : null}
            {e.diff?.trim()
              ? (
                <details className="transcript-diff">
                  <summary className="transcript-diff-header">
                    <span>Diff from main</span>
                  </summary>
                  <div className="transcript-diff-body">
                    <DiffView diff={e.diff} />
                  </div>
                </details>
              )
              : null}
          </div>
        </article>
      );

    case "error":
      return <ErrorEventBubble event={e} />;

    default:
      return null;
  }
}

function activityLabel(envelope: EventEnvelope): string | undefined {
  const event = envelope.event;
  switch (event.type) {
    case "started":
      return "Session started";
    case "tool_call":
      return `${teamActorPresentation(event.actor).name} started ${
        TOOL_FRIENDLY_NAMES[event.tool] || event.tool
      }`;
    case "tool_result":
      return "Tool result received";
    case "controller_observation":
      return `${
        teamActorPresentation(event.actor || workflowStewardActor()).name
      } ${
        event.receipt.operation === "read_file" ? "read" : "inspected"
      } ${event.receipt.path}`;
    case "controller_closure":
      return `${
        teamActorPresentation(event.actor || workflowStewardActor()).name
      } closed no-change work`;
    case "controller_mutation":
      return `${
        teamActorPresentation(event.actor || workflowStewardActor()).name
      } deleted ${event.receipt.path}`;
    case "user_question":
      return "Waiting for an answer";
    case "user_answer":
      return "Answer received";
    case "user_message":
      return "Message sent to running agent";
    case "user_message_applied":
      return "Running agent received message";
    case "final":
      return "Response completed";
    case "session_summary":
      return "Session completed";
    case "error":
      return "Error reported";
    default:
      return undefined;
  }
}

export function SessionActivity({ events }: { events: EventEnvelope[] }) {
  const items = events
    .map((envelope) => ({
      label: activityLabel(envelope),
      timestampMs: "timestamp_ms" in envelope.event
        ? envelope.event.timestamp_ms
        : undefined,
    }))
    .filter(
      (item): item is { label: string; timestampMs: number | undefined } =>
        Boolean(item.label),
    )
    .slice(-8);

  if (items.length === 0) {
    return (
      <p className="drawer-empty-copy">
        Activity will appear as the session progresses.
      </p>
    );
  }

  return (
    <ol className="session-activity-list">
      {items.map((item, index) => (
        <li key={`${item.label}-${index}`}>
          <span>{item.label}</span>
          {item.timestampMs
            ? <time>{formatEventTime(item.timestampMs)}</time>
            : null}
        </li>
      ))}
    </ol>
  );
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
