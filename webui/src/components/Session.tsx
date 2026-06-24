import type React from "react";
import { useState } from "react";
import type { AgentEvent, EventEnvelope, SessionItem } from "../types";
import { TOOL_FRIENDLY_NAMES, TOOL_ICONS } from "../lib/constants";
import { formatEventTime, getAvatarForProfile, projectName, relativeTime, sessionTitle } from "../lib/helpers";
import { TODO_STATUS_LABELS, errorSummary, getToolDetail, profileJobTitle, profileName } from "../lib/sessionUtils";
import type { TodoTask, ToolSummary } from "../lib/sessionUtils";

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

export function ToolGroupBubble({
  toolCalls,
  toolResults,
}: {
  toolCalls: EventEnvelope[];
  toolResults: EventEnvelope[];
}) {
  const [isOpen, setIsOpen] = useState(false);

  if (toolCalls.length === 0) return null;

  const collapseId = `tools-${Math.random().toString(36).substr(2, 9)}`;

  const toolItems = toolCalls
    .map((e, i) => {
      if (e.event.type !== "tool_call") return null;

      const toolName = e.event.tool;
      const friendlyName = TOOL_FRIENDLY_NAMES[toolName] || toolName;
      const iconClass = TOOL_ICONS[toolName] || "bi bi-file-earmark-text";

      let statusClass = "success";
      let detailText: string | null = null;

      const result = i < toolResults.length ? toolResults[i] : undefined;
      detailText = getToolDetail(e, result);

      return (
        <div key={i} className={`tool-item ${statusClass}`}>
          <i className={iconClass}></i>
          <span>{friendlyName}</span>
          {detailText && <small>{detailText}</small>}
        </div>
      );
    })
    .filter(Boolean);

  const toolNames = toolCalls
    .map((e, i) => {
      if (e.event.type === "tool_call")
        return TOOL_FRIENDLY_NAMES[e.event.tool] || e.event.tool;
      return "";
    })
    .filter(Boolean)
    .join(" · ");

  return (
    <article className="message-row compact tool-message">
      <div className="bubble thought-bubble">
        <button
          className={`tool-strip${isOpen ? "" : " collapsed"}`}
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          type="button"
        >
          <span>
            <i className="bi bi-tools"></i> {toolCalls.length} tools used
          </span>
          <span className="tool-names">{toolNames}</span>
          <i
            className={`bi bi-chevron-down${isOpen ? "" : " collapsed"}`}
          ></i>
        </button>
        <div className={`collapse${isOpen ? " show" : ""}`} id={collapseId}>
          <div className="tool-list">{toolItems}</div>
        </div>
      </div>
    </article>
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
        <p>Todo tool activity will appear here as the agent plans and updates work.</p>
      </div>
    );
  }

  return (
    <ol className="todo-list">
      {tasks.map((task) => (
        <li key={task.id} className={`todo-item ${task.status}`}>
          <div className="todo-title-row">
            <span className="todo-id">#{task.id}</span>
            <span className="todo-status">{TODO_STATUS_LABELS[task.status] || task.status}</span>
          </div>
          <strong>{task.title}</strong>
          {task.description && <p>{task.description}</p>}
          {task.parent_id ? <small>Parent #{task.parent_id}</small> : null}
          {task.notes?.length ? (
            <ul className="todo-notes">
              {task.notes.map((note, index) => (
                <li key={index}>{note}</li>
              ))}
            </ul>
          ) : null}
          {task.timestampMs && <time>{formatEventTime(task.timestampMs)}</time>}
        </li>
      ))}
    </ol>
  );
}

export function ToolDrawerSummary({ summary }: { summary: ToolSummary }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="drawer-tool-group">
      <button
        className="drawer-item"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        type="button"
      >
        <span>
          <i className={summary.icon}></i>
          {summary.friendlyName}
        </span>
        <span className="drawer-count">
          <strong>{summary.count}</strong>
          <i
            className={`bi bi-chevron-down${isOpen ? "" : " collapsed"}`}
          ></i>
        </span>
      </button>
      {isOpen && (
        <ol className="drawer-tool-details">
          {summary.items.map((item, index) => (
            <li key={`${summary.toolName}-${index}`}>
              <span className="drawer-detail-text">{item.detail}</span>
              {item.timestampMs && (
                <time>{formatEventTime(item.timestampMs)}</time>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function ErrorEventBubble({
  event,
}: {
  event: Extract<AgentEvent, { type: "error" }>;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const summary = errorSummary(event);
  const detail = String(event.message || "").trim() || "No error details provided.";
  const hasDetail = detail !== summary;

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
          ></i>
        </button>
        <div className={`collapse${isOpen ? " show" : ""}`}>
          <div className="error-detail">
            {hasDetail ? <strong>{summary}</strong> : null}
            <pre className="mb-0 small result-pre">{detail}</pre>
          </div>
        </div>
      </div>
    </article>
  );
}


export function InitialUserMessage({ task, timestampMs }: { task: string; timestampMs?: number }) {
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

export function MessageBubble({ envelope }: { envelope: EventEnvelope }) {
  const e = envelope.event;

  switch (e.type) {
    case "started":
      return (
        <article className="user message-row user-message">
          <div className="message-container">
            <div className="author-line">
              <strong>{profileName(e.profile)}</strong>
              <span>{profileJobTitle(e.profile)}</span>
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

    case "step_started":
      const sd = e.nesting_depth || 0;
      return (
        <article
          className="message-row assistant-message compact typing-row"
          style={{ marginLeft: `${sd}rem` }}
          aria-label={`Working step ${e.step} of ${e.max_steps}`}
        >
          <div className="typing-indicator" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </article>
      );

    case "reasoning":
      const rd = e.nesting_depth || 0;
      return (
        <article
          className="bot message-row assistant-message"
          style={{ paddingLeft: `${rd}rem` }}
        >
          <div className="bot-avatar">
            <img src={`/static/images/avatar-${e.profile}.png`} />
          </div>
          <div className="message-container">
            <div className="author-line">
              <strong>{profileName(e.profile)}</strong>
              <span>{profileJobTitle(e.profile)}</span>
              <time>{formatEventTime(e.timestamp_ms)}</time>
            </div>
            <div className="bubble thought-bubble">
              <p>{e.content}</p>
            </div>
          </div>
        </article>
      );

    case "user_question":
      return (
        <article className="user message-row user-message">
          <div className="message-container">
            <div className="author-line">
              <strong>{profileName(e.profile)}</strong>
              <span>{profileJobTitle(e.profile)}</span>
              <time>{formatEventTime(e.timestamp_ms)}</time>
            </div>
            <div className="bubble user-bubble">
              <p>{e.question}</p>
              {e.choices?.length ? (
                <div className="d-flex gap-2 flex-wrap mt-2">
                  {e.choices.map((choice) => (
                    <span className="badge text-bg-warning" key={choice}>{choice}</span>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
          <div className="user-avatar">
            <img src="/api/current-user.png" alt="Current user" />
          </div>
        </article>
      );

    case "user_answer":
      return (
        <article className="bot message-row assistant-message compact">
          <div className="bot-avatar">
            <i className="bi bi-stars"></i>
          </div>
          <div className="bubble thought-bubble">
            <p>Your answer:</p>
            <pre className="mb-0 small">{e.answer}</pre>
          </div>
        </article>
      );

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
          className="message-row assistant-message"
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
            <details open className="card border-info mb-0">
              <summary className="card-header py-1 small d-flex align-items-center gap-2">
                <span className="badge bg-info text-dark">diff view</span>
              </summary>
              <div className="card-body p-0 overflow-auto">
                <DiffView diff={e.diff} />
              </div>
            </details>
          </details>
        </article>
      );

    case "final":
      const ffd = e.nesting_depth || 0;
      return (
        <article
          className="bot message-row assistant-message"
          style={{ marginLeft: `${ffd}rem` }}
        >
          <div className="bot-avatar">
            <i className="bi bi-stars"></i>
          </div>
          <div className="message-container">
            <div className="author-line">
              <strong>{profileName(e.profile)}</strong>
              <span>{profileJobTitle(e.profile)}</span>
              <time>{formatEventTime(e.timestamp_ms)}</time>
            </div>
            <div className="bubble thought-bubble">
              <p>{e.content}</p>
            </div>
          </div>
        </article>
      );

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
            {e.summary?.trim() ? (
              <>
                <strong>Summary</strong>
                <pre className="small result-pre">{e.summary}</pre>
              </>
            ) : null}
            {e.commits?.trim() ? (
              <>
                <strong>Commits</strong>
                <pre className="small result-pre">{e.commits}</pre>
              </>
            ) : null}
            {e.diff_stat?.trim() ? (
              <>
                <strong>Diff stat from main</strong>
                <pre className="small result-pre">{e.diff_stat}</pre>
              </>
            ) : null}
            {e.diff?.trim() ? (
              <details className="card border-info mb-0">
                <summary className="card-header py-1 small d-flex align-items-center gap-2">
                  <span className="badge bg-info text-dark">diff from main</span>
                </summary>
                <div className="card-body p-0 overflow-auto">
                  <DiffView diff={e.diff} />
                </div>
              </details>
            ) : null}
          </div>
        </article>
      );

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
  if (session.status === "running") {
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
    badge = session.pending_question ? (
      <span className="badge bg-warning text-dark">Needs answer</span>
    ) : (
      <span className="badge bg-warning text-dark">Paused after restart</span>
    );
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
