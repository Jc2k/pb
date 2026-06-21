import type React from "react";
import { useEffect, useRef, useState } from "react";
import {
  BrowserRouter,
  Link,
  Routes,
  Route,
  useNavigate,
  useParams,
} from "react-router-dom";
import "./session.css";
import { Aside } from "./Aside";

/* ─── constants ──────────────────────────────────────────────── */

const SCROLL_THRESHOLD = 80;

const TOOL_FRIENDLY_NAMES: Record<string, string> = {
  read_file: "Read file",
  glob: "List files",
  ripgrep: "Search code",
  search: "Search",
  web_search: "Web search",
  web_fetch: "Fetch URL",
  git_log: "Git log",
  todo: "Manage todos",
  skill_search: "Search skills",
  skill: "Load skill",
  ask_user: "Ask user",
  run_command: "Run command",
  edit_file: "Edit file",
  apply_patch: "Apply patch",
  mv: "Move/rename",
  rm: "Remove",
  git_commit: "Commit changes",
  git_revert: "Revert commit",
  sub_agent: "Sub-agent",
};

const TOOL_ICONS: Record<string, string> = {
  read_file: "bi bi-file-earmark-text",
  glob: "bi bi-files",
  ripgrep: "bi bi-search",
  search: "bi bi-search",
  web_search: "bi bi-globe",
  web_fetch: "bi bi-link",
  git_log: "bi bi-clock-history",
  todo: "bi bi-check2-square",
  skill_search: "bi bi-bookmark-star",
  skill: "bi bi-file-earmark-code",
  ask_user: "bi bi-person-circle",
  run_command: "bi bi-play-circle",
  edit_file: "bi bi-pencil-square",
  apply_patch: "bi bi-diff",
  mv: "bi bi-graph-up-arrow",
  rm: "bi bi-trash",
  git_commit: "bi bi-git",
  git_revert: "bi bi-x-octagon",
  sub_agent: "bi bi-people",
};

/* ─── types ──────────────────────────────────────────────────── */

type AgentEvent =
  | {
      type: "started";
      task: string;
      model: string;
      workspace: string;
      branch: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "step_started";
      step: number;
      max_steps: number;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | { type: "reasoning"; content: string; profile: string; timestamp_ms?: number }
  | {
      type: "tool_call";
      tool: string;
      arguments: unknown;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "tool_result";
      tool: string;
      result: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "user_question";
      question_id: string;
      question: string;
      choices?: string[];
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "user_answer";
      question_id: string;
      answer: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "sub_agent_started";
      profile: string;
      task: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "sub_agent_finished";
      profile: string;
      result: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | { type: "diff"; path: string; diff: string; nesting_depth?: number; timestamp_ms?: number }
  | { type: "final"; content: string; nesting_depth?: number; timestamp_ms?: number }
  | {
      type: "session_summary";
      branch: string;
      commits: string;
      summary?: string;
      diff_stat?: string;
      diff?: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "error";
      message: string;
      summary?: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | { type: string; [key: string]: unknown };

interface EventEnvelope {
  version: string;
  event: AgentEvent;
}

type SessionStatus = "queued" | "running" | "paused" | "completed" | "failed";

interface SessionItem {
  session_id: string;
  task: string;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  branch?: string;
  workdir?: string;
  pending_question?: { question_id: string; question: string; choices?: string[] };
  updated_at_ms: number;
}

interface SessionDetails {
  session_id: string;
  task: string;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  branch?: string;
  workdir?: string;
  pending_question?: { question_id: string; question: string; choices?: string[] };
  events: EventEnvelope[];
  updated_at_ms: number;
}

interface ProjectEntry {
  name: string;
  path: string;
  notify_on_finish: boolean;
}

/* ─── custom router removed - using react-router-dom instead ─ */

/* ─── avatar helpers ─────────────────────────────────────────── */

function getAvatarForProfile(profile: string): string {
  const validProfiles = [
    "build",
    "scout",
    "review",
    "explore",
    "plan",
    "ask",
    "research",
  ];
  if (validProfiles.includes(profile)) {
    return `/avatar-${profile}.png`;
  }
  return "/avatar.png";
}

/* ─── helpers ────────────────────────────────────────────────── */

function groupToolEvents(
  events: EventEnvelope[],
): (
  | EventEnvelope
  | {
      type: "tool_group";
      toolCalls: EventEnvelope[];
      toolResults: EventEnvelope[];
    }
)[] {
  const grouped: (
    | EventEnvelope
    | {
      type: "tool_group";
      toolCalls: EventEnvelope[];
      toolResults: EventEnvelope[];
    }
  )[] = [];

  let currentToolCalls: EventEnvelope[] = [];
  let currentToolResults: EventEnvelope[] = [];

  for (let i = 0; i < events.length; i++) {
    const event = events[i];

    if (event.event.type === "tool_call") {
      currentToolCalls.push(event);
    } else if (
      event.event.type === "tool_result" &&
      currentToolCalls.length > currentToolResults.length
    ) {
      currentToolResults.push(event);
    } else {
      if (currentToolCalls.length > 0 || currentToolResults.length > 0) {
        grouped.push({
          type: "tool_group",
          toolCalls: [...currentToolCalls],
          toolResults: [...currentToolResults],
        });
        currentToolCalls = [];
        currentToolResults = [];
      }
      grouped.push(event);
    }
  }

  if (currentToolCalls.length > 0 || currentToolResults.length > 0) {
    grouped.push({
      type: "tool_group",
      toolCalls: [...currentToolCalls],
      toolResults: [...currentToolResults],
    });
  }

  return grouped;
}


function notificationSupport(): boolean {
  return typeof window !== "undefined" && "Notification" in window;
}

async function ensureNotificationPermission(): Promise<boolean> {
  if (!notificationSupport()) return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission !== "default") return false;
  return (await Notification.requestPermission()) === "granted";
}

async function notifySessionFinished(session: SessionItem, projects: ProjectEntry[]) {
  if (session.status !== "completed" && session.status !== "failed") return;
  const project = projects.find((entry) => entry.path === session.workdir);
  if (!project?.notify_on_finish) return;
  if (!(await ensureNotificationPermission())) return;
  const title = session.status === "completed" ? "pb session completed" : "pb session failed";
  const body = `${project.name}: ${session.task}`;
  const url = `/sessions/${session.session_id}`;
  const registration = await navigator.serviceWorker?.getRegistration?.();
  if (registration?.showNotification) {
    await registration.showNotification(title, {
      body,
      icon: "/apple-touch-icon.png",
      badge: "/apple-touch-icon.png",
      data: { url },
      tag: `pb-${session.session_id}-${session.status}`,
    });
    return;
  }
  const notification = new Notification(title, { body, icon: "/apple-touch-icon.png" });
  notification.onclick = () => {
    window.focus();
    window.location.href = url;
  };
}

function useProjectFinishNotifications(sessions: SessionItem[], projects: ProjectEntry[]) {
  const seenRef = useRef<Record<string, SessionStatus>>({});

  useEffect(() => {
    for (const session of sessions) {
      const previous = seenRef.current[session.session_id];
      seenRef.current[session.session_id] = session.status;
      if (
        previous &&
        previous !== session.status &&
        (session.status === "completed" || session.status === "failed")
      ) {
        void notifySessionFinished(session, projects);
      }
    }
  }, [sessions, projects]);
}

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

function formatEventTime(timestamp_ms?: number): string {
  if (!timestamp_ms) return "";
  const date = new Date(timestamp_ms);
  const now = new Date();
  const diffMs = now.getTime() - timestamp_ms;
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const hours = Math.floor(diffMin / 60);
  const minutes = diffMin % 60;
  if (hours < 24) {
    const timeStr = date.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
    return `at ${timeStr}`;
  }
  return date.toLocaleDateString();
}

function formatStartTime(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - timestamp;
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return "Started just now";
  if (diffMin < 60) return `Started ${diffMin} min ago`;
  const hours = Math.floor(diffMin / 60);
  const minutes = diffMin % 60;
  if (hours < 24) {
    const timeStr = date.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
    return `Started ${timeStr}`;
  }
  return date.toLocaleDateString();
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

/* ─── tool detail helper ─────────────────────────────────────── */

function getToolDetail(
  toolCall: EventEnvelope,
  toolResult?: EventEnvelope,
): string | null {
  if (toolCall.event.type !== "tool_call") return null;

  const args = toolCall.event.arguments as Record<string, unknown>;

  switch (toolCall.event.tool) {
    case "read_file":
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
    case "glob":
      return (
        (args.pattern as string) ||
        "(no pattern)" + (args.relative_path ? ` in ${args.relative_path}` : "")
      );
    case "ripgrep":
    case "search":
      return (
        (args.pattern as string) ||
        "(no pattern)" + (args.path ? ` in ${args.path}` : "")
      );
    case "web_search":
      return (args.query as string) || "(no query)";
    case "web_fetch":
      return (args.url as string) || "(no url)";
    case "run_command":
      return (args.cmd as string) || "(no cmd)";
    case "skill_search": {
      const query = args.query as string;
      if (!query) return "";
      if (!toolResult) return `${query} (pending)`;
      const skillMatches =
        toolResult.event.type === "tool_result"
          ? toolResult.event.result.match(/name: /g)?.length || 0
          : 0;
      return `${query} (${skillMatches} skills)`;
    }
    case "skill": {
      const name = args.name as string;
      if (!name) return "(no name)";
      if (name === "list") return "loaded skills list";
      return name;
    }
    case "mv":
      return `from ${(args.source as string) || ""} to ${(args.destination as string) || ""}`;
    case "rm":
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
    case "edit_file": {
      const path = args.path as string;
      if (!path) return "(no path)";
      return path + (args.diff ? " (patch)" : "");
    }
    case "apply_patch":
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
    case "git_commit":
      return (args.message as string) || "(no message)";
    case "git_revert":
      return (args.commit as string) || "(no commit)";
    default:
      if (toolResult && toolResult.event.type === "tool_result") {
        try {
          const parsed = JSON.parse(toolResult.event.result);
          if (Array.isArray(parsed)) {
            return `${parsed.length} items`;
          }
          if (typeof parsed === "object" && parsed !== null) {
            const keys = Object.keys(parsed);
            return `result (${keys.length} fields)`;
          }
        } catch {
          const result = toolResult.event.result;
          if (result.length < 80) {
            return result.replace(/\n/g, " ");
          }
        }
      }
      if (!toolResult) return "(pending)";
      return null;
  }
}

/* ─── message bubble component for grouping ────────────────── */

function ToolGroupBubble({
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

interface ToolSummaryItem {
  detail: string;
  timestampMs?: number;
}

interface ToolSummary {
  toolName: string;
  friendlyName: string;
  icon: string;
  count: number;
  items: ToolSummaryItem[];
}

type TodoStatus = "pending" | "in_progress" | "completed" | "blocked";

interface TodoTask {
  id: number;
  title: string;
  description: string;
  status: TodoStatus;
  parent_id?: number | null;
  notes?: string[];
  timestampMs?: number;
}

function isTodoTask(value: unknown): value is TodoTask {
  if (!value || typeof value !== "object") return false;
  const task = value as Record<string, unknown>;
  return typeof task.id === "number" && typeof task.title === "string";
}

function parseTodoTasks(result: string): TodoTask[] | null {
  if (result === "no todos" || result === "no pending todos") return [];
  try {
    const parsed = JSON.parse(result) as unknown;
    if (Array.isArray(parsed) && parsed.every(isTodoTask)) return parsed;
    if (parsed && typeof parsed === "object") {
      const payload = parsed as Record<string, unknown>;
      if (isTodoTask(payload.added)) return [payload.added];
      if (isTodoTask(payload.updated)) return [payload.updated];
    }
  } catch {
    return null;
  }
  return null;
}

function buildTodoTasks(events: EventEnvelope[]): TodoTask[] {
  const tasks = new Map<number, TodoTask>();
  const pendingCalls: EventEnvelope[] = [];

  events.forEach((event) => {
    if (event.event.type === "tool_call") {
      pendingCalls.push(event);
      return;
    }

    if (event.event.type !== "tool_result") return;

    const callIndex = pendingCalls.findIndex(
      (call) => call.event.type === "tool_call" && call.event.tool === event.event.tool,
    );
    const call = callIndex >= 0 ? pendingCalls.splice(callIndex, 1)[0] : undefined;

    if (event.event.tool !== "todo") return;

    const parsedTasks = parseTodoTasks(event.event.result);
    if (!parsedTasks) return;

    const action =
      call?.event.type === "tool_call"
        ? ((call.event.arguments as Record<string, unknown> | undefined)?.action as string | undefined)
        : undefined;

    if (action === "list") tasks.clear();

    parsedTasks.forEach((task) => {
      tasks.set(task.id, { ...tasks.get(task.id), ...task, timestampMs: event.event.timestamp_ms });
    });
  });

  return Array.from(tasks.values()).sort((a, b) => a.id - b.id);
}

const TODO_STATUS_LABELS: Record<TodoStatus, string> = {
  pending: "Pending",
  in_progress: "In progress",
  completed: "Completed",
  blocked: "Blocked",
};

function buildToolSummaries(events: EventEnvelope[]): ToolSummary[] {
  const summaries: Record<string, ToolSummary> = {};
  const pendingCalls: EventEnvelope[] = [];

  events.forEach((event) => {
    if (event.event.type === "tool_call") {
      pendingCalls.push(event);
      return;
    }

    if (event.event.type === "tool_result" && pendingCalls.length > 0) {
      const call = pendingCalls.shift();
      if (!call || call.event.type !== "tool_call") return;
      addToolSummaryItem(summaries, call, event);
    }
  });

  pendingCalls.forEach((call) => addToolSummaryItem(summaries, call));

  return Object.values(summaries);
}

function addToolSummaryItem(
  summaries: Record<string, ToolSummary>,
  call: EventEnvelope,
  result?: EventEnvelope,
) {
  if (call.event.type !== "tool_call") return;

  const toolName = call.event.tool;
  if (!summaries[toolName]) {
    summaries[toolName] = {
      toolName,
      friendlyName: TOOL_FRIENDLY_NAMES[toolName] || toolName,
      icon: TOOL_ICONS[toolName] || "bi bi-file-earmark-text",
      count: 0,
      items: [],
    };
  }

  summaries[toolName].count++;
  summaries[toolName].items.push({
    detail: getToolDetail(call, result) || "(no details)",
    timestampMs: call.event.timestamp_ms,
  });
}

function DrawerPanel({
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

function TodoDrawer({ tasks }: { tasks: TodoTask[] }) {
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

function ToolDrawerSummary({ summary }: { summary: ToolSummary }) {
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

function isHiddenChatEvent(event: EventEnvelope): boolean {
  return (
    event.event.type === "sub_agent_started" ||
    event.event.type === "sub_agent_finished"
  );
}

function chatEventsWithOnlyLatestStep(events: EventEnvelope[]): EventEnvelope[] {
  const chatEvents = events.filter((event) => !isHiddenChatEvent(event));
  const lastVisibleIndex = chatEvents.length - 1;

  return chatEvents.filter(
    (event, index) =>
      event.event.type !== "step_started" || index === lastVisibleIndex,
  );
}

function profileName(profile: string): string {
  switch (profile) {
    case "plan":
      return "Dade Murphy";
    case "build":
      return "Kate Libby";
    case "review":
      return "Eugene Belford";
    case "scout":
      return "Ramon Sanchez";
    case "explore":
      return "Paul Cook";
    case "research":
      return "Emmanuel Goldstein";
    case "monitor":
      return "Trinity Walker";
    case "ask":
      return "Joey Pardella";
    default:
      return "Jon Appleseed";
  }
}

function profileJobTitle(profile: string): string {
  switch (profile) {
    case "plan":
      return "Ticket Goblin";
    case "build":
      return "Patch Crafter";
    case "review":
      return "Review Gremlin";
    case "scout":
      return "Env Scout";
    case "explore":
      return "Repo Mapper";
    case "research":
      return "Web Sleuth";
    case "monitor":
      return "Progress Monitor";
    case "ask":
      return "Question Wrangler";
    default:
      return "Unknown";
  }
}

function errorSummary(event: Extract<AgentEvent, { type: "error" }>): string {
  const summary = event.summary?.trim();
  if (summary) return summary;

  const message = String(event.message || "").trim();
  const firstLine = message.split("\n").find((line) => line.trim())?.trim();
  if (!firstLine) return "Agent error";
  return firstLine.length > 120 ? `${firstLine.slice(0, 117)}…` : firstLine;
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


function InitialUserMessage({ task, timestampMs }: { task: string; timestampMs?: number }) {
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

function MessageBubble({ envelope }: { envelope: EventEnvelope }) {
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
          <div class="message-container">
            <div class="author-line">
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

/* ─── home page (/) ──────────────────────────────────────────── */

export function Asidenav(to: string) {
  const navigate = useNavigate();
  return () => navigate(to);
}

function HomePage() {
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
                    let statusText = s.status;
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
                          <strong>{s.task}</strong>
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

/* ─── projects pages ─────────────────────────────────────────── */

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="app-shell">
      <Aside />
      <section className="main-panel">
        <header className="mobile-topbar d-lg-none d-flex align-items-center justify-content-between px-3 py-2">
          <div className="brand compact d-flex align-items-center gap-2">
            <div className="brand-mark">&gt;_</div>
            <strong>LocalAgent</strong>
          </div>
        </header>
        <div className="content-wrap">{children}</div>
      </section>
    </div>
  );
}

function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);

  const fetchProjects = () =>
    fetch("/api/projects")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: ProjectEntry[]) => setProjects(entries));

  useEffect(() => {
    void fetchProjects();
    void fetch("/api/sessions")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: SessionItem[]) => setSessions(entries));
  }, []);

  useProjectFinishNotifications(sessions, projects);

  const toggleProjectNotifications = async (project: ProjectEntry) => {
    if (!project.notify_on_finish && !(await ensureNotificationPermission())) return;
    const res = await fetch(`/api/projects/${encodeURIComponent(project.name)}/notifications`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notify_on_finish: !project.notify_on_finish }),
    });
    if (res.ok) void fetchProjects();
  };

  return (
    <PageShell>
      <section className="hero-section">
        <h1>Projects</h1>
        <p className="text-secondary mb-3">
          Choose a registered project to view its sessions and start focused project work.
        </p>
      </section>

      <section className="sessions-section">
        <div className="session-list list-group">
          {projects.length === 0 ? (
            <div className="list-group-item text-secondary small">
              No registered projects. Add one with <code>pb projects add</code>.
            </div>
          ) : (
            projects.map((project) => {
              const projectSessions = sessions.filter((session) => session.workdir === project.path);
              const running = projectSessions.filter((session) => session.status === "running").length;
              return (
                <div
                  key={project.name}
                  className="session-row list-group-item py-3 px-4"
                >
                  <div className="session-icon"><i className="bi bi-folder2-open"></i></div>
                  <Link
                    className="session-main text-decoration-none text-reset"
                    to={`/projects/${encodeURIComponent(project.name)}`}
                  >
                    <strong>{project.name}</strong>
                    <span>{project.path}</span>
                  </Link>
                  <span className={`status-pill ${running ? "status-running" : "status-completed"}`}>
                    {projectSessions.length} session{projectSessions.length === 1 ? "" : "s"}
                  </span>
                  <button
                    type="button"
                    className={`btn btn-sm btn-icon ${project.notify_on_finish ? "btn-primary" : "btn-outline-secondary"}`}
                    title={project.notify_on_finish ? "Disable finish notifications" : "Notify me when sessions complete or fail"}
                    aria-label={project.notify_on_finish ? "Disable finish notifications" : "Enable finish notifications"}
                    onClick={(event) => {
                      event.preventDefault();
                      void toggleProjectNotifications(project);
                    }}
                  >
                    <i className={`bi ${project.notify_on_finish ? "bi-alarm-fill" : "bi-alarm"}`}></i>
                  </button>
                  <Link className="chevron text-decoration-none" to={`/projects/${encodeURIComponent(project.name)}`}>›</Link>
                </div>
              );
            })
          )}
        </div>
      </section>
    </PageShell>
  );
}

function ProjectPage() {
  const { projectName: encodedProjectName } = useParams<{ projectName: string }>();
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [task, setTask] = useState("");
  const [branch, setBranch] = useState("main");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const name = encodedProjectName ? decodeURIComponent(encodedProjectName) : "";
  const project = projects.find((entry) => entry.name === name);
  const projectSessions = project ? sessions.filter((session) => session.workdir === project.path) : [];

  useProjectFinishNotifications(sessions, projects);

  useEffect(() => {
    void fetch("/api/projects")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: ProjectEntry[]) => setProjects(entries));
  }, []);

  useEffect(() => {
    const fetchSessions = () =>
      fetch("/api/sessions")
        .then((res) => (res.ok ? res.json() : []))
        .then((entries: SessionItem[]) => setSessions(entries));
    void fetchSessions();
    const timer = window.setInterval(() => void fetchSessions(), 5000);
    return () => window.clearInterval(timer);
  }, []);

  const startProjectSession = async () => {
    if (!project || !task.trim()) return;
    setIsSubmitting(true);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: task.trim(), workdir: project.path, branch: branch.trim() || "main" }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { session_id: string };
      navigate(`/sessions/${data.session_id}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <PageShell>
      <section className="hero-section">
        <Link to="/projects" className="text-decoration-none small fw-medium text-blue">← All projects</Link>
        <h1>{project?.name || name || "Project"}</h1>
        <p className="text-secondary mb-3">{project?.path || "Project not found"}</p>

        {project && (
          <form className="start-card card" onSubmit={(e) => { e.preventDefault(); void startProjectSession(); }}>
            <div className="task-editor position-relative">
              <textarea
                className="form-control"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder={`Ask the agent to work in ${project.name}…`}
                rows={4}
              />
            </div>
            <div className="session-controls row g-3 align-items-end p-3">
              <div className="col-12 col-md-8">
                <label className="form-label small fw-semibold">Base branch</label>
                <select className="form-select" value={branch} onChange={(e) => setBranch(e.target.value)}>
                  <option>main</option>
                  <option>develop</option>
                  <option>feature/ui-refresh</option>
                </select>
              </div>
              <div className="col-12 col-md-4 d-grid">
                <button className="btn btn-primary start-button" type="submit" disabled={!task.trim() || isSubmitting}>▷ Start project chat</button>
              </div>
            </div>
          </form>
        )}
      </section>

      <section className="sessions-section">
        <div className="section-header d-flex align-items-center justify-content-between mb-3">
          <h2 className="h6 fw-bold m-0">Project sessions</h2>
        </div>
        <div className="session-list list-group">
          {projectSessions.length === 0 ? (
            <div className="list-group-item text-secondary small">No sessions for this project yet</div>
          ) : (
            projectSessions.map((session) => (
              <SessionCard key={session.session_id} session={session} onClick={() => navigate(`/sessions/${session.session_id}`)} />
            ))
          )}
        </div>
      </section>
    </PageShell>
  );
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
          {session.task}
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

/* ─── session page (/sessions/:id) ──────────────────────────── */

function SessionPage() {
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
      title: `pb session: ${session.task}`,
      text: `View this pb session: ${session.task}`,
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
            <h1>{session.task}</h1>
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

/* ─── main app ───────────────────────────────────────────────── */

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/sessions/:sessionId" element={<SessionPage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:projectName" element={<ProjectPage />} />
      </Routes>
    </BrowserRouter>
  );
}
