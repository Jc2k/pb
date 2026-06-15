import type React from "react";
import { useEffect, useRef, useState } from "react";
import "./session.css";

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
  sub_agent: "Sub-agent"
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
  sub_agent: "bi bi-people"
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
    }
  | { type: "step_started"; step: number; max_steps: number; nesting_depth?: number }
  | { type: "reasoning"; content: string; nesting_depth?: number }
  | { type: "tool_call"; tool: string; arguments: unknown; nesting_depth?: number }
  | { type: "tool_result"; tool: string; result: string; nesting_depth?: number }
  | { type: "user_question"; question_id: string; question: string; nesting_depth?: number }
  | { type: "user_answer"; question_id: string; answer: string; nesting_depth?: number }
  | { type: "sub_agent_started"; profile: string; task: string; nesting_depth?: number }
  | { type: "sub_agent_finished"; profile: string; result: string; nesting_depth?: number }
  | { type: "diff"; path: string; diff: string; nesting_depth?: number }
  | { type: "final"; content: string; nesting_depth?: number }
  | { type: "session_summary"; branch: string; commits: string; nesting_depth?: number }
  | { type: "error"; message: string; nesting_depth?: number }
  | { type: string; [key: string]: unknown };

interface EventEnvelope {
  version: string;
  event: AgentEvent;
}

type SessionStatus = "queued" | "running" | "paused" | "completed";

interface SessionItem {
  session_id: string;
  task: string;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  branch?: string;
  workdir?: string;
  pending_question?: { question_id: string; question: string };
  updated_at_ms: number;
}

interface SessionDetails {
  session_id: string;
  task: string;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  branch?: string;
  pending_question?: { question_id: string; question: string };
  events: EventEnvelope[];
}

interface ProjectEntry {
  name: string;
  path: string;
}

/* ─── simple router ──────────────────────────────────────────── */

function useRoute(): string {
  const [path, setPath] = useState(() => window.location.pathname);
  useEffect(() => {
    const handler = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);
  return path;
}

function navigate(to: string) {
  window.history.pushState(null, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

/* ─── helpers ────────────────────────────────────────────────── */

function groupToolEvents(events: EventEnvelope[]): (EventEnvelope | { type: "tool_group"; toolCalls: AgentEvent[]; toolResults: AgentEvent[] })[] {
  const grouped: (EventEnvelope | { type: "tool_group"; toolCalls: AgentEvent[]; toolResults: AgentEvent[] })[] = [];
  
  let currentToolCalls: AgentEvent[] = [];
  let currentToolResults: AgentEvent[] = [];
  
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    
    if (event.event.type === "tool_call") {
      currentToolCalls.push(event);
    } else if (event.event.type === "tool_result" && currentToolCalls.length > currentToolResults.length) {
      currentToolResults.push(event);
    } else {
      if (currentToolCalls.length > 0 || currentToolResults.length > 0) {
        grouped.push({ type: "tool_group", toolCalls: [...currentToolCalls], toolResults: [...currentToolResults] });
        currentToolCalls = [];
        currentToolResults = [];
      }
      grouped.push(event);
    }
  }
  
  if (currentToolCalls.length > 0 || currentToolResults.length > 0) {
    grouped.push({ type: "tool_group", toolCalls: [...currentToolCalls], toolResults: [...currentToolResults] });
  }
  
  return grouped;
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
    const timeStr = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
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

function getToolDetail(toolCall: AgentEvent, toolResult?: AgentEvent): string | null {
  if (toolCall.event.type !== "tool_call") return null;
  
  const args = toolCall.event.arguments as Record<string, unknown>;
  
  switch (toolCall.event.tool) {
    case "read_file":
      const filePath = args ? (args.path as string) : undefined; return filePath || "(no path)";
    case "glob":
      return (args.pattern as string) || "(no pattern)" + (args.relative_path ? ` in ${args.relative_path}` : "");
    case "ripgrep":
    case "search":
      return (args.pattern as string) || "(no pattern)" + (args.path ? ` in ${args.path}` : "");
    case "web_search":
      return (args.query as string) || "(no query)";
    case "web_fetch":
      return (args.url as string) || "(no url)";
    case "run_command":
      return (args.cmd as string) || "(no cmd)";
    case "skill_search": {
      const query = args.query as string;
      if (!query) return "";
      const skillMatches = toolResult?.event.type === "tool_result" ? (toolResult.event.result.match(/name: /g)?.length || 0) : 0;
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
      const filePath = args ? (args.path as string) : undefined; return filePath || "(no path)";
    case "edit_file": {
      const path = args.path as string;
      if (!path) return "(no path)";
      return path + (args.diff ? " (patch)" : "");
    }
    case "apply_patch":
      const filePath = args ? (args.path as string) : undefined; return filePath || "(no path)";
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
      return null;
  }
}

/* ─── message bubble component for grouping ────────────────── */

function ToolGroupBubble({ toolCalls, toolResults }: { toolCalls: AgentEvent[]; toolResults: AgentEvent[] }) {
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
      
      if (i < toolResults.length && toolResults[i].event.type === "tool_result") {
        detailText = getToolDetail(e, toolResults[i]);
      }
      
      return (
        <div key={i} className={`tool-item ${statusClass}`}>
          <i className={iconClass}></i>
          <span>{friendlyName}</span>
          {detailText && <small>{detailText}</small>}
        </div>
      );
    })
    .filter(Boolean);
  
  const toolNames = toolCalls.map((e, i) => {
    if (e.event.type === "tool_call") return TOOL_FRIENDLY_NAMES[e.event.tool] || e.event.tool;
    return "";
  }).filter(Boolean).join(" · ");

  return (
    <article className="message-row assistant-message compact">
      <div className="bot-avatar"><i className="bi bi-stars"></i></div>
      <div className="bubble thought-bubble">
        <button 
          className={`tool-strip${isOpen ? "" : " collapsed"}`}
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          type="button"
        >
          <span><i className="bi bi-tools"></i> {toolCalls.length} tools used</span>
          <span className="tool-names">{toolNames}</span>
          <i className={`bi bi-chevron-down${isOpen ? "" : " collapsed"}`}></i>
        </button>
        <div className={`collapse${isOpen ? " show" : ""}`} id={collapseId}>
          <div className="tool-list">
            {toolItems}
          </div>
        </div>
        <time>{relativeTime(Date.now())}</time>
      </div>
    </article>
  );
}


function MessageBubble({ envelope }: { envelope: EventEnvelope }) {
  const e = envelope.event;
  
  switch (e.type) {
    case "reasoning":
      const rd = e.nesting_depth || 0;
      return (
        <article className="message-row assistant-message" style={{ paddingLeft: `${rd}rem` }}>
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <div className="bubble thought-bubble">
            <p>{e.content}</p>
            <time>{relativeTime(Date.now())}</time>
          </div>
        </article>
      );

    case "user_question":
      return (
        <article className="message-row user-message">
          <div className="bubble user-bubble">
            <p>{e.question}</p>
            <time>{relativeTime(Date.now())}</time>
          </div>
          <div className="user-avatar"><i className="bi bi-person"></i></div>
        </article>
      );

    case "user_answer":
      return (
        <article className="message-row assistant-message compact">
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <div className="bubble thought-bubble">
            <p>Your answer:</p>
            <pre className="mb-0 small">{e.answer}</pre>
            <time>{relativeTime(Date.now())}</time>
          </div>
        </article>
      );

    case "sub_agent_started":
      const saDepth = e.nesting_depth || 0;
      return (
        <article className="message-row assistant-message compact" style={{ marginLeft: `${saDepth}rem` }}>
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <div className="bubble thought-bubble">
            <p>
              <span className="badge bg-primary me-2">{e.profile}</span>
              {e.task}
            </p>
            <time>{relativeTime(Date.now())}</time>
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
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <details open className="bubble thought-bubble">
            <summary style={{ display: "none" }} />
            <p>Sub-agent {e.profile} completed</p>
            <pre className="mb-0 small result-pre">{e.result}</pre>
            <time>{relativeTime(Date.now())}</time>
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
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <details open className="bubble thought-bubble">
            <summary style={{ display: "none" }} />
            <p><code>{e.path}</code> changed</p>
            <details open className="card border-info mb-0">
              <summary className="card-header py-1 small d-flex align-items-center gap-2">
                <span className="badge bg-info text-dark">diff view</span>
              </summary>
              <div className="card-body p-0 overflow-auto">
                <DiffView diff={e.diff} />
              </div>
            </details>
            <time>{relativeTime(Date.now())}</time>
          </details>
        </article>
      );

    case "final":
      const ffd = e.nesting_depth || 0;
      return (
        <article className="message-row assistant-message" style={{ marginLeft: `${ffd}rem` }}>
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <div className="bubble thought-bubble">
            <p>{e.content}</p>
            <time>{relativeTime(Date.now())}</time>
          </div>
        </article>
      );

    case "session_summary":
      const ssd = e.nesting_depth || 0;
      return (
        <article className="message-row assistant-message compact" style={{ marginLeft: `${ssd}rem` }}>
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <div className="bubble thought-bubble">
            <p>
              Session complete <code>{e.branch}</code>
            </p>
            <pre className="mb-0 small result-pre">{e.commits}</pre>
            <time>{relativeTime(Date.now())}</time>
          </div>
        </article>
      );

    case "error":
      return (
        <article className="message-row assistant-message compact">
          <div className="bot-avatar"><i className="bi bi-stars"></i></div>
          <div className="bubble thought-bubble" style={{ border: "1px solid #fda4af", background: "#fef2f2" }}>
            <p>Error</p>
            <pre className="mb-0 small result-pre">{String(e.message)}</pre>
            <time>{relativeTime(Date.now())}</time>
          </div>
        </article>
      );

    default:
      return null;
  }
}

/* ─── home page (/) ──────────────────────────────────────────── */

function HomePage() {
  const [task, setTask] = useState("");
  const [workdir, setWorkdir] = useState("");
  const [branch, setBranch] = useState("main");
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

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
    setWorkdir((current) => current || entries[0]?.path || "");
  };

  const startSession = async () => {
    if (!task.trim() || !workdir) return;
    setIsSubmitting(true);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          workdir: workdir.trim() || undefined,
          branch: branch.trim() || "main",
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
        <aside className="sidebar d-none d-lg-flex flex-column">
          <div className="brand d-flex align-items-center gap-2 px-3 py-3">
            <div className="brand-mark">&gt;_</div>
            <strong>LocalAgent</strong>
          </div>

          <nav className="nav nav-pills flex-column gap-1 px-2">
            <a className="nav-link active" href="#"><i className="bi bi-house-door"></i> Home</a>
            <a className="nav-link" href="#"><i className="bi bi-chat-square-text"></i> Sessions</a>
            <a className="nav-link" href="#"><i className="bi bi-folder2-open"></i> Projects</a>
            <a className="nav-link" href="#"><i className="bi bi-gear"></i> Settings</a>
          </nav>

          <div className="mt-auto user-menu p-3 d-flex align-items-center gap-2">
            <div className="avatar-sm">JD</div>
            <div>
              <strong>Jane Doe</strong>
              <small className="d-block text-secondary">Local workspace</small>
            </div>
          </div>
        </aside>

        <section className="main-panel">
          <header className="mobile-topbar d-lg-none d-flex align-items-center justify-content-between px-3 py-2">
            <div className="brand compact d-flex align-items-center gap-2">
              <div className="brand-mark">&gt;_</div>
              <strong>LocalAgent</strong>
            </div>
            <button className="btn btn-light btn-icon" aria-label="Open menu">☰</button>
          </header>

          <div className="content-wrap">
            <section className="hero-section">
              <h1>Start a new session</h1>
              <p className="text-secondary mb-3">Describe what you'd like the agent to work on.</p>

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
                    <button type="button" className="btn btn-sm border rounded-2 text-secondary bg-transparent" aria-label="Attach context">⌕</button>
                    <button type="button" className="btn btn-sm border rounded-2 text-secondary bg-transparent" aria-label="Improve prompt">✣</button>
                  </div>
                </div>

                <div className="session-controls row g-3 align-items-end p-3">
                  <div className="col-12 col-md-5">
                    <label className="form-label small fw-semibold">Project</label>
                    <select
                      className="form-select"
                      value={workdir}
                      onChange={(e) => setWorkdir(e.target.value)}
                      disabled={projects.length === 0}
                    >
                      {projects.length === 0 ? (
                        <option value="">No registered projects</option>
                      ) : (
                        projects.map((project) => (
                          <option key={project.name} value={project.path}>
                            {project.name}
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                  <div className="col-12 col-md-4">
                    <label className="form-label small fw-semibold">Base branch</label>
                    <select
                      className="form-select"
                      value={branch}
                      onChange={(e) => setBranch(e.target.value)}
                    >
                      <option>main</option>
                      <option>develop</option>
                      <option>feature/ui-refresh</option>
                    </select>
                  </div>
                  <div className="col-12 col-md-3 d-grid">
                    <button
                      className="btn btn-primary start-button"
                      type="submit"
                      disabled={!task.trim() || !workdir || isSubmitting}
                    >
                      ▷ Start session
                    </button>
                  </div>
                </div>
              </form>
            </section>

            <section className="sessions-section">
              <div className="section-header d-flex align-items-center justify-content-between mb-3">
                <h2 className="h6 fw-bold m-0">Recent sessions</h2>
                <a href="#" className="text-decoration-none small fw-medium text-blue">View all sessions</a>
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
                        <div className={`state-dot rounded-circle bg-${s.status === "running" ? "green" : s.status === "completed" ? "blue" : "gray"}`} />
                        <div className="session-icon">&gt;_</div>
                        <div className="session-main">
                          <strong>{s.task}</strong>
                          <span>{projectName(s.workdir)} · {formatStartTime(s.updated_at_ms)}</span>
                        </div>
                        <span className={`status-pill ${statusClass}`}>{statusText}</span>
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

function SessionPage({ sessionId }: { sessionId: string }) {
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [sessionRunning, setSessionRunning] = useState(false);
  const [followUp, setFollowUp] = useState("");
  const [answer, setAnswer] = useState("");
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

  const answerQuestion = async () => {
    if (!answer.trim() || !session?.pending_question) return;
    await fetch(`/api/sessions/${sessionId}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question_id: session.pending_question.question_id,
        answer: answer.trim(),
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

  const isRunning = session?.status === "running" || false;

  if (!session) {
    return (
      <div className="app-shell">
        <aside className="sidebar d-none d-lg-flex flex-column">
          <div className="brand d-flex align-items-center gap-2 px-3 py-3">
            <div className="brand-mark"><i className="bi bi-terminal"></i></div>
            <div>
              <strong>LocalAgent</strong>
              <small className="d-block text-secondary">Private by default</small>
            </div>
          </div>
        </aside>
        <section className="session-panel">
          <header className="session-header">
            <span className="navbar-brand fw-bold mb-0 d-flex align-items-center gap-2">
              <img src="/logo.svg" alt="pb" width="32" height="32" style={{ borderRadius: "6px" }} />
              pb
            </span>
          </header>
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar d-none d-lg-flex flex-column">
        <div className="brand d-flex align-items-center gap-2 px-3 py-3">
          <div className="brand-mark"><i className="bi bi-terminal"></i></div>
          <strong>LocalAgent</strong>
        </div>

        <nav className="nav nav-pills flex-column gap-1 px-2">
          <a className="nav-link active" href="#"><i className="bi bi-chat-square-text"></i> Sessions</a>
          <a className="nav-link" href="#"><i className="bi bi-folder2-open"></i> Projects</a>
          <a className="nav-link" href="#"><i className="bi bi-files"></i> Files</a>
          <a className="nav-link" href="#"><i className="bi bi-shield-lock"></i> Privacy</a>
          <a className="nav-link" href="#"><i className="bi bi-gear"></i> Settings</a>
        </nav>

        <div className="mt-auto p-3 user-mini">
          <div className="avatar-sm">JC</div>
          <div>
            <strong>John Carr</strong>
            <small className="d-block text-secondary">Local workspace</small>
          </div>
        </div>
      </aside>

      <section className="session-panel">
        <header className="session-header">
          <button
            type="button"
            className="btn btn-link d-lg-none p-0 text-body"
            onClick={() => navigate("/")}
          >
            <i className="bi bi-chevron-left fs-4"></i>
          </button>
          <div className="session-icon d-none d-sm-grid"><i className="bi bi-terminal"></i></div>
          <div className="min-w-0 flex-grow-1">
            <h1>{session.task}</h1>
            <div className="status-line">
              {isRunning && <span className="live-dot" />}
              <span>{session.status === "running" ? "Running" : session.status === "queued" ? "Queued" :
                session.status === "paused" ? (session.pending_question ? "Waiting for answer" : "Paused") : "Completed"}
              </span>
              {session.updated_at_ms && (
                <>
                  <span className="dot-sep"></span>
                  <span>{formatStartTime(session.updated_at_ms)}</span>
                </>
              )}
              <span className="dot-sep d-none d-sm-inline"></span>
              <span className="d-none d-sm-inline">Model: {session.branch}</span>
            </div>
          </div>
          <button className="btn btn-light rounded-pill d-none d-sm-inline-flex">
            <i className="bi bi-box-arrow-up me-2"></i>Share
          </button>
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
               <div className="text-body-secondary small">Waiting for queue events…</div>
             ) : (
               groupToolEvents(events.filter(e => e.event.type !== "sub_agent_started" && e.event.type !== "sub_agent_finished")).map((grouped, i) => {
                 if ((grouped as any).type === "tool_group") {
                   const tc = (grouped as any).toolCalls;
                   return <ToolGroupBubble key={i} toolCalls={tc} toolResults={(grouped as any).toolResults} />;
                 }
                 return <MessageBubble key={i} envelope={grouped as EventEnvelope} />;
               })
             )}
           </main>

          <aside className="tool-drawer d-none d-xl-block">
            <div className="drawer-header">
              <h2>Tools</h2>
              <span className="badge rounded-pill text-bg-light">{events.filter(e => (e.event.type === "tool_call" || e.event.type === "tool_result") && e.event.type !== "sub_agent_started" && e.event.type !== "sub_agent_finished").length}</span>
            </div>
            {events.filter(e => e.event.type === "tool_call" || e.event.type === "tool_result").map((e, i) => (
              <button key={i} className="drawer-item">
                <span><i className="bi bi-file-earmark-text"></i>{e.event.type === "tool_call" ? ` ${e.event.tool}` : ""}</span>
              </button>
            ))}

            <div className="empty-detail">
              <i className="bi bi-file-earmark-code"></i>
              <h3>Select a tool</h3>
              <p>Inspect files, commands, and outputs without cluttering the main session.</p>
            </div>
          </aside>
        </div>

        {session.status === "paused" && session.pending_question ? (
          <form className="composer" onSubmit={(e) => { e.preventDefault(); void answerQuestion(); }}>
            <button className="btn btn-light rounded-circle" type="button"><i className="bi bi-plus-lg"></i></button>
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
          </form>
        ) : session.status === "paused" ? (
          <footer className="composer">
            <div className="flex-grow-1 small text-body-secondary">
              This session was restored after a daemon restart and is paused until you resume it.
            </div>
            <button
              className="btn btn-warning"
              onClick={() => void resumeSession()}
            >
              Resume
            </button>
          </footer>
        ) : !isRunning && session.status === "completed" ? (
          <form className="composer" onSubmit={(e) => { e.preventDefault(); void continueSession(); }}>
            <button className="btn btn-light rounded-circle" type="button"><i className="bi bi-plus-lg"></i></button>
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
  const path = useRoute();

  const sessionMatch = path.match(/^\/sessions\/([^/]+)$/);
  if (sessionMatch) {
    return <SessionPage sessionId={sessionMatch[1]} />;
  }

  return <HomePage />;
}
