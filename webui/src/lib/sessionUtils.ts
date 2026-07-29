import type { AgentEvent, EventEnvelope, TeamActor } from "../types";
import { TOOL_FRIENDLY_NAMES, TOOL_ICONS } from "./constants";
import { formatEnergy, formatPower } from "./energy";
import { toolEventsMatch } from "./helpers";
import { workflowStewardActor } from "./team";

export { profileJobTitle, profileName } from "./team";

export interface ToolSummaryItem {
  detail: string;
  timestampMs?: number;
}

export interface ToolSummary {
  toolName: string;
  friendlyName: string;
  icon: string;
  count: number;
  items: ToolSummaryItem[];
}

export interface ActionTimelineItem {
  actor?: TeamActor;
  assistingProfile?: string;
  envelope: EventEnvelope;
  result?: EventEnvelope;
}

export type TodoStatus = "pending" | "in_progress" | "completed" | "blocked";

export interface TodoTask {
  id: number;
  title: string;
  description: string;
  status: TodoStatus;
  parent_id?: number | null;
  notes?: string[];
  timestampMs?: number;
}

export const TODO_STATUS_LABELS: Record<TodoStatus, string> = {
  pending: "Pending",
  in_progress: "In progress",
  completed: "Completed",
  blocked: "Blocked",
};

export function getToolDetail(
  toolCall: EventEnvelope,
  toolResult?: EventEnvelope,
): string | null {
  if (toolCall.event.type !== "tool_call") return null;

  const args = toolCall.event.arguments as Record<string, unknown>;

  const scopedValue = (
    value: unknown,
    scope: unknown,
    fallback: string,
  ): string => {
    const label = typeof value === "string" && value ? value : fallback;
    return typeof scope === "string" && scope
      ? `${label} · in ${scope}`
      : label;
  };

  const arrayCount = (value: unknown): number =>
    Array.isArray(value) ? value.length : 0;

  switch (toolCall.event.tool) {
    case "read_file": {
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
    }
    case "inspect_change":
      return (args.path as string) || "(no path)";
    case "glob":
      return scopedValue(
        args.pattern,
        args.path ?? args.relative_path,
        "(no pattern)",
      );
    case "ripgrep":
    case "search":
      return scopedValue(args.pattern, args.path, "(no pattern)");
    case "web_search":
      return (args.query as string) || "(no query)";
    case "web_fetch":
      return (args.url as string) || "(no url)";
    case "run_command":
      return (args.cmd as string) || "(no cmd)";
    case "run_task":
    case "run_check":
      return (args.id as string) || "(no id)";
    case "session_changes": {
      const filters = [
        typeof args.path === "string" && args.path
          ? `File: ${args.path}`
          : null,
        typeof args.commits === "string" && args.commits
          ? `Commits: ${args.commits}`
          : null,
      ].filter(Boolean);
      return filters.length > 0
        ? filters.join(" · ")
        : "Recent sessions and changes";
    }
    case "lsp_proactive_diagnostics": {
      const mode = typeof args.mode === "string" ? args.mode : "automatic";
      const requested = Array.isArray(args.paths) ? args.paths.length : 0;
      if (!toolResult || toolResult.event.type !== "tool_result") {
        return `${mode} · ${requested} ${
          requested === 1 ? "file" : "files"
        } (pending)`;
      }
      try {
        const report = JSON.parse(toolResult.event.result) as {
          scanned_paths?: unknown[];
          diagnostics?: unknown[];
          failures?: unknown[];
          omitted_paths?: number;
          stale?: boolean;
          complete?: boolean;
          requested_targets?: unknown[];
          completed_targets?: unknown[];
          incomplete_reasons?: string[];
        };
        const scanned = report.scanned_paths?.length || 0;
        const diagnostics = report.diagnostics?.length || 0;
        const failures = report.failures?.length || 0;
        const omitted = report.omitted_paths || 0;
        const requestedTargets = report.requested_targets?.length || 0;
        const completedTargets = report.completed_targets?.length || 0;
        if (report.stale) return `${mode} · stale evidence discarded`;
        if (diagnostics > 0) {
          return `${mode} · ${diagnostics} blocking ${
            diagnostics === 1 ? "diagnostic" : "diagnostics"
          } in ${scanned} ${scanned === 1 ? "file" : "files"}${
            omitted > 0 ? ` · ${omitted} deferred` : ""
          }`;
        }
        if (failures > 0) {
          return `${mode} · ${scanned}/${requested} files · ${failures} ${
            failures === 1 ? "server issue" : "server issues"
          }${omitted > 0 ? ` · ${omitted} deferred` : ""}`;
        }
        if (report.complete !== true) {
          return `${mode} · incomplete evidence · ${completedTargets}/${requestedTargets} server/file targets${
            omitted > 0 ? ` · ${omitted} deferred` : ""
          }`;
        }
        if (omitted > 0) {
          return `${mode} · ${scanned} files · ${omitted} deferred`;
        }
        return `${mode} · ${scanned} ${
          scanned === 1 ? "file" : "files"
        } · clean`;
      } catch {
        return `${mode} · ${requested} ${requested === 1 ? "file" : "files"}`;
      }
    }
    case "skill_search": {
      const query = args.query as string;
      if (!query) return "";
      if (!toolResult) return `${query} (pending)`;
      const skillMatches = toolResult.event.type === "tool_result"
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
      return `from ${(args.source as string) || ""} to ${
        (args.destination as string) || ""
      }`;
    case "rm": {
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
    }
    case "write_file":
    case "replace_file": {
      const path = args ? (args.path as string) : undefined;
      return path || "(no path)";
    }
    case "edit_file": {
      const path = args.path as string;
      if (!path) return "(no path)";
      return path + (args.diff ? " (patch)" : "");
    }
    case "apply_patch": {
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
    }
    case "git_commit":
      return (args.message as string) || "(no message)";
    case "session_title":
      return (args.title as string) || "(no title)";
    case "memory_search":
      return (args.query as string) || "All relevant project memory";
    case "memory_read":
      return (args.id as string) || "(no memory id)";
    case "memory_propose":
      return (args.title as string) || (args.kind as string) ||
        "New project memory";
    case "memory_supersede":
      return (args.id as string) || "(no memory id)";
    case "propose_delivery":
    case "start_delivery":
      return (args.task_summary as string) || "(no delivery summary)";
    case "propose_goal":
    case "start_goal":
      return (args.objective as string) || "(no goal objective)";
    case "goal_pause":
    case "goal_request_budget":
      return (args.reason as string) || "(no reason)";
    case "goal_request_amendment":
      return (args.summary as string) || "(no change summary)";
    case "submit_plan": {
      const requirements = arrayCount(args.requirements);
      const steps = arrayCount(args.steps);
      const acceptance = arrayCount(args.acceptance);
      if (requirements === 0 || steps === 0 || acceptance === 0) {
        return "Incomplete plan · missing required sections";
      }
      return `${requirements} ${
        requirements === 1 ? "requirement" : "requirements"
      } · ${steps} ${
        steps === 1 ? "step" : "steps"
      } · ${acceptance} acceptance ${acceptance === 1 ? "check" : "checks"}`;
    }
    case "submit_plan_review":
    case "submit_code_review": {
      const rawVerdict = typeof args.verdict === "string"
        ? args.verdict.replaceAll("_", " ")
        : "";
      const verdict = rawVerdict
        ? `${rawVerdict[0].toUpperCase()}${rawVerdict.slice(1)}`
        : "Review submitted";
      const concerns = arrayCount(args.challenges || args.findings);
      return concerns > 0
        ? `${verdict} · ${concerns} ${concerns === 1 ? "finding" : "findings"}`
        : verdict;
    }
    case "submit_implementation": {
      const steps = arrayCount(args.steps);
      return `${steps} implementation ${steps === 1 ? "step" : "steps"}`;
    }
    case "request_replan":
      return (args.reason as string) || "(no reason)";
    case "git_revert":
      return (args.commit as string) || "(no commit)";
    default:
      if (toolResult && toolResult.event.type === "tool_result") {
        try {
          const parsed = JSON.parse(toolResult.event.result);
          if (Array.isArray(parsed)) return `${parsed.length} items`;
          if (typeof parsed === "object" && parsed !== null) {
            return `result (${Object.keys(parsed).length} fields)`;
          }
        } catch {
          const result = toolResult.event.result;
          if (result.length < 80) return result.replace(/\n/g, " ");
        }
      }
      if (!toolResult) return "(pending)";
      return null;
  }
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

export function buildTodoTasks(events: EventEnvelope[]): TodoTask[] {
  const tasks = new Map<number, TodoTask>();
  const pendingCalls: EventEnvelope[] = [];

  events.forEach((event) => {
    if (event.event.type === "tool_call") {
      pendingCalls.push(event);
      return;
    }
    if (event.event.type !== "tool_result") return;
    const callIndex = pendingCalls.findIndex((call) =>
      toolEventsMatch(call, event)
    );
    const call = callIndex >= 0
      ? pendingCalls.splice(callIndex, 1)[0]
      : undefined;
    if (event.event.tool !== "todo") return;
    const parsedTasks = parseTodoTasks(event.event.result);
    if (!parsedTasks) return;
    const action = call?.event.type === "tool_call"
      ? ((call.event.arguments as Record<string, unknown> | undefined)
        ?.action as string | undefined)
      : undefined;
    if (action === "list") tasks.clear();
    parsedTasks.forEach((task) => {
      tasks.set(task.id, {
        ...tasks.get(task.id),
        ...task,
        timestampMs: event.event.timestamp_ms,
      });
    });
  });

  return Array.from(tasks.values()).sort((a, b) => a.id - b.id);
}

export function buildToolSummaries(events: EventEnvelope[]): ToolSummary[] {
  const summaries: Record<string, ToolSummary> = {};
  const pendingCalls: EventEnvelope[] = [];
  events.forEach((event) => {
    if (event.event.type === "tool_call") {
      pendingCalls.push(event);
      return;
    }
    if (event.event.type === "tool_result" && pendingCalls.length > 0) {
      const index = pendingCalls.findIndex((call) =>
        toolEventsMatch(call, event)
      );
      const call = index >= 0 ? pendingCalls.splice(index, 1)[0] : undefined;
      if (!call || call.event.type !== "tool_call") return;
      addToolSummaryItem(summaries, call, event);
    }
  });
  pendingCalls.forEach((call) => addToolSummaryItem(summaries, call));
  return Object.values(summaries);
}

export function buildActionTimeline(
  events: EventEnvelope[],
): ActionTimelineItem[] {
  const items: ActionTimelineItem[] = [];
  const pending: ActionTimelineItem[] = [];

  events.forEach((envelope) => {
    const event = envelope.event;
    if (event.type === "tool_call") {
      const item = { actor: event.actor, envelope };
      items.push(item);
      pending.push(item);
      return;
    }
    if (event.type === "tool_result") {
      const index = pending.findIndex((item) =>
        toolEventsMatch(item.envelope, envelope)
      );
      const item = index >= 0 ? pending.splice(index, 1)[0] : undefined;
      if (item) item.result = envelope;
      return;
    }
    if (
      event.type === "controller_observation" ||
      event.type === "controller_closure" ||
      event.type === "controller_mutation"
    ) {
      items.push({
        actor: event.actor || workflowStewardActor(),
        assistingProfile: event.assisting_profile,
        envelope,
      });
    }
  });

  return items;
}

export function trustedSessionSummaryCommitLines(
  commits: string | undefined,
  events: EventEnvelope[],
): string[] {
  const lines = commits?.trim()
    ? commits.trim().split("\n").filter(Boolean)
    : [];
  if (lines.length === 0) return [];

  const isStrictWorkflow = events.some((envelope) =>
    envelope.event.type === "workflow_started"
  );
  if (!isStrictWorkflow) return lines;

  const hasCommitReceipt = events.some((envelope) =>
    envelope.event.type === "commit_result" && envelope.event.success &&
    (envelope.event.created || envelope.event.reused || envelope.event.oid)
  );
  return hasCommitReceipt ? lines : [];
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
  const detail = getToolDetail(call, result) || "(no details)";
  const durationMs = result?.event.type === "tool_result"
    ? result.event.duration_ms
    : undefined;
  const duration = durationMs === undefined
    ? ""
    : durationMs < 1000
    ? ` · ${durationMs} ms`
    : ` · ${(durationMs / 1000).toFixed(1)} s`;
  const energyJoules = result?.event.type === "tool_result"
    ? result.event.energy_joules
    : undefined;
  const averagePower = result?.event.type === "tool_result"
    ? result.event.average_power_watts
    : undefined;
  const sharedCalls = result?.event.type === "tool_result"
    ? result.event.energy_shared_calls
    : undefined;
  const energy = energyJoules === undefined
    ? ""
    : ` · ${formatEnergy(energyJoules)}${
      averagePower === undefined ? "" : ` at ${formatPower(averagePower)}`
    }${
      sharedCalls && sharedCalls > 1
        ? ` across ${sharedCalls} parallel calls`
        : ""
    }`;
  summaries[toolName].items.push({
    detail: `${detail}${duration}${energy}`,
    timestampMs: call.event.timestamp_ms,
  });
}

function isHiddenChatEvent(event: EventEnvelope): boolean {
  const handoffCorrection = event.event.type === "correction" && [
    "Acceptance contract rejected final response",
    "Completion gate blocked final response",
    "The handoff teammate returned failed checks for repair",
  ].includes(event.event.summary || "");
  const internalCheckpoint = event.event.type === "correction" &&
    event.event.summary === "Workflow closure checkpoint";
  const internalProgressCredit = event.event.type === "correction" &&
    event.event.summary === "Work-unit progress earned one bounded turn";
  return event.event.type === "sub_agent_started" ||
    event.event.type === "sub_agent_finished" ||
    event.event.type === "user_message_applied" ||
    event.event.type === "executor_started" ||
    event.event.type === "check_result" ||
    event.event.type === "commit_result" ||
    event.event.type === "handoff_summary" ||
    event.event.type === "final_grace" ||
    handoffCorrection ||
    internalCheckpoint ||
    internalProgressCredit;
}

function isRepeatedToolCorrection(event: EventEnvelope): boolean {
  return event.event.type === "correction" &&
    (event.event.summary === "Repeated tool call detected" ||
      event.event.summary === "Repeated tool call blocked" ||
      event.event.summary === "No-progress tool outcome detected" ||
      event.event.summary?.includes("repeated the same action") === true);
}

function isTerminalToolLoopError(event: EventEnvelope): boolean {
  return event.event.type === "error" &&
    (event.event.summary === "Deterministic tool loop" ||
      event.event.summary === "No-progress tool loop" ||
      event.event.summary?.includes("reached the repeat limit") === true);
}

function isTransientActivityEvent(event: EventEnvelope): boolean {
  return event.event.type === "model_loading" ||
    event.event.type === "step_started";
}

function withoutDuplicateSessionSummary(
  event: EventEnvelope,
  previousFinalContent?: string,
): EventEnvelope {
  if (
    event.event.type !== "session_summary" || !event.event.summary ||
    !previousFinalContent
  ) {
    return event;
  }

  if (event.event.summary.trim() !== previousFinalContent.trim()) {
    return event;
  }

  const { summary: _summary, ...sessionSummary } = event.event;
  return {
    ...event,
    event: sessionSummary,
  };
}

export function chatEventsWithOnlyLatestStep(
  events: EventEnvelope[],
): EventEnvelope[] {
  let lastFinalContent: string | undefined;
  let lastWorkflowBlockReason: string | undefined;
  const chatEvents = events
    .filter((event) => !isHiddenChatEvent(event))
    .map((event) => {
      let normalized = withoutDuplicateSessionSummary(
        event,
        lastFinalContent,
      );
      if (event.event.type === "final") lastFinalContent = event.event.content;
      if (event.event.type === "workflow_blocked") {
        lastWorkflowBlockReason = event.event.reason;
      }
      if (
        normalized.event.type === "session_summary" &&
        normalized.event.summary?.trim() === lastWorkflowBlockReason?.trim()
      ) {
        const { summary: _summary, ...sessionSummary } = normalized.event;
        normalized = { ...normalized, event: sessionSummary };
      }
      return normalized;
    });
  const lastVisibleIndex = chatEvents.length - 1;
  return chatEvents.filter((event, index) => {
    if (isTransientActivityEvent(event) && index !== lastVisibleIndex) {
      return false;
    }
    if (
      event.event.type === "correction" &&
      event.event.summary === "Repeated tool call detected" &&
      chatEvents.slice(index + 1, index + 5).some(isRepeatedToolCorrection)
    ) {
      return false;
    }
    if (
      event.event.type === "correction" &&
      isRepeatedToolCorrection(event) &&
      event.event.summary !== "Repeated tool call detected" &&
      chatEvents.slice(index + 1, index + 4).some((later) =>
        later.event.type === "workflow_blocked"
      )
    ) {
      // Terminal delivery feedback combines the repeat stop with the workflow outcome so Trinity
      // does not appear as two adjacent cards for one stopped pass.
      return false;
    }
    if (
      isTerminalToolLoopError(event) &&
      chatEvents.slice(Math.max(0, index - 2), index).some(
        isRepeatedToolCorrection,
      )
    ) {
      // The preceding Trinity correction owns the user-facing explanation. Keep the typed error
      // in stored evidence and the details view, but do not render a second actorless red card.
      return false;
    }
    if (
      event.event.type === "team_message" &&
      event.event.actor.kind === "automation" &&
      (event.event.actor.id === "handoff" ||
        event.event.actor.id === "trinity") &&
      event.event.tone === "info"
    ) {
      return !chatEvents.slice(index + 1).some((later) =>
        later.event.type === "team_message" &&
        later.event.actor.kind === "automation" &&
        (later.event.actor.id === "handoff" ||
          later.event.actor.id === "trinity")
      );
    }
    return true;
  });
}

export function latestAssistantProfile(
  events: EventEnvelope[],
): string | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i].event;
    if (
      event.type === "started" ||
      event.type === "reasoning" ||
      event.type === "final" ||
      event.type === "user_question" ||
      event.type === "sub_agent_started" ||
      event.type === "sub_agent_finished"
    ) {
      return event.profile;
    }
  }
  return undefined;
}

export function errorSummary(
  event: Extract<AgentEvent, { type: "error" }>,
): string {
  const summary = event.summary?.trim();
  if (summary) return summary;
  const message = String(event.message || "").trim();
  const firstLine = message.split("\n").find((line) => line.trim())?.trim();
  if (!firstLine) return "Agent error";
  return firstLine.length > 120 ? `${firstLine.slice(0, 117)}…` : firstLine;
}
