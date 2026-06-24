import type { AgentEvent, EventEnvelope } from "../types";
import { TOOL_FRIENDLY_NAMES, TOOL_ICONS } from "./constants";

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

  switch (toolCall.event.tool) {
    case "read_file": {
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
    }
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
    case "rm": {
      const filePath = args ? (args.path as string) : undefined;
      return filePath || "(no path)";
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
    case "git_revert":
      return (args.commit as string) || "(no commit)";
    default:
      if (toolResult && toolResult.event.type === "tool_result") {
        try {
          const parsed = JSON.parse(toolResult.event.result);
          if (Array.isArray(parsed)) return `${parsed.length} items`;
          if (typeof parsed === "object" && parsed !== null) return `result (${Object.keys(parsed).length} fields)`;
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
    const resultTool = event.event.tool;
    const callIndex = pendingCalls.findIndex(
      (call) => call.event.type === "tool_call" && call.event.tool === resultTool,
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

export function buildToolSummaries(events: EventEnvelope[]): ToolSummary[] {
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

function addToolSummaryItem(summaries: Record<string, ToolSummary>, call: EventEnvelope, result?: EventEnvelope) {
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

function isHiddenChatEvent(event: EventEnvelope): boolean {
  return event.event.type === "sub_agent_started" || event.event.type === "sub_agent_finished";
}

export function chatEventsWithOnlyLatestStep(events: EventEnvelope[]): EventEnvelope[] {
  const chatEvents = events.filter((event) => !isHiddenChatEvent(event));
  const lastVisibleIndex = chatEvents.length - 1;
  return chatEvents.filter((event, index) => event.event.type !== "step_started" || index === lastVisibleIndex);
}

export function profileName(profile: string): string {
  switch (profile) {
    case "plan": return "Dade Murphy";
    case "build": return "Kate Libby";
    case "review": return "Eugene Belford";
    case "scout": return "Ramon Sanchez";
    case "explore": return "Paul Cook";
    case "research": return "Emmanuel Goldstein";
    case "monitor": return "Trinity Walker";
    case "ask": return "Joey Pardella";
    default: return "Jon Appleseed";
  }
}

export function profileJobTitle(profile: string): string {
  switch (profile) {
    case "plan": return "Ticket Goblin";
    case "build": return "Patch Crafter";
    case "review": return "Review Gremlin";
    case "scout": return "Env Scout";
    case "explore": return "Repo Mapper";
    case "research": return "Web Sleuth";
    case "monitor": return "Progress Monitor";
    case "ask": return "Question Wrangler";
    default: return "Unknown";
  }
}

export function errorSummary(event: Extract<AgentEvent, { type: "error" }>): string {
  const summary = event.summary?.trim();
  if (summary) return summary;
  const message = String(event.message || "").trim();
  const firstLine = message.split("\n").find((line) => line.trim())?.trim();
  if (!firstLine) return "Agent error";
  return firstLine.length > 120 ? `${firstLine.slice(0, 117)}…` : firstLine;
}
