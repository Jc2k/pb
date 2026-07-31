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

export interface HarnessEfficiencyStats {
  proactiveActions: number;
  proactiveReads: number;
  proactiveInspections: number;
  collarCandidatesFiltered: number;
  mutationCandidatesFiltered: number;
  duplicateActionsPrevented: number;
  dependentBatchesPrevented: number;
  noProgressLoopsStopped: number;
}

export function harnessEfficiencyStats(
  events: EventEnvelope[],
): HarnessEfficiencyStats {
  const stats: HarnessEfficiencyStats = {
    proactiveActions: 0,
    proactiveReads: 0,
    proactiveInspections: 0,
    collarCandidatesFiltered: 0,
    mutationCandidatesFiltered: 0,
    duplicateActionsPrevented: 0,
    dependentBatchesPrevented: 0,
    noProgressLoopsStopped: 0,
  };

  for (const envelope of events) {
    const event = envelope.event;
    if (
      event.type === "controller_observation" &&
      event.receipt.included_in_prompt
    ) {
      stats.proactiveActions += 1;
      if (event.receipt.operation === "read_file") {
        stats.proactiveReads += 1;
      } else {
        stats.proactiveInspections += 1;
      }
      continue;
    }

    if (event.type === "llm_invocation" && event.native) {
      stats.collarCandidatesFiltered += Math.max(
        0,
        event.native.rejected_constraint_candidates || 0,
      );
      stats.mutationCandidatesFiltered += Object.values(
        event.native.mutation_constraint_rejections || {},
      ).reduce((total, count) => total + Math.max(0, count), 0);
      continue;
    }

    if (event.type !== "correction") continue;
    if (
      event.summary === "Repeated tool call detected" ||
      event.summary === "Repeated tool call blocked"
    ) {
      stats.duplicateActionsPrevented += 1;
    } else if (event.summary === "Dependent tool batch rejected") {
      stats.dependentBatchesPrevented += 1;
    } else if (event.summary === "No-progress tool outcome detected") {
      stats.noProgressLoopsStopped += 1;
    }
  }

  return stats;
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

type ToolFailureDetail = {
  type?: unknown;
  tool?: unknown;
  message?: unknown;
};

export function toolFailureFeedback(
  detail: string,
  teammateName: string,
): string | null {
  let failure: ToolFailureDetail;
  try {
    failure = JSON.parse(detail) as ToolFailureDetail;
  } catch {
    return null;
  }

  if (
    failure.type !== "tool_failure" || typeof failure.tool !== "string" ||
    !failure.tool.trim()
  ) {
    return null;
  }

  const firstName = teammateName.trim().split(/\s+/, 1)[0] || "Teammate";
  const failureMessage = typeof failure.message === "string"
    ? failure.message
    : "";
  const missingPath = failureMessage.match(/failed to resolve path '([^']+)'/)
    ?.[1];
  const problem = missingPath
    ? `\`${missingPath}\` does not exist.`
    : /permission denied/i.test(failureMessage)
    ? "The requested resource could not be accessed."
    : "The action failed before it returned a result.";

  return `${firstName}, your call to the \`${failure.tool}\` tool was not executed successfully. ${problem} Fix the mistake, choose a different action, or report the blocker.`;
}

export type TrinityCorrectionCopy = {
  headline?: string;
  message: string;
};

function correctionExcerpt(detail: string): string | null {
  const trimmed = detail.trim();
  if (!trimmed || trimmed.startsWith("{") || trimmed.startsWith("[")) {
    return null;
  }
  const firstParagraph = trimmed.split(/\n\s*\n/, 1)[0]
    .replace(/(?:\/private)?\/tmp\/[^\s,;:)]+/g, "the temporary workspace")
    .replace(/\/Users\/[^/\s]+\/[^\s,;:)]+/g, "the workspace")
    .replace(/\s+/g, " ")
    .trim();
  if (firstParagraph.length <= 360) return firstParagraph;
  const sentenceEnd = firstParagraph.lastIndexOf(". ", 340);
  const end = sentenceEnd >= 120 ? sentenceEnd + 1 : 340;
  return `${firstParagraph.slice(0, end).trimEnd()}…`;
}

export function artifactValidationProblem(message: string): string {
  const lower = message.toLowerCase();
  if (
    lower.includes("requires non-empty requirements, steps, and acceptance")
  ) {
    return "The plan was missing its requirements, implementation steps, and acceptance checks.";
  }
  if (lower.includes("requirements") && lower.includes("acceptance")) {
    return "The plan did not include all of the required planning and acceptance information.";
  }
  if (lower.includes("fingerprint")) {
    return "The submission described an older version of the workspace, so it was not safe to accept.";
  }
  if (
    lower.includes("revision") && lower.includes("every assessment passes")
  ) {
    return "The review asked for changes but marked every review area as passing.";
  }
  return "The submission did not match the delivery structure the team needs to continue safely.";
}

export function trinityCorrectionCopy(
  summary: string | undefined,
  detail: string,
  teammateName: string,
  artifactLabel: string,
): TrinityCorrectionCopy {
  const firstName = teammateName.trim().split(/\s+/, 1)[0] || "Teammate";
  const normalizedSummary = summary?.trim() || "";
  const isArtifactValidation = normalizedSummary ===
      "Workflow artifact validation failed" ||
    /^submit_(?:plan|plan_review|implementation|code_review) tool call was not executed successfully$/
      .test(normalizedSummary);
  if (isArtifactValidation) {
    return {
      headline: `${teammateName}’s ${artifactLabel} needs another pass`,
      message: `${
        artifactValidationProblem(detail)
      } I sent it back so you can correct the submission before the team continues.`,
    };
  }

  const toolFailure = toolFailureFeedback(detail, teammateName);
  if (toolFailure) return { message: toolFailure };

  if (normalizedSummary === "Task-focused repository evidence") {
    return {
      message:
        `${firstName}, I found the task-relevant code and pulled out the strongest matching sections. Use them to finish the ${artifactLabel}. If one concrete fact is still missing, read only the relevant lines instead of rereading the whole file.`,
    };
  }

  if (
    normalizedSummary.includes("contract planning evidence") ||
    normalizedSummary.includes("contract review evidence") ||
    normalizedSummary.includes("proposed-path review evidence")
  ) {
    return {
      message:
        `${firstName}, I rechecked the exact code this stage depends on. You have enough evidence now—finish the ${artifactLabel} instead of rereading broad sections of the repository.`,
    };
  }

  if (normalizedSummary === "Active accepted-plan work unit") {
    return {
      message:
        `${firstName}, I picked the next item from the accepted plan and confirmed exactly which file operation it needs. Complete only that item before moving on.`,
    };
  }

  if (normalizedSummary === "Next accepted-plan creation work unit") {
    return {
      message:
        `${firstName}, the next planned file does not exist yet. Create it now with one complete write, then move on to the next item.`,
    };
  }

  if (normalizedSummary.includes("using host execution")) {
    return {
      message:
        "This task includes an Apple-only component, so I’m running that part directly on the Mac while keeping the rest of the session isolated.",
    };
  }

  if (normalizedSummary.includes("CPU-only llama.cpp fallback")) {
    return {
      message:
        "The preferred model runtime was unavailable, so I’m using the CPU-only model fallback for this session. Responses may take longer.",
    };
  }

  if (normalizedSummary.includes("reached the repeat limit")) {
    return {
      message:
        `${firstName}, you repeated the same action after guidance, so I blocked the duplicate before you spent more time on it. Choose a different approach or report the blocker.`,
    };
  }

  if (
    normalizedSummary === "Repeated tool call detected" ||
    normalizedSummary === "Repeated tool call blocked" ||
    normalizedSummary.includes("repeated the same action")
  ) {
    const repeatedTool = detail.match(/-\s+([A-Za-z_][\w-]*) with args/)?.[1];
    return {
      message: repeatedTool
        ? `${firstName}, you repeated the same \`${repeatedTool}\` call, so I blocked the duplicate before it ran. Change the path or action, or report that you are blocked.`
        : `${firstName}, you repeated the same action, so I blocked the duplicate before it ran. Change approach or report that you are blocked.`,
    };
  }

  if (normalizedSummary === "No-progress tool outcome detected") {
    return {
      message:
        `${firstName}, that action returned the same outcome without adding new evidence. I stopped the loop; choose an action that changes the work or report the blocker.`,
    };
  }

  if (normalizedSummary === "Dependent tool batch rejected") {
    return {
      message:
        `${firstName}, those tool calls depend on one another, so I did not run them as one batch. Run the prerequisite first, wait for its result, then submit the dependent action.`,
    };
  }

  if (normalizedSummary === "Workflow stage submission required") {
    return {
      message:
        `${firstName}, a prose reply will not complete this stage. Submit the ${artifactLabel} in the required format so the team can continue.`,
    };
  }

  if (
    normalizedSummary === "Teammate action retries exhausted" ||
    normalizedSummary.includes("Parse retry limit")
  ) {
    return {
      message:
        `${firstName}, your reply still did not form a valid action after several retries, so I stopped the pass instead of letting it loop. Start again with one small, complete action.`,
    };
  }

  if (
    normalizedSummary.includes("Invalid pb JSON action") ||
    normalizedSummary.toLowerCase().includes("unparsable")
  ) {
    return {
      message:
        `${firstName}, that reply was not a valid action, so nothing ran. Retry with one complete tool call or finish the stage in the required format.`,
    };
  }

  if (normalizedSummary.includes("reached the bounded step limit")) {
    return {
      message:
        `${firstName}, you reached this pass’s step limit before completing the work, so I stopped it instead of letting it run in circles. Continue with a tighter next action or report the blocker.`,
    };
  }

  if (normalizedSummary === "Advisory budget exhausted") {
    return {
      message:
        "I skipped the optional step-limit review because its advisory budget was already used. The main work and repository were left unchanged.",
    };
  }

  if (normalizedSummary === "Requesting missing bounded evidence") {
    return {
      message:
        `${firstName}, the edit stopped before it became a valid action because one small file excerpt is still missing. Read only the lines around that detail, then retry the edit.`,
    };
  }

  if (normalizedSummary.includes("truncated action")) {
    return {
      message:
        `${firstName}, the action was cut off before it became valid, so nothing ran. Try again once with one concise, complete tool call.`,
    };
  }

  if (
    normalizedSummary.includes("mutation recovery") ||
    normalizedSummary.includes("compact atomic mutation") ||
    normalizedSummary.includes("constrained mutation")
  ) {
    return {
      message:
        `${firstName}, the edit was incomplete and was not executed. I’m giving you one fresh attempt for the smallest complete change; do not repeat the rejected payload.`,
    };
  }

  if (normalizedSummary === "Run cancelled") {
    return {
      message:
        "This run was cancelled. I preserved the repository and the evidence collected so far.",
    };
  }

  if (normalizedSummary === "Goal pausing") {
    return {
      message:
        "I’m pausing the goal at a safe checkpoint before anyone starts another action.",
    };
  }

  if (normalizedSummary === "Cancellation requested") {
    return {
      message:
        "Cancellation is requested. I’m preserving the repository and the workflow evidence while the current work stops safely.",
    };
  }

  if (normalizedSummary === "Restarting delivery from current files") {
    return {
      message:
        "I kept the earlier plan and review in the transcript, accepted the project’s current files as the new baseline, and started a fresh planning pass.",
    };
  }

  if (normalizedSummary === "Retrying Task planning") {
    return {
      message:
        "I’m retrying task planning from the preserved repository state. No files or commits change until the new workflow begins delivery.",
    };
  }

  if (normalizedSummary === "Running as one Build") {
    return {
      message:
        "I’m keeping the repository as-is and retrying this request as one Build task instead of splitting it into several tasks.",
    };
  }

  if (normalizedSummary === "Tool not available") {
    return {
      message:
        `${firstName}, that tool is not available in this stage, so the action did not run. Choose one of the available actions or report the blocker.`,
    };
  }

  if (normalizedSummary.includes("Task requirements remain")) {
    return {
      message:
        `${firstName}, the handoff still leaves part of the user’s request unfinished. I sent the missing requirements back for one focused repair pass.`,
    };
  }

  if (normalizedSummary === "Handoff executor unavailable") {
    return {
      message:
        "I could not run the final handoff checks because their executor is unavailable. I preserved the work so the checks can be retried when it returns.",
    };
  }

  if (normalizedSummary === "Handoff commit blocked") {
    return {
      message:
        "The final commit was blocked, so I left the completed changes uncommitted and preserved the handoff evidence for review.",
    };
  }

  if (
    normalizedSummary.startsWith("Agent terminated:") ||
    normalizedSummary.startsWith("Final grace terminated:")
  ) {
    return {
      message:
        `${firstName}, your pass ended before it produced a usable result. I preserved every completed action and stopped at the current safe boundary.`,
    };
  }

  if (normalizedSummary === "Harness diagnostic preview") {
    return {
      message:
        `${firstName}, I ran an early diagnostic check and found issues you should account for while you complete the current work item.`,
    };
  }

  if (normalizedSummary.toLowerCase().includes("diagnostic")) {
    return {
      message:
        `${firstName}, the automatic diagnostics found issues that need repair before this stage can continue. Fix the reported problems, then retry the handoff.`,
    };
  }

  const excerpt = correctionExcerpt(detail);
  if (excerpt) {
    return {
      headline: normalizedSummary || undefined,
      message: excerpt,
    };
  }

  return {
    headline: normalizedSummary || "Trinity update",
    message:
      `${firstName}, I could not summarize this safely without losing the exact cause. Check the technical details before choosing your next action.`,
  };
}

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
      event.event.summary?.includes("repeated the same action") === true ||
      event.event.summary?.includes("reached the repeat limit") === true);
}

function isTerminalToolLoopError(event: EventEnvelope): boolean {
  return event.event.type === "error" &&
    (event.event.summary === "Deterministic tool loop" ||
      event.event.summary === "No-progress tool loop" ||
      event.event.summary?.includes("reached the repeat limit") === true);
}

function repeatsEarlierCorrectionAfterAction(
  events: EventEnvelope[],
  index: number,
): boolean {
  const current = events[index]?.event;
  if (current?.type !== "correction") return false;
  const failureIdentity = (
    correction: Extract<AgentEvent, { type: "correction" }>,
  ) => {
    try {
      const detail = JSON.parse(correction.message) as ToolFailureDetail;
      if (
        detail.type !== "tool_failure" || typeof detail.tool !== "string" ||
        typeof detail.message !== "string"
      ) return undefined;
      const failedPath = detail.message.match(
        /failed to resolve path '([^']+)'/,
      )
        ?.[1];
      return failedPath
        ? `${detail.tool}:missing:${failedPath}`
        : `${detail.tool}:${detail.message}`;
    } catch {
      return undefined;
    }
  };
  const currentFailure = failureIdentity(current);
  const precedingToolCall = (beforeIndex: number) => {
    for (let toolIndex = beforeIndex - 1; toolIndex >= 0; toolIndex--) {
      const candidate = events[toolIndex];
      if (candidate.event.type === "tool_call") return candidate;
    }
    return undefined;
  };
  const currentCall = precedingToolCall(index);
  if (currentCall?.event.type !== "tool_call") return false;
  for (let priorIndex = index - 1; priorIndex >= 0; priorIndex--) {
    const prior = events[priorIndex].event;
    const sameCorrection = prior.type === "correction" &&
      (prior.summary === current.summary ||
        (currentFailure !== undefined &&
          failureIdentity(prior) === currentFailure));
    if (sameCorrection && prior.type === "correction") {
      const priorCall = precedingToolCall(priorIndex);
      if (
        priorCall?.event.type === "tool_call" &&
        priorCall.event.tool === currentCall.event.tool &&
        JSON.stringify(priorCall.event.arguments) ===
          JSON.stringify(currentCall.event.arguments)
      ) {
        return true;
      }
    }
  }
  return false;
}

function reachesWorkflowBlockBeforeMoreVisibleWork(
  events: EventEnvelope[],
  index: number,
): boolean {
  for (let laterIndex = index + 1; laterIndex < events.length; laterIndex++) {
    const later = events[laterIndex].event;
    if (later.type === "workflow_blocked") return true;
    if (
      later.type === "tool_call" || later.type === "reasoning" ||
      later.type === "final" || later.type === "team_message" ||
      later.type === "user_message" || later.type === "user_answer" ||
      later.type === "started" || later.type === "controller_mutation" ||
      later.type === "controller_observation" ||
      later.type === "controller_closure"
    ) {
      return false;
    }
  }
  return false;
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
      reachesWorkflowBlockBeforeMoreVisibleWork(chatEvents, index)
    ) {
      return false;
    }
    if (
      event.event.type === "correction" &&
      repeatsEarlierCorrectionAfterAction(chatEvents, index) &&
      reachesWorkflowBlockBeforeMoreVisibleWork(chatEvents, index)
    ) {
      // Keep the first actionable explanation and the repeated model action, then let the terminal
      // Trinity message own the outcome. Repeating the same failure card immediately before that
      // outcome makes it look as if Trinity spoke twice without intervening work.
      return false;
    }
    if (
      event.event.type === "correction" &&
      isRepeatedToolCorrection(event) &&
      event.event.summary !== "Repeated tool call detected" &&
      reachesWorkflowBlockBeforeMoreVisibleWork(chatEvents, index)
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
