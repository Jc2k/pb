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

    switch (envelope.transcript?.kind) {
      case "repeated_tool_detected":
      case "repeated_tool_correction":
        stats.duplicateActionsPrevented += 1;
        break;
      case "dependent_tool_batch_correction":
        stats.dependentBatchesPrevented += 1;
        break;
      case "no_progress_correction":
        stats.noProgressLoopsStopped += 1;
        break;
    }
  }

  return stats;
}

export function getToolDetail(
  toolCall: EventEnvelope,
  toolResult?: EventEnvelope,
): string | null {
  if (toolCall.event.type !== "tool_call") return null;
  return toolResult?.transcript?.tool_summary ??
    toolCall.transcript?.tool_summary ?? null;
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

function withoutRedundantSessionSummary(event: EventEnvelope): EventEnvelope {
  if (
    event.event.type !== "session_summary" || !event.event.summary ||
    !event.transcript?.summary_redundant
  ) {
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
  const superseded = new Set(
    events.flatMap((event) => event.transcript?.supersedes || []),
  );
  const chatEvents = events
    .filter((event) =>
      event.transcript?.visibility !== "evidence_only" &&
      (!event.transcript?.entry_key ||
        !superseded.has(event.transcript.entry_key))
    )
    .map(withoutRedundantSessionSummary);
  const lastVisibleIndex = chatEvents.length - 1;
  return chatEvents.filter((event, index) => {
    return event.transcript?.visibility !== "activity" ||
      index === lastVisibleIndex;
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
