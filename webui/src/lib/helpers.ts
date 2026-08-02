import type {
  EventEnvelope,
  HandoffOutcome,
  InstalledIntegration,
  MarketplaceIntegration,
  ProjectEntry,
  ProjectSessionTerminalTransition,
  ProjectUsageStats,
  SessionItem,
  TeamActor,
} from "../types/index";
import { metricEnergyJoules, metricRuntimeMs } from "./energy";
import { teamActorKey } from "./team";

export { getAvatarForProfile } from "./team";

export function uniqueIntegrations<
  T extends Pick<MarketplaceIntegration, "kind" | "name" | "container_image">,
>(entries: T[]): T[] {
  const seen = new Set<string>();
  return entries.filter((entry) => {
    const key = `${entry.kind}:${entry.name}:${entry.container_image}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function uniqueInstalledIntegrations(
  entries: InstalledIntegration[],
): InstalledIntegration[] {
  const seen = new Set<string>();
  return entries.filter((entry) => {
    const key = `${entry.kind}:${entry.name}:${entry.container_image}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function isIntegrationInstalled(
  item: MarketplaceIntegration,
  installed: InstalledIntegration[],
): boolean {
  return installed.some((entry) =>
    entry.kind === item.kind &&
    (entry.container_image === item.container_image || entry.name === item.name)
  );
}

/* ─── helpers ────────────────────────────────────────────────── */

export function sessionTitle(
  session: Pick<SessionItem, "task" | "title">,
): string {
  return session.title?.trim() || session.task;
}

export function sessionPageDocumentTitle(
  session: Pick<SessionItem, "task" | "title">,
): string {
  return `pb session: ${sessionTitle(session)}`;
}

export type ActionGroup = {
  type: "action_group";
  actor: TeamActor;
  assistingProfile?: string;
  inferenceEvents: EventEnvelope[];
  toolCalls: EventEnvelope[];
  toolResults: EventEnvelope[];
  controllerActions: EventEnvelope[];
};

export type ChatPresentationItem = {
  item: EventEnvelope | ActionGroup;
  showIdentity: boolean;
  timeDividerMs?: number;
};

export const CHAT_TIME_GAP_MS = 5 * 60 * 1000;

export function isActionGroup(
  item: EventEnvelope | ActionGroup,
): item is ActionGroup {
  return "type" in item && item.type === "action_group";
}

export function toolEventsMatch(
  call: EventEnvelope,
  result: EventEnvelope,
): boolean {
  if (call.event.type !== "tool_call" || result.event.type !== "tool_result") {
    return false;
  }
  return call.event.call_id === result.event.call_id;
}

export function toolResultForCall(
  call: EventEnvelope,
  results: EventEnvelope[],
): EventEnvelope | undefined {
  return results.find((result) => toolEventsMatch(call, result));
}

export function isControllerActionEvent(event: EventEnvelope): boolean {
  return event.event.type === "controller_observation" ||
    event.event.type === "controller_closure" ||
    event.event.type === "controller_mutation";
}

export function groupActionEvents(
  events: EventEnvelope[],
): (EventEnvelope | ActionGroup)[] {
  const grouped: (EventEnvelope | ActionGroup)[] = [];

  let currentToolCalls: EventEnvelope[] = [];
  let currentToolResults: EventEnvelope[] = [];
  let currentControllerActions: EventEnvelope[] = [];
  let currentInferenceEvents: EventEnvelope[] = [];
  let pendingInferenceEvents: EventEnvelope[] = [];
  let pendingInferenceActor: TeamActor | undefined;
  let pendingInferenceHasOutput = false;
  let currentActor: TeamActor | undefined;
  let currentAssistingProfile: string | undefined;

  const flush = () => {
    const hasActions = currentToolCalls.length > 0 ||
      currentToolResults.length > 0 || currentControllerActions.length > 0;
    if (!hasActions) {
      grouped.push(...currentInferenceEvents);
      currentInferenceEvents = [];
      currentActor = undefined;
      currentAssistingProfile = undefined;
      return;
    }
    grouped.push({
      type: "action_group",
      actor: currentActor!,
      assistingProfile: currentAssistingProfile,
      inferenceEvents: [...currentInferenceEvents],
      toolCalls: [...currentToolCalls],
      toolResults: [...currentToolResults],
      controllerActions: [...currentControllerActions],
    });
    currentToolCalls = [];
    currentToolResults = [];
    currentControllerActions = [];
    currentInferenceEvents = [];
    currentActor = undefined;
    currentAssistingProfile = undefined;
  };

  const beginOrSwitchGroup = (actor: TeamActor, assistingProfile?: string) => {
    const hasActions = currentToolCalls.length > 0 ||
      currentToolResults.length > 0 || currentControllerActions.length > 0 ||
      currentInferenceEvents.length > 0;
    if (
      hasActions &&
      (currentActor === undefined ||
        teamActorKey(currentActor) !== teamActorKey(actor) ||
        currentAssistingProfile !== assistingProfile)
    ) flush();
    const groupIsEmpty = currentToolCalls.length === 0 &&
      currentToolResults.length === 0 &&
      currentControllerActions.length === 0 &&
      currentInferenceEvents.length === 0;
    if (groupIsEmpty) {
      currentActor = actor;
      currentAssistingProfile = assistingProfile;
    }
  };

  const flushPendingInferences = () => {
    if (pendingInferenceEvents.length === 0) return;
    flush();
    if (pendingInferenceHasOutput) grouped.push(...pendingInferenceEvents);
    pendingInferenceEvents = [];
    pendingInferenceActor = undefined;
    pendingInferenceHasOutput = false;
  };

  for (let i = 0; i < events.length; i++) {
    const event = events[i];

    if (event.event.type === "llm_invocation") {
      flushPendingInferences();
      pendingInferenceActor = {
        kind: "agent" as const,
        id: event.event.profile,
      };
      pendingInferenceEvents.push(event);
    } else if (
      event.event.type === "reasoning" ||
      event.event.type === "final" ||
      event.event.type === "user_question"
    ) {
      flush();
      grouped.push(event);
      pendingInferenceHasOutput = pendingInferenceEvents.length > 0;
    } else if (event.event.type === "tool_call") {
      const actor = event.event.actor;
      if (
        pendingInferenceEvents.length > 0 &&
        pendingInferenceActor !== undefined &&
        teamActorKey(pendingInferenceActor) !== teamActorKey(actor)
      ) {
        flushPendingInferences();
      }
      beginOrSwitchGroup(actor);
      currentInferenceEvents.push(...pendingInferenceEvents);
      pendingInferenceEvents = [];
      pendingInferenceActor = undefined;
      pendingInferenceHasOutput = false;
      currentToolCalls.push(event);
    } else if (event.event.type === "tool_result") {
      const currentMatch = currentToolCalls.some((call) =>
        toolEventsMatch(call, event) &&
        !toolResultForCall(call, currentToolResults)
      );
      if (currentMatch) {
        currentToolResults.push(event);
        continue;
      }
      const prior = [...grouped].reverse().find((candidate) =>
        "type" in candidate && candidate.type === "action_group" &&
        candidate.toolCalls.some((call) =>
          toolEventsMatch(call, event) &&
          !toolResultForCall(call, candidate.toolResults)
        )
      );
      if (prior && "type" in prior && prior.type === "action_group") {
        prior.toolResults.push(event);
        continue;
      }
      flush();
      grouped.push(event);
    } else if (event.event.type === "tool_batch") {
      // Batch bookkeeping is represented by the calls themselves. It must not split adjacent
      // tool-only turns by the same teammate into several chat messages.
      continue;
    } else if (
      event.event.type === "controller_observation" ||
      event.event.type === "controller_closure" ||
      event.event.type === "controller_mutation"
    ) {
      flushPendingInferences();
      beginOrSwitchGroup(
        event.event.actor,
        event.event.assisting_profile,
      );
      currentControllerActions.push(event);
    } else {
      flushPendingInferences();
      flush();
      grouped.push(event);
    }
  }

  flush();
  if (pendingInferenceEvents.length > 0 && pendingInferenceHasOutput) {
    grouped.push(...pendingInferenceEvents);
  }

  return grouped;
}

function chatItemTimestamp(
  item: EventEnvelope | ActionGroup,
): number | undefined {
  if (isActionGroup(item)) {
    const timestamps = [
      ...item.inferenceEvents,
      ...item.toolCalls,
      ...item.controllerActions,
    ].flatMap((envelope) =>
      "timestamp_ms" in envelope.event && envelope.event.timestamp_ms
        ? [envelope.event.timestamp_ms]
        : []
    );
    return timestamps.length > 0 ? Math.min(...timestamps) : undefined;
  }
  return "timestamp_ms" in item.event ? item.event.timestamp_ms : undefined;
}

function chatItemSpeakerKey(
  item: EventEnvelope | ActionGroup,
): string | undefined {
  if (isActionGroup(item)) {
    return teamActorKey(item.actor);
  }

  const event = item.event;
  switch (event.type) {
    case "started":
    case "user_answer":
    case "user_message":
      return "user:local";
    case "reasoning":
    case "final":
    case "user_question":
      return `agent:${event.profile}`;
    case "correction":
      return teamActorKey(event.actor);
    case "workflow_blocked": {
      const speaker = item.chatter.find((entry) => entry.audience === "team")
        ?.actor;
      return speaker ? teamActorKey(speaker) : undefined;
    }
    case "team_message":
      return teamActorKey(event.actor);
    case "workflow_artifact_accepted":
      return event.artifact_kind === "plan" ? "agent:plan" : undefined;
    case "session_summary":
      return "system:delivery";
    case "tool_call":
    case "tool_result":
      return teamActorKey(event.actor);
    default:
      return undefined;
  }
}

export function buildChatPresentation(
  items: (EventEnvelope | ActionGroup)[],
): ChatPresentationItem[] {
  let lastSpeakerKey: string | undefined;
  let lastTimestampMs: number | undefined;

  return items.map((item) => {
    const speakerKey = chatItemSpeakerKey(item);
    const timestampMs = chatItemTimestamp(item);
    const presentation = {
      item,
      showIdentity: speakerKey === undefined || speakerKey !== lastSpeakerKey,
      timeDividerMs:
        timestampMs !== undefined && lastTimestampMs !== undefined &&
          timestampMs - lastTimestampMs >= CHAT_TIME_GAP_MS
          ? timestampMs
          : undefined,
    };

    if (speakerKey !== undefined) lastSpeakerKey = speakerKey;
    if (timestampMs !== undefined) lastTimestampMs = timestampMs;
    return presentation;
  });
}

export function projectSettingsPath(projectId: string): string {
  return `/projects/${encodeURIComponent(projectId)}/settings`;
}

export function notificationSupport(): boolean {
  return typeof window !== "undefined" && "Notification" in window;
}

export async function ensureNotificationPermission(): Promise<boolean> {
  if (!notificationSupport()) return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission !== "default") return false;
  return (await Notification.requestPermission()) === "granted";
}

export async function notifySessionFinished(
  transition: ProjectSessionTerminalTransition,
) {
  const project = transition.project;
  if (!project?.notify_on_finish) return;
  if (!(await ensureNotificationPermission())) return;
  const title = handoffNotificationTitle(
    transition.handoff_outcome,
    transition.status,
  );
  const body = `${project.name}: ${sessionTitle(transition)}`;
  const url = `/sessions/${transition.session_id}`;
  const registration = await navigator.serviceWorker?.getRegistration?.();
  if (registration?.showNotification) {
    await registration.showNotification(title, {
      body,
      icon: "/apple-touch-icon.png",
      badge: "/apple-touch-icon.png",
      data: { url },
      tag: `pb-${transition.session_id}-${transition.entry_key}`,
    });
    return;
  }
  const notification = new Notification(title, {
    body,
    icon: "/apple-touch-icon.png",
  });
  notification.onclick = () => {
    window.focus();
    window.location.href = url;
  };
}

export function handoffNotificationTitle(
  outcome: HandoffOutcome | null | undefined,
  status: SessionItem["status"],
): string {
  switch (outcome) {
    case "ready":
      return "The team wrapped this up";
    case "no_change":
      return "The team left the code untouched";
    case "checks_failed":
    case "repair_exhausted":
      return "This needs another pass";
    case "executor_unavailable":
    case "commit_blocked":
      return "The team needs help to continue";
    case "incomplete":
      return "The task stopped before handoff";
    default:
      return status === "completed"
        ? "The team wrapped this up"
        : "The task stopped before handoff";
  }
}

export function sessionBelongsToProject(
  session: SessionItem,
  project: ProjectEntry,
): boolean {
  return session.project?.id === project.id;
}

export function sessionProjectName(session: SessionItem): string {
  return session.project?.name ?? "Unknown project";
}

export function usageStatsForToday(
  sessions: SessionItem[],
  now = new Date(),
): ProjectUsageStats {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const startMs = startOfToday.getTime();
  const endMs = startMs + 86_400_000;

  const totals: ProjectUsageStats = { tokens: 0, runtime_ms: 0, tool_calls: 0 };
  sessions.forEach((session) => {
    if (!session.metrics) return;
    session.usage_records.forEach((metrics) => {
      const runtime = metricRuntimeMs(metrics);
      const endedAt = metrics.ended_at_ms;
      const startedAt = metrics.started_at_ms;
      const interval = Math.max(1, endedAt - startedAt);
      const overlap = Math.max(
        0,
        Math.min(endedAt, endMs) - Math.max(startedAt, startMs),
      );
      if (overlap <= 0) return;
      const share = Math.min(1, overlap / interval);
      totals.tokens += (metrics.prompt_tokens + metrics.generated_tokens) *
        share;
      totals.runtime_ms += runtime * share;
      totals.tool_calls += metrics.tool_calls * share;
      const energy = metricEnergyJoules(metrics);
      if (energy !== undefined) {
        totals.energy_joules = (totals.energy_joules ?? 0) + energy * share;
      }
    });
  });
  totals.tokens = Math.round(totals.tokens);
  totals.runtime_ms = Math.round(totals.runtime_ms);
  totals.tool_calls = Math.round(totals.tool_calls);
  if (totals.energy_joules !== undefined) {
    totals.energy_kwh = (totals.energy_joules ?? 0) / 3_600_000;
  }
  return totals;
}

export function relativeTime(ms: number): string {
  const diff = Date.now() - ms;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(ms).toLocaleDateString();
}

export function formatEventTime(timestamp_ms?: number): string {
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

export function formatTranscriptTime(timestampMs?: number): string {
  if (!timestampMs) return "";
  return new Date(timestampMs).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatStartTime(timestamp: number): string {
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
