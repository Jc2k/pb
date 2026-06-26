import type { EventEnvelope, InstalledIntegration, MarketplaceIntegration, ProjectEntry, ProjectUsageStats, SessionItem } from "../types/index";

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

export function uniqueInstalledIntegrations(entries: InstalledIntegration[]): InstalledIntegration[] {
  const seen = new Set<string>();
  return entries.filter((entry) => {
    const key = `${entry.kind}:${entry.name}:${entry.container_image}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function isIntegrationInstalled(item: MarketplaceIntegration, installed: InstalledIntegration[]): boolean {
  return installed.some((entry) =>
    entry.kind === item.kind && (entry.container_image === item.container_image || entry.name === item.name)
  );
}


export function getAvatarForProfile(profile: string): string {
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

export function sessionTitle(session: Pick<SessionItem, "task" | "title">): string {
  return session.title?.trim() || session.task;
}

export function groupToolEvents(
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


export function projectSettingsPath(projectName: string): string {
  return `/projects/${encodeURIComponent(projectName)}/settings`;
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

export async function notifySessionFinished(session: SessionItem, projects: ProjectEntry[]) {
  if (session.status !== "completed" && session.status !== "failed") return;
  const project = projects.find((entry) => entry.path === session.workdir);
  if (!project?.notify_on_finish) return;
  if (!(await ensureNotificationPermission())) return;
  const title = session.status === "completed" ? "pb session completed" : "pb session failed";
  const body = `${project.name}: ${sessionTitle(session)}`;
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

export function projectName(workdir?: string): string {
  if (!workdir) return "Unknown project";
  const parts = workdir.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts[parts.length - 1] || workdir;
}


export function usageStatsForToday(sessions: SessionItem[], now = new Date()): ProjectUsageStats {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const startMs = startOfToday.getTime();
  const endMs = startMs + 86_400_000;

  return sessions.reduce<ProjectUsageStats>((totals, session) => {
    if (session.updated_at_ms < startMs || session.updated_at_ms >= endMs || !session.metrics) {
      return totals;
    }

    totals.tokens += session.metrics.prompt_tokens + session.metrics.generated_tokens;
    totals.runtime_ms += session.metrics.llm_runtime_ms + session.metrics.tool_runtime_ms;
    totals.tool_calls += session.metrics.tool_calls;

    const energy = (session.metrics.llm_energy_kwh ?? 0) + (session.metrics.tool_energy_kwh ?? 0);
    if (energy > 0) {
      totals.energy_kwh = (totals.energy_kwh ?? 0) + energy;
    }

    return totals;
  }, { tokens: 0, runtime_ms: 0, tool_calls: 0 });
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
