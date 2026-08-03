import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ProjectEntry,
  ProjectSessionSnapshot,
  ProjectSessionTerminalTransition,
  ProjectUsageSummary,
  SessionItem,
} from "../types";
import { notifySessionFinished } from "./helpers";
import { apiErrorMessage } from "./integrationConfig";
import { parseProjectSessionSnapshotJson } from "./eventContract";

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export class LatestRequest {
  private controller?: AbortController;

  start(): AbortController {
    this.controller?.abort();
    this.controller = new AbortController();
    return this.controller;
  }

  owns(controller: AbortController): boolean {
    return this.controller === controller && !controller.signal.aborted;
  }

  abort(): void {
    this.controller?.abort();
    this.controller = undefined;
  }
}

export class LatestSubscription {
  private generation = 0;

  start(): number {
    this.generation += 1;
    return this.generation;
  }

  owns(generation: number): boolean {
    return this.generation === generation;
  }

  close(generation: number): void {
    if (this.owns(generation)) this.generation += 1;
  }
}

export type ProjectSessionSnapshotSource = "http" | "stream";

export interface ProjectSessionSnapshotDecision {
  applyData: boolean;
  terminalTransitions: ProjectSessionTerminalTransition[];
}

export interface UsageWindow {
  start_ms: number;
  end_ms: number;
}

export function projectSnapshotMatchesUsageWindow(
  snapshot: ProjectSessionSnapshot,
  window: UsageWindow,
): boolean {
  return snapshot.usage_window_start_ms === window.start_ms &&
    snapshot.usage_window_end_ms === window.end_ms;
}

export function projectSnapshotApplicationScope(
  snapshot: ProjectSessionSnapshot,
  window: UsageWindow,
  allowCollectionAcrossUsageWindow: boolean,
): "reject" | "collection" | "full" {
  if (projectSnapshotMatchesUsageWindow(snapshot, window)) return "full";
  return allowCollectionAcrossUsageWindow ? "collection" : "reject";
}

export function currentUsageWindow(now = new Date()): UsageWindow {
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  const end = new Date(start);
  end.setDate(end.getDate() + 1);
  return { start_ms: start.getTime(), end_ms: end.getTime() };
}

export function projectSessionUrl(
  path: string,
  window: UsageWindow,
  lastEventId?: string,
): string {
  const query = new URLSearchParams({
    usage_window_start_ms: String(window.start_ms),
    usage_window_end_ms: String(window.end_ms),
  });
  if (lastEventId) query.set("last_event_id", lastEventId);
  return `${path}?${query}`;
}

const MAX_RETIRED_PROJECT_STREAM_IDS = 8;

export class ProjectSessionStreamCursor {
  private eventStreamId: string | null = null;
  private eventRevision = -1;
  private dataStreamId: string | null = null;
  private revision = -1;
  private usageWindowStartMs = -1;
  private readonly retiredStreamIds: string[] = [];

  accept(
    snapshot: ProjectSessionSnapshot,
    source: ProjectSessionSnapshotSource,
  ): ProjectSessionSnapshotDecision | null {
    if (source === "http") {
      if (
        this.eventStreamId !== null &&
        snapshot.stream_id !== this.eventStreamId
      ) return null;
      if (snapshot.stream_id !== this.dataStreamId) {
        this.dataStreamId = snapshot.stream_id;
        this.revision = -1;
      }
    } else if (snapshot.stream_id !== this.eventStreamId) {
      if (this.retiredStreamIds.includes(snapshot.stream_id)) return null;
      if (this.eventStreamId) {
        this.retiredStreamIds.push(this.eventStreamId);
        if (
          this.retiredStreamIds.length > MAX_RETIRED_PROJECT_STREAM_IDS
        ) this.retiredStreamIds.shift();
      }
      this.eventStreamId = snapshot.stream_id;
      this.eventRevision = -1;
      if (snapshot.stream_id !== this.dataStreamId) {
        this.dataStreamId = snapshot.stream_id;
        this.revision = -1;
      }
    }

    const advancesEventCursor = source === "stream" &&
      snapshot.revision > this.eventRevision;
    const terminalTransitions = advancesEventCursor
      ? snapshot.terminal_transitions
      : [];
    if (advancesEventCursor) this.eventRevision = snapshot.revision;
    const usageWindowChanged =
      snapshot.usage_window_start_ms !== this.usageWindowStartMs;
    const applyData = snapshot.stream_id === this.dataStreamId &&
      (snapshot.revision > this.revision ||
        (snapshot.revision === this.revision && usageWindowChanged));
    if (applyData) {
      this.revision = snapshot.revision;
      this.usageWindowStartMs = snapshot.usage_window_start_ms;
    }
    return { applyData, terminalTransitions };
  }

  resumeCursor(): string | undefined {
    return this.eventStreamId !== null && this.eventRevision >= 0
      ? `${this.eventStreamId}:${this.eventRevision}`
      : undefined;
  }
}

const EMPTY_USAGE_SUMMARY: ProjectUsageSummary = {
  total: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
  today: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
};

export function useProjectSessionData(
  { finishNotifications = true }: { finishNotifications?: boolean } = {},
) {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [projectUsage, setProjectUsage] = useState<
    Record<string, ProjectUsageSummary>
  >({});
  const [overallUsage, setOverallUsage] = useState<ProjectUsageSummary>(
    EMPTY_USAGE_SUMMARY,
  );
  const [usageWindow, setUsageWindow] = useState<UsageWindow>(() =>
    currentUsageWindow()
  );
  const [dataLoading, setDataLoading] = useState(true);
  const [dataError, setDataError] = useState("");
  const [hasSnapshot, setHasSnapshot] = useState(false);
  const usageWindowRef = useRef(usageWindow);
  usageWindowRef.current = usageWindow;
  const dataRequest = useRef(new LatestRequest());
  const streamSubscription = useRef(new LatestSubscription());
  const streamCursor = useRef(new ProjectSessionStreamCursor());

  const applySnapshot = useCallback((
    text: string,
    source: ProjectSessionSnapshotSource,
    allowCollectionAcrossUsageWindow = false,
  ) => {
    const snapshot = parseProjectSessionSnapshotJson(text);
    const scope = projectSnapshotApplicationScope(
      snapshot,
      usageWindowRef.current,
      allowCollectionAcrossUsageWindow,
    );
    if (scope === "reject") return;
    const decision = streamCursor.current.accept(snapshot, source);
    if (!decision) return;
    if (finishNotifications) {
      for (const transition of decision.terminalTransitions) {
        void notifySessionFinished(transition).catch((error) => {
          console.error(
            "Could not show the session finish notification",
            error,
          );
        });
      }
    }
    if (!decision.applyData) return;
    setDataError(snapshot.warnings.join(" "));
    setDataLoading(false);
    setHasSnapshot(true);
    setProjects(snapshot.projects);
    setSessions(snapshot.sessions);
    if (scope === "full") {
      setOverallUsage(snapshot.overall_usage);
      setProjectUsage(snapshot.project_usage);
    }
  }, [finishNotifications]);

  const applyServerSnapshot = useCallback((text: string) => {
    // A project mutation receipt remains authoritative for its revisioned collection if the local
    // day rolls over while the request is in flight. Its old-window aggregates are not applied;
    // the already-open stream supplies the new window without making mutation success depend on it.
    applySnapshot(text, "http", true);
  }, [applySnapshot]);

  const fetchData = useCallback(async () => {
    const controller = dataRequest.current.start();
    try {
      const res = await fetch(
        projectSessionUrl("/api/project-sessions", usageWindow),
        {
          signal: controller.signal,
        },
      );
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Project data request failed"),
        );
      }
      const snapshot = await res.text();
      if (!dataRequest.current.owns(controller)) return;
      applySnapshot(snapshot, "http");
    } catch (error) {
      if (isAbortError(error) || !dataRequest.current.owns(controller)) {
        return;
      }
      const message = error instanceof Error
        ? error.message
        : "Project data request failed";
      setDataError(message);
    } finally {
      if (dataRequest.current.owns(controller)) {
        setDataLoading(false);
      }
    }
  }, [applySnapshot, usageWindow]);

  useEffect(() => {
    const timer = globalThis.setTimeout(() => {
      setUsageWindow(currentUsageWindow());
    }, Math.max(1_000, usageWindow.end_ms - Date.now() + 100));
    return () => globalThis.clearTimeout(timer);
  }, [usageWindow.end_ms]);

  useEffect(() => {
    const generation = streamSubscription.current.start();
    const source = new EventSource(
      projectSessionUrl(
        "/api/project-sessions/events",
        usageWindow,
        streamCursor.current.resumeCursor(),
      ),
    );
    source.addEventListener("project_session_snapshot", (message) => {
      if (!streamSubscription.current.owns(generation)) return;
      try {
        applySnapshot((message as MessageEvent<string>).data, "stream");
      } catch (error) {
        setDataError(
          error instanceof Error ? error.message : "Project update was invalid",
        );
        setDataLoading(false);
      }
    });
    source.onerror = () => {
      if (!streamSubscription.current.owns(generation)) return;
      setDataError("Live project updates are temporarily unavailable");
      setDataLoading(false);
    };
    return () => {
      streamSubscription.current.close(generation);
      source.close();
      dataRequest.current.abort();
    };
  }, [applySnapshot, usageWindow]);

  return {
    sessions,
    projects,
    overallUsage,
    projectUsage,
    dataLoading,
    dataError,
    hasSnapshot,
    usageWindow,
    applyServerSnapshot,
    refresh: fetchData,
  };
}
