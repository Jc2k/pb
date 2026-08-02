import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ProjectEntry,
  ProjectSessionSnapshot,
  ProjectSessionTerminalTransition,
  ProjectUsageStats,
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

export type ProjectSessionSnapshotSource = "http" | "stream";

export interface ProjectSessionSnapshotDecision {
  applyData: boolean;
  terminalTransitions: ProjectSessionTerminalTransition[];
}

export class ProjectSessionStreamCursor {
  private eventStreamId: string | null = null;
  private dataStreamId: string | null = null;
  private revision = -1;
  private readonly retiredStreamIds = new Set<string>();
  private readonly seenTerminalEntries = new Set<string>();

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
      if (this.retiredStreamIds.has(snapshot.stream_id)) return null;
      if (this.eventStreamId) this.retiredStreamIds.add(this.eventStreamId);
      this.eventStreamId = snapshot.stream_id;
      if (snapshot.stream_id !== this.dataStreamId) {
        this.dataStreamId = snapshot.stream_id;
        this.revision = -1;
      }
      this.seenTerminalEntries.clear();
    }

    const terminalTransitions = source === "stream"
      ? snapshot.terminal_transitions.filter((transition) => {
        if (
          transition.revision <= snapshot.terminal_transition_floor ||
          this.seenTerminalEntries.has(transition.entry_key)
        ) return false;
        this.seenTerminalEntries.add(transition.entry_key);
        return true;
      })
      : [];
    const applyData = snapshot.stream_id === this.dataStreamId &&
      snapshot.revision > this.revision;
    if (applyData) this.revision = snapshot.revision;
    return { applyData, terminalTransitions };
  }
}

export function useProjectSessionData(
  { finishNotifications = true }: { finishNotifications?: boolean } = {},
) {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [projectUsage, setProjectUsage] = useState<
    Record<string, ProjectUsageStats>
  >({});
  const [dataLoading, setDataLoading] = useState(true);
  const [dataError, setDataError] = useState("");
  const dataRequest = useRef(new LatestRequest());
  const streamCursor = useRef(new ProjectSessionStreamCursor());

  const applySnapshot = useCallback((
    text: string,
    source: ProjectSessionSnapshotSource,
  ) => {
    const snapshot = parseProjectSessionSnapshotJson(text);
    const decision = streamCursor.current.accept(snapshot, source);
    if (!decision) return;
    setDataError("");
    setDataLoading(false);
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
    setProjects(snapshot.projects);
    setSessions(snapshot.sessions);
    setProjectUsage(snapshot.project_usage);
  }, [finishNotifications]);

  const fetchData = useCallback(async () => {
    const controller = dataRequest.current.start();
    try {
      const res = await fetch("/api/project-sessions", {
        signal: controller.signal,
      });
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
  }, [applySnapshot]);

  useEffect(() => {
    const source = new EventSource("/api/project-sessions/events");
    source.addEventListener("project_session_snapshot", (message) => {
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
      setDataError("Live project updates are temporarily unavailable");
      setDataLoading(false);
    };
    return () => {
      source.close();
      dataRequest.current.abort();
    };
  }, [applySnapshot]);

  return {
    sessions,
    projects,
    projectUsage,
    dataLoading,
    dataError,
    refresh: fetchData,
  };
}
