import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ProjectEntry,
  ProjectSessionSnapshot,
  SessionItem,
  SessionStatus,
} from "../types";
import { notifySessionFinished } from "./helpers";
import { apiErrorMessage } from "./integrationConfig";

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

export function useProjectFinishNotifications(
  sessions: SessionItem[],
  projects: ProjectEntry[],
) {
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

export function useProjectSessionData(pollMs = 5000) {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [sessionsError, setSessionsError] = useState("");
  const [projectsError, setProjectsError] = useState("");
  const dataRequest = useRef(new LatestRequest());

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
      const snapshot = (await res.json()) as ProjectSessionSnapshot;
      if (!dataRequest.current.owns(controller)) return;
      setProjects(snapshot.projects);
      setSessions(snapshot.sessions);
      setProjectsError("");
      setSessionsError("");
    } catch (error) {
      if (isAbortError(error) || !dataRequest.current.owns(controller)) {
        return;
      }
      const message = error instanceof Error
        ? error.message
        : "Project data request failed";
      setProjectsError(message);
      setSessionsError(message);
    } finally {
      if (dataRequest.current.owns(controller)) {
        setProjectsLoading(false);
        setSessionsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void fetchData();
    const timer = window.setInterval(() => void fetchData(), pollMs);
    return () => {
      window.clearInterval(timer);
      dataRequest.current.abort();
    };
  }, [fetchData, pollMs]);

  useProjectFinishNotifications(sessions, projects);

  return {
    sessions,
    projects,
    sessionsLoading,
    projectsLoading,
    sessionsError,
    projectsError,
    refreshProjects: fetchData,
    refreshSessions: fetchData,
  };
}
