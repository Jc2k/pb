import { useCallback, useEffect, useRef, useState } from "react";
import type { ProjectEntry, SessionItem, SessionStatus } from "../types";
import { notifySessionFinished } from "./helpers";

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
  const sessionsRequest = useRef(new LatestRequest());
  const projectsRequest = useRef(new LatestRequest());

  const fetchSessions = useCallback(async () => {
    const controller = sessionsRequest.current.start();
    try {
      const res = await fetch("/api/sessions", { signal: controller.signal });
      if (!res.ok) {
        throw new Error(`Session request failed (${res.status})`);
      }
      const nextSessions = (await res.json()) as SessionItem[];
      if (!sessionsRequest.current.owns(controller)) return;
      setSessions(nextSessions);
      setSessionsError("");
    } catch (error) {
      if (isAbortError(error) || !sessionsRequest.current.owns(controller)) {
        return;
      }
      setSessionsError(
        error instanceof Error ? error.message : "Session request failed",
      );
    } finally {
      if (sessionsRequest.current.owns(controller)) setSessionsLoading(false);
    }
  }, []);

  const fetchProjects = useCallback(async () => {
    const controller = projectsRequest.current.start();
    try {
      const res = await fetch("/api/projects", { signal: controller.signal });
      if (!res.ok) {
        throw new Error(`Project request failed (${res.status})`);
      }
      const nextProjects = (await res.json()) as ProjectEntry[];
      if (!projectsRequest.current.owns(controller)) return;
      setProjects(nextProjects);
      setProjectsError("");
    } catch (error) {
      if (isAbortError(error) || !projectsRequest.current.owns(controller)) {
        return;
      }
      setProjectsError(
        error instanceof Error ? error.message : "Project request failed",
      );
    } finally {
      if (projectsRequest.current.owns(controller)) setProjectsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchProjects();
    void fetchSessions();
    const timer = window.setInterval(() => void fetchSessions(), pollMs);
    return () => {
      window.clearInterval(timer);
      sessionsRequest.current.abort();
      projectsRequest.current.abort();
    };
  }, [fetchProjects, fetchSessions, pollMs]);

  useProjectFinishNotifications(sessions, projects);

  return {
    sessions,
    projects,
    sessionsLoading,
    projectsLoading,
    sessionsError,
    projectsError,
    setProjects,
    refreshProjects: fetchProjects,
    refreshSessions: fetchSessions,
  };
}
