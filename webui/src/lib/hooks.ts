import { useCallback, useEffect, useRef, useState } from "react";
import type { ProjectEntry, SessionItem, SessionStatus } from "../types";
import { notifySessionFinished } from "./helpers";

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

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("/api/sessions");
      if (!res.ok) {
        throw new Error(`Session request failed (${res.status})`);
      }
      setSessions((await res.json()) as SessionItem[]);
      setSessionsError("");
    } catch (error) {
      setSessionsError(
        error instanceof Error ? error.message : "Session request failed",
      );
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  const fetchProjects = useCallback(async () => {
    try {
      const res = await fetch("/api/projects");
      if (!res.ok) {
        throw new Error(`Project request failed (${res.status})`);
      }
      setProjects((await res.json()) as ProjectEntry[]);
      setProjectsError("");
    } catch (error) {
      setProjectsError(
        error instanceof Error ? error.message : "Project request failed",
      );
    } finally {
      setProjectsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchProjects();
    void fetchSessions();
    const timer = window.setInterval(() => void fetchSessions(), pollMs);
    return () => window.clearInterval(timer);
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
