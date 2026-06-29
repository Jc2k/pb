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

  const fetchSessions = useCallback(async () => {
    const res = await fetch("/api/sessions");
    if (res.ok) setSessions((await res.json()) as SessionItem[]);
  }, []);

  const fetchProjects = useCallback(async () => {
    const res = await fetch("/api/projects");
    if (res.ok) setProjects((await res.json()) as ProjectEntry[]);
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
    setProjects,
    refreshProjects: fetchProjects,
    refreshSessions: fetchSessions,
  };
}
