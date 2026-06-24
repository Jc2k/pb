import { useEffect, useRef } from "react";
import type { ProjectEntry, SessionItem, SessionStatus } from "../types";
import { notifySessionFinished } from "./helpers";

export function useProjectFinishNotifications(sessions: SessionItem[], projects: ProjectEntry[]) {
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
