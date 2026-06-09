import { useEffect, useMemo, useRef, useState } from "react";

interface EventEnvelope {
  event: {
    type: string;
    [key: string]: unknown;
  };
}

interface SessionItem {
  session_id: string;
  task: string;
  running: boolean;
  branch?: string;
  updated_at_ms: number;
}

interface SessionDetails {
  session_id: string;
  task: string;
  running: boolean;
  branch?: string;
  events: EventEnvelope[];
}

export default function App() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [task, setTask] = useState("");
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [followUp, setFollowUp] = useState("");
  const sourceRef = useRef<EventSource | null>(null);

  const refreshSessions = async () => {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const data = (await res.json()) as SessionItem[];
    setSessions(data);
  };

  const status = useMemo(() => {
    const last = events[events.length - 1];
    if (last?.event?.type) return last.event.type;
    const selected = sessions.find((item) => item.session_id === sessionId);
    if (!selected) return "idle";
    return selected.running ? "running" : "idle";
  }, [events, sessionId, sessions]);

  const openEvents = (id: string) => {
    if (sourceRef.current) {
      sourceRef.current.close();
    }
    const source = new EventSource(`/api/sessions/${id}/events`);
    sourceRef.current = source;
    source.onmessage = (message) => {
      try {
        const parsed = JSON.parse(message.data) as EventEnvelope;
        setEvents((prev) => [...prev, parsed]);
      } catch (error) {
        console.error(error);
      }
    };
    source.onerror = () => {
      source.close();
    };
  };

  const selectSession = async (id: string) => {
    const res = await fetch(`/api/sessions/${id}`);
    if (!res.ok) return;
    const details = (await res.json()) as SessionDetails;
    setSessionId(details.session_id);
    setEvents(details.events);
    openEvents(details.session_id);
  };

  useEffect(() => {
    refreshSessions();
    const timer = window.setInterval(refreshSessions, 2000);
    return () => {
      window.clearInterval(timer);
      if (sourceRef.current) {
        sourceRef.current.close();
      }
    };
  }, []);

  const start = async () => {
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
    });
    if (!res.ok) return;
    const data = (await res.json()) as { session_id: string };
    setSessionId(data.session_id);
    setEvents([]);
    await refreshSessions();
    openEvents(data.session_id);
  };

  const continueRun = async () => {
    if (!sessionId || !followUp) return;
    await fetch(`/api/sessions/${sessionId}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: followUp }),
    });
    setFollowUp("");
    await refreshSessions();
  };

  return (
    <div>
      <div className="row g-3 mb-3">
        <div className="col-12">
          <h3 className="mb-1">pb serve</h3>
          <div className="text-body-secondary">
            {`Schema v1 · status: ${status} · session: ${sessionId || "none"}`}
          </div>
        </div>
        <div className="col-md-9">
          <input
            className="form-control"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Enter a task"
          />
        </div>
        <div className="col-md-3 d-grid">
          <button className="btn btn-primary" onClick={start}>
            Start
          </button>
        </div>
      </div>

      <div className="main-grid">
        <div>
          <h5>Sessions</h5>
          <div className="log-box">
            {sessions.length === 0 && <div>No sessions yet</div>}
            {sessions.map((item) => (
              <button
                key={item.session_id}
                className={`btn btn-sm w-100 text-start mb-2 ${
                  item.session_id === sessionId ? "btn-primary" : "btn-outline-secondary"
                }`}
                onClick={() => selectSession(item.session_id)}
              >
                <div className="fw-semibold">{item.task}</div>
                <div className="small">
                  {item.session_id} · {item.running ? "running" : "idle"}
                </div>
              </button>
            ))}
          </div>
        </div>
        <div>
          <h5>Live timeline</h5>
          <div className="log-box mono">
            {events.map((item) => JSON.stringify(item.event, null, 2)).join("\n\n") ||
              "No events yet"}
          </div>
        </div>
        <div>
          <h5>Session controls</h5>
          <div className="card">
            <div className="card-body">
              <p className="mb-2">Continue current session</p>
              <textarea
                className="form-control mb-2"
                value={followUp}
                onChange={(e) => setFollowUp(e.target.value)}
                placeholder="Follow-up prompt"
              />
              <button className="btn btn-outline-primary" onClick={continueRun}>
                Continue
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
