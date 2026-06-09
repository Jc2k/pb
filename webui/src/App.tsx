import { useMemo, useState } from "react";

interface EventEnvelope {
  event: {
    type: string;
    [key: string]: unknown;
  };
}

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [task, setTask] = useState("");
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [followUp, setFollowUp] = useState("");

  const status = useMemo(() => {
    const last = events[events.length - 1];
    return last?.event?.type ?? "idle";
  }, [events]);

  const openEvents = (id: string) => {
    const source = new EventSource(`/api/sessions/${id}/events`);
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
