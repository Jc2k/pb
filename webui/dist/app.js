(() => {
  const { useMemo, useState } = React;

  function App() {
    const [sessionId, setSessionId] = useState("");
    const [task, setTask] = useState("");
    const [events, setEvents] = useState([]);
    const [followUp, setFollowUp] = useState("");

    const status = useMemo(() => {
      const last = events[events.length - 1];
      return last?.event?.type || "idle";
    }, [events]);

    const openEvents = (id) => {
      const source = new EventSource(`/api/sessions/${id}/events`);
      source.onmessage = (message) => {
        try {
          const parsed = JSON.parse(message.data);
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
      const data = await res.json();
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

    return React.createElement(
      "div",
      null,
      React.createElement(
        "div",
        { className: "row g-3 mb-3" },
        React.createElement(
          "div",
          { className: "col-12" },
          React.createElement("h3", { className: "mb-1" }, "pb serve"),
          React.createElement(
            "div",
            { className: "text-body-secondary" },
            `Schema v1 · status: ${status} · session: ${sessionId || "none"}`
          )
        ),
        React.createElement(
          "div",
          { className: "col-md-9" },
          React.createElement("input", {
            className: "form-control",
            value: task,
            onChange: (e) => setTask(e.target.value),
            placeholder: "Enter a task",
          })
        ),
        React.createElement(
          "div",
          { className: "col-md-3 d-grid" },
          React.createElement(
            "button",
            { className: "btn btn-primary", onClick: start },
            "Start"
          )
        )
      ),
      React.createElement(
        "div",
        { className: "main-grid" },
        React.createElement(
          "div",
          null,
          React.createElement("h5", null, "Live timeline"),
          React.createElement(
            "div",
            { className: "log-box mono" },
            events
              .map((item) => JSON.stringify(item.event, null, 2))
              .join("\n\n") || "No events yet"
          )
        ),
        React.createElement(
          "div",
          null,
          React.createElement("h5", null, "Session controls"),
          React.createElement(
            "div",
            { className: "card" },
            React.createElement(
              "div",
              { className: "card-body" },
              React.createElement("p", { className: "mb-2" }, "Continue current session"),
              React.createElement("textarea", {
                className: "form-control mb-2",
                value: followUp,
                onChange: (e) => setFollowUp(e.target.value),
                placeholder: "Follow-up prompt",
              }),
              React.createElement(
                "button",
                { className: "btn btn-outline-primary", onClick: continueRun },
                "Continue"
              )
            )
          )
        )
      )
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
})();
