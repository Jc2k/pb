import type { TurnIntent } from "../types";

interface IntentControlProps {
  intent: Exclude<TurnIntent, "auto">;
  onChange: (intent: Exclude<TurnIntent, "auto">) => void;
  disabled?: boolean;
}

export function IntentControl({ intent, onChange, disabled = false }: IntentControlProps) {
  return (
    <div className="intent-control" role="group" aria-label="Session intent">
      <button
        type="button"
        className={`intent-option ${intent === "discuss" ? "active" : ""}`}
        aria-pressed={intent === "discuss"}
        disabled={disabled}
        onClick={() => onChange("discuss")}
      >
        <i className="bi bi-chat-dots"></i> Discuss
      </button>
      <button
        type="button"
        className={`intent-option ${intent === "deliver" ? "active" : ""}`}
        aria-pressed={intent === "deliver"}
        disabled={disabled}
        onClick={() => onChange("deliver")}
      >
        <i className="bi bi-hammer"></i> Build
      </button>
    </div>
  );
}
