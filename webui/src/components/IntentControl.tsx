import type { ComposerMode } from "../types";

interface IntentControlProps {
  intent: ComposerMode;
  onChange: (intent: ComposerMode) => void;
  disabled?: boolean;
  activeGoal?: boolean;
}

export function IntentControl({
  intent,
  onChange,
  disabled = false,
  activeGoal = false,
}: IntentControlProps) {
  return (
    <div className="intent-control" role="group" aria-label="Session intent">
      <button
        type="button"
        className={`intent-option ${intent === "discuss" ? "active" : ""}`}
        aria-pressed={intent === "discuss"}
        disabled={disabled}
        onClick={() =>
          onChange("discuss")}
      >
        <i className="bi bi-chat-dots"></i> Discuss
      </button>
      <button
        type="button"
        className={`intent-option ${intent === "deliver" ? "active" : ""}`}
        aria-pressed={intent === "deliver"}
        disabled={disabled}
        onClick={() =>
          onChange("deliver")}
      >
        <i className="bi bi-hammer"></i> Build
      </button>
      <button
        type="button"
        className={`intent-option ${
          intent === "goal" || activeGoal ? "active" : ""
        }`}
        aria-pressed={intent === "goal" || activeGoal}
        disabled={disabled || activeGoal}
        onClick={() =>
          onChange("goal")}
      >
        <i className="bi bi-bullseye"></i> {activeGoal ? "Goal active" : "Goal"}
      </button>
    </div>
  );
}
