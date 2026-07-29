import { useVoiceInput } from "../lib/useVoiceInput";

interface VoiceInputButtonProps {
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
  onActiveChange?: (active: boolean) => void;
}

export function VoiceInputButton({
  value,
  onValueChange,
  disabled = false,
  onActiveChange,
}: VoiceInputButtonProps) {
  const { supported, active, state, error, toggle } = useVoiceInput({
    value,
    onValueChange,
    disabled,
    onActiveChange,
  });

  if (!supported) return null;

  const label = active ? "Stop voice input" : "Start voice input";
  return (
    <div className="voice-input-control">
      <button
        className={`btn voice-input-button${active ? " is-listening" : ""}`}
        type="button"
        onClick={toggle}
        disabled={disabled}
        aria-label={label}
        aria-pressed={active}
        title={active ? "Finish dictating" : "Speak instead of typing"}
      >
        <i
          className={active ? "bi bi-stop-fill" : "bi bi-mic-fill"}
          aria-hidden="true"
        >
        </i>
      </button>
      <span className="visually-hidden" aria-live="polite">
        {state === "starting"
          ? "Starting voice input."
          : state === "listening"
          ? "Listening. Your words are appearing in the prompt. Press stop when finished."
          : "Voice input is off. The transcript is ready to edit or submit."}
      </span>
      {error
        ? (
          <span className="voice-input-error" role="alert">
            {error}
          </span>
        )
        : null}
    </div>
  );
}
