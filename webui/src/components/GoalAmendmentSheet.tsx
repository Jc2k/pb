import { useState } from "react";
import type {
  GoalBudget,
  GoalCheckpoint,
  GoalContinuationPolicy,
  GoalCriterionInput,
  SessionStreamSnapshot,
} from "../types";
import { VoiceInputButton } from "./VoiceInputButton";
import { apiErrorMessage } from "../lib/integrationConfig";
import { parseSessionStreamSnapshotJson } from "../lib/eventContract";

interface Props {
  goal: GoalCheckpoint;
  onClose: () => void;
  onSubmitted: () => void;
  onSessionUpdated: (snapshot: SessionStreamSnapshot) => void;
}

export function GoalAmendmentSheet({
  goal,
  onClose,
  onSubmitted,
  onSessionUpdated,
}: Props) {
  const initialDraft = goal.run.stage === "awaiting_plan_approval" &&
    !goal.run.pending_amendment;
  const [objective, setObjective] = useState(goal.run.objective);
  const [criteria, setCriteria] = useState<GoalCriterionInput[]>(
    goal.run.criteria.map(({ text, verifier }) => ({ text, verifier })),
  );
  const [continuation, setContinuation] = useState<GoalContinuationPolicy>(
    goal.run.continuation,
  );
  const [budget, setBudget] = useState<GoalBudget>(goal.run.budget);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [voiceInputActive, setVoiceInputActive] = useState(false);
  const submit = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(
        `/api/goals/${goal.run.id}/${initialDraft ? "draft" : "amendments"}`,
        {
          method: initialDraft ? "PATCH" : "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            goal_sha256: goal.sha256,
            objective,
            criteria,
            continuation,
            budget,
          }),
        },
      );
      if (!response.ok) {
        setError(await apiErrorMessage(response, "Could not update the goal"));
        return;
      }
      onSessionUpdated(
        parseSessionStreamSnapshotJson(await response.text()),
      );
      onSubmitted();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not update the goal",
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="goal-sheet-backdrop" onMouseDown={onClose}>
      <section
        className="goal-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="goal-amend-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="goal-sheet-header">
          <div>
            <span className="goal-eyebrow">
              {initialDraft ? "Goal draft" : "Checkpointed amendment"}
            </span>
            <h2 id="goal-amend-title">
              {initialDraft
                ? "Edit goal before approval"
                : "Edit remaining goal"}
            </h2>
          </div>
          <button
            className="btn btn-light btn-icon"
            type="button"
            onClick={onClose}
            aria-label="Close goal editor"
          >
            <i className="bi bi-x-lg"></i>
          </button>
        </header>
        <p className="small text-body-secondary">
          {initialDraft
            ? "Updating the draft generates a new plan digest. No repository work starts until you approve it."
            : "Completed milestones and their evidence remain unchanged. pb will review a new plan for unfinished work."}
        </p>
        {error
          ? <p className="text-danger small" role="alert">{error}</p>
          : null}
        <div className="goal-field">
          <label htmlFor="goal-amend-objective">Objective</label>
          <div className="voice-prompt-field">
            <textarea
              id="goal-amend-objective"
              className="form-control"
              rows={3}
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              readOnly={voiceInputActive}
            />
            <VoiceInputButton
              value={objective}
              onValueChange={setObjective}
              disabled={busy}
              onActiveChange={setVoiceInputActive}
            />
          </div>
        </div>
        <fieldset className="goal-field">
          <legend>Done when</legend>
          {criteria.map((criterion, index) => (
            <div className="goal-criterion-row" key={index}>
              <input
                className="form-control"
                value={criterion.text}
                onChange={(event) =>
                  setCriteria((current) =>
                    current.map((value, position) =>
                      position === index
                        ? { ...value, text: event.target.value }
                        : value
                    )
                  )}
              />
              <button
                className="btn btn-light btn-icon"
                type="button"
                onClick={() =>
                  setCriteria((current) =>
                    current.filter((_, position) =>
                      position !== index
                    )
                  )}
                aria-label={`Remove criterion ${index + 1}`}
              >
                <i className="bi bi-trash"></i>
              </button>
            </div>
          ))}
          <button
            className="btn btn-sm btn-light align-self-start"
            type="button"
            onClick={() => setCriteria((
              current,
            ) => [...current, { text: "", verifier: "review_required" }])}
          >
            <i className="bi bi-plus-lg"></i> Add criterion
          </button>
        </fieldset>
        <label className="goal-field">
          <span>Continuation</span>
          <select
            className="form-select"
            value={continuation}
            onChange={(event) =>
              setContinuation(event.target.value as GoalContinuationPolicy)}
          >
            <option value="review_plan_then_automatic">
              Review plan, then automatic
            </option>
            <option value="manual_milestones">
              Ask before every milestone
            </option>
            <option value="automatic_within_limits">
              Automatic within limits
            </option>
          </select>
        </label>
        <fieldset className="goal-field">
          <legend>Budget</legend>
          <div className="goal-budget-grid">
            {([
              ["max_milestones", "Milestones"],
              ["max_workflows", "Workflow attempts"],
              ["total_model_invocations", "Model invocations"],
              ["total_generated_tokens", "Generated tokens"],
              ["wall_time_minutes", "Wall time (minutes)"],
            ] as const).map(([field, label]) => (
              <label key={field}>
                <span>{label}</span>
                <input
                  className="form-control"
                  type="number"
                  min={1}
                  value={budget[field]}
                  onChange={(event) =>
                    setBudget((current) => ({
                      ...current,
                      [field]: Math.max(1, Number(event.target.value) || 1),
                    }))}
                />
              </label>
            ))}
          </div>
          <small className="text-body-secondary">
            Project ceilings still apply. An amendment cannot grant publishing
            authority.
          </small>
        </fieldset>
        <footer className="goal-sheet-actions">
          <button className="btn btn-light" type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            type="button"
            disabled={busy || voiceInputActive || !objective.trim() ||
              criteria.some((criterion) => !criterion.text.trim())}
            onClick={() => void submit()}
          >
            {busy
              ? "Reviewing…"
              : initialDraft
              ? "Update plan"
              : "Review changes"}
          </button>
        </footer>
      </section>
    </div>
  );
}
