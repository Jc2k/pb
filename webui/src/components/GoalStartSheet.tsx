import { useEffect, useState } from "react";
import type {
  GoalBudget,
  GoalContinuationPolicy,
  ProjectEntry,
} from "../types";
import { apiErrorMessage } from "../lib/integrationConfig";
import { VoiceInputButton } from "./VoiceInputButton";

interface GoalStartSheetProps {
  open: boolean;
  initialObjective: string;
  initialCriteria?: string[];
  sessionId?: string;
  projectName?: string;
  projects?: ProjectEntry[];
  onClose: () => void;
  onStarted: (sessionId: string) => void;
}

const EMPTY_PROJECTS: ProjectEntry[] = [];

type BudgetPreset = "compact" | "standard" | "extended" | "advanced";

const GOAL_BUDGET_PRESETS: Record<
  Exclude<BudgetPreset, "advanced">,
  GoalBudget
> = {
  compact: {
    max_milestones: 3,
    max_workflows: 5,
    total_model_invocations: 40,
    total_generated_tokens: 30_000,
    wall_time_minutes: 45,
  },
  standard: {
    max_milestones: 5,
    max_workflows: 8,
    total_model_invocations: 80,
    total_generated_tokens: 60_000,
    wall_time_minutes: 90,
  },
  extended: {
    max_milestones: 8,
    max_workflows: 12,
    total_model_invocations: 120,
    total_generated_tokens: 100_000,
    wall_time_minutes: 120,
  },
};

export function GoalStartSheet({
  open,
  initialObjective,
  initialCriteria,
  sessionId,
  projectName,
  projects = EMPTY_PROJECTS,
  onClose,
  onStarted,
}: GoalStartSheetProps) {
  const initialCriteriaKey = JSON.stringify(initialCriteria ?? []);
  const projectOptionsKey = projects.map(({ name }) => name).join("\u0000");
  const [objective, setObjective] = useState(initialObjective);
  const [criteria, setCriteria] = useState<string[]>([""]);
  const [continuation, setContinuation] = useState<GoalContinuationPolicy>(
    "review_plan_then_automatic",
  );
  const [selectedProjectName, setSelectedProjectName] = useState(
    projectName ?? "",
  );
  const [budgetPreset, setBudgetPreset] = useState<BudgetPreset>("standard");
  const [advancedBudget, setAdvancedBudget] = useState<GoalBudget>(
    GOAL_BUDGET_PRESETS.standard,
  );
  const [submitting, setSubmitting] = useState(false);
  const [voiceInputActive, setVoiceInputActive] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setObjective(initialObjective);
    setCriteria(initialCriteria?.length ? initialCriteria : [""]);
    setSelectedProjectName(projectName ?? projects[0]?.name ?? "");
    setBudgetPreset("standard");
    setAdvancedBudget(GOAL_BUDGET_PRESETS.standard);
    setError("");
  }, [
    open,
    initialObjective,
    initialCriteriaKey,
    projectName,
    projectOptionsKey,
  ]);

  if (!open) return null;

  const submit = async () => {
    if (!objective.trim() || (!sessionId && !selectedProjectName)) return;
    setSubmitting(true);
    setError("");
    try {
      const response = await fetch("/api/goals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          objective: objective.trim(),
          project_name: sessionId ? undefined : selectedProjectName,
          continuation,
          budget: budgetPreset === "advanced"
            ? advancedBudget
            : GOAL_BUDGET_PRESETS[budgetPreset],
          criteria: criteria
            .map((criterion) => criterion.trim())
            .filter(Boolean)
            .map((text) => ({ text, verifier: "review_required" })),
        }),
      });
      if (!response.ok) {
        setError(
          await apiErrorMessage(response, "pb could not create this goal."),
        );
        return;
      }
      const result = (await response.json()) as { session_id: string };
      onStarted(result.session_id);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "pb could not create this goal.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="goal-sheet-backdrop" onMouseDown={onClose}>
      <section
        className="goal-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="goal-sheet-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="goal-sheet-header">
          <div>
            <span className="goal-eyebrow">Durable Goal</span>
            <h2 id="goal-sheet-title">Plan a bounded goal</h2>
          </div>
          <button
            className="btn btn-light btn-icon"
            type="button"
            onClick={onClose}
            aria-label="Close goal setup"
          >
            <i className="bi bi-x-lg"></i>
          </button>
        </header>

        <div className="goal-field">
          <label htmlFor="goal-objective">Objective</label>
          <div className="voice-prompt-field">
            <textarea
              id="goal-objective"
              className="form-control"
              rows={3}
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              readOnly={voiceInputActive}
              autoFocus
            />
            <VoiceInputButton
              value={objective}
              onValueChange={setObjective}
              disabled={submitting}
              onActiveChange={setVoiceInputActive}
            />
          </div>
        </div>

        {!sessionId && !projectName
          ? (
            <label className="goal-field">
              <span>Project</span>
              <select
                className="form-select"
                value={selectedProjectName}
                onChange={(event) => setSelectedProjectName(event.target.value)}
              >
                <option value="">Select a registered project…</option>
                {projects.map((project) => (
                  <option key={project.name} value={project.name}>
                    {project.name}
                  </option>
                ))}
              </select>
            </label>
          )
          : null}

        <fieldset className="goal-field">
          <legend>Done when</legend>
          <p className="small text-body-secondary">
            Each criterion becomes a reviewed, sequential Build milestone. If
            left empty, pb adds a final objective review criterion for plan
            approval.
          </p>
          {criteria.map((criterion, index) => (
            <div className="goal-criterion-row" key={index}>
              <input
                className="form-control"
                value={criterion}
                placeholder={`Criterion ${index + 1}`}
                onChange={(event) =>
                  setCriteria((current) =>
                    current.map((value, position) =>
                      position === index ? event.target.value : value
                    )
                  )}
              />
              <button
                className="btn btn-light btn-icon"
                type="button"
                onClick={() =>
                  setCriteria((current) => {
                    if (index === 0) {
                      return current;
                    }
                    const next = [...current];
                    [next[index - 1], next[index]] = [
                      next[index],
                      next[index - 1],
                    ];
                    return next;
                  })}
                disabled={index === 0}
                aria-label={`Move criterion ${index + 1} up`}
              >
                <i className="bi bi-arrow-up"></i>
              </button>
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
            onClick={() => setCriteria((current) => [...current, ""])}
          >
            <i className="bi bi-plus-lg"></i> Add criterion
          </button>
        </fieldset>

        <fieldset className="goal-field">
          <legend>How to continue</legend>
          <label className="goal-radio">
            <input
              type="radio"
              name="goal-continuation"
              checked={continuation === "review_plan_then_automatic"}
              onChange={() => setContinuation("review_plan_then_automatic")}
            />
            <span>
              <strong>Review the plan, then continue</strong>
              <small>Recommended</small>
            </span>
          </label>
          <label className="goal-radio">
            <input
              type="radio"
              name="goal-continuation"
              checked={continuation === "manual_milestones"}
              onChange={() => setContinuation("manual_milestones")}
            />
            <span>
              <strong>Ask before every milestone</strong>
              <small>Maximum control</small>
            </span>
          </label>
          <label className="goal-radio">
            <input
              type="radio"
              name="goal-continuation"
              checked={continuation === "automatic_within_limits"}
              onChange={() => setContinuation("automatic_within_limits")}
            />
            <span>
              <strong>Continue within limits</strong>
              <small>Still stops for authority or review</small>
            </span>
          </label>
        </fieldset>

        <fieldset className="goal-field">
          <legend>Budget</legend>
          <div
            className="goal-budget-presets"
            role="radiogroup"
            aria-label="Goal budget"
          >
            {(["compact", "standard", "extended"] as const).map((preset) => (
              <label className="goal-budget-preset" key={preset}>
                <input
                  type="radio"
                  name="goal-budget"
                  checked={budgetPreset === preset}
                  onChange={() => setBudgetPreset(preset)}
                />
                <strong>{preset[0].toUpperCase() + preset.slice(1)}</strong>
                <small>
                  {GOAL_BUDGET_PRESETS[preset].max_milestones} milestones ·{" "}
                  {GOAL_BUDGET_PRESETS[preset].wall_time_minutes} min
                </small>
              </label>
            ))}
          </div>
          <details className="goal-budget-advanced">
            <summary>Advanced limits</summary>
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
                    value={(budgetPreset === "advanced"
                      ? advancedBudget
                      : GOAL_BUDGET_PRESETS[budgetPreset])[field]}
                    onChange={(event) => {
                      const base = budgetPreset === "advanced"
                        ? advancedBudget
                        : GOAL_BUDGET_PRESETS[budgetPreset];
                      setAdvancedBudget({
                        ...base,
                        [field]: Math.max(1, Number(event.target.value) || 1),
                      });
                      setBudgetPreset("advanced");
                    }}
                  />
                </label>
              ))}
            </div>
            {budgetPreset === "advanced"
              ? (
                <small>
                  Custom limits are checked against this project's ceiling.
                </small>
              )
              : null}
          </details>
        </fieldset>

        <div className="goal-authority-summary">
          <i className="bi bi-shield-check"></i>
          <span>
            Local repository work only. Strict Build stages and managed commits.
            No publishing.
          </span>
        </div>
        {error
          ? <p className="text-danger small" role="alert">{error}</p>
          : null}
        <footer className="goal-sheet-actions">
          <button className="btn btn-light" type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            type="button"
            disabled={submitting || voiceInputActive || !objective.trim() ||
              (!sessionId && !selectedProjectName)}
            onClick={() => void submit()}
          >
            {submitting ? "Planning…" : "Plan goal"}
          </button>
        </footer>
      </section>
    </div>
  );
}
