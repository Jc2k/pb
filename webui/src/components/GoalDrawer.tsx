import type { GoalCheckpoint } from "../types";
import { goalPercent, goalProgress, goalStageLabel } from "../lib/goalUtils";

export function GoalDrawer({ goal }: { goal: GoalCheckpoint }) {
  const { run } = goal;
  const progress = goalProgress(run);
  const activeWorkflow = run.milestones.find((milestone) =>
    milestone.id === run.active_milestone_id
  )?.workflow?.run.counters;
  const activeWorkflowCount = run.active_milestone_id ? 1 : 0;
  const generatedTokens = run.counters.generated_tokens +
    (activeWorkflow?.generated_tokens ?? 0);
  const modelInvocations = run.counters.model_invocations +
    (activeWorkflow?.model_invocations ?? 0);
  const workflows = run.counters.workflows + activeWorkflowCount;
  const wallMinutes = Math.min(
    run.budget.wall_time_minutes,
    Math.max(0, Math.floor((Date.now() - run.created_at_ms) / 60_000)),
  );
  const tokenPercent = goalPercent(
    generatedTokens,
    run.budget.total_generated_tokens,
  );
  return (
    <div className="goal-drawer-content">
      <div className="goal-overview-grid">
        <div>
          <span>Status</span>
          <strong>{goalStageLabel(run.stage, run.pause_requested)}</strong>
        </div>
        <div>
          <span>Milestones</span>
          <strong>{progress.completed} of {progress.total}</strong>
        </div>
      </div>
      <p className="goal-objective">{run.objective}</p>
      {run.blocked_reason
        ? (
          <p className="goal-blocker">
            <i className="bi bi-exclamation-triangle"></i> {run.blocked_reason}
          </p>
        )
        : null}
      <h3>Milestones</h3>
      <ol className="goal-milestone-list">
        {run.milestones.map((milestone) => (
          <li
            key={milestone.id}
            className={`goal-milestone-${milestone.status}`}
          >
            <i
              className={`bi ${
                milestone.status === "completed"
                  ? "bi-check-circle-fill"
                  : milestone.status === "running"
                  ? "bi-play-circle-fill"
                  : milestone.status === "superseded"
                  ? "bi-dash-circle"
                  : "bi-circle"
              }`}
            >
            </i>
            <div>
              <strong>{milestone.title}</strong>
              <span>
                {milestone.status.replaceAll("_", " ")}
                {milestone.attempts
                  ? ` · ${milestone.attempts} attempt${
                    milestone.attempts === 1 ? "" : "s"
                  }`
                  : ""}
              </span>
            </div>
          </li>
        ))}
      </ol>
      <h3>Done when</h3>
      <ul className="goal-criteria-list">
        {run.criteria.map((criterion) => (
          <li key={criterion.id}>
            <i
              className={`bi ${
                criterion.status === "machine_verified" ||
                  criterion.status === "user_accepted"
                  ? "bi-patch-check-fill"
                  : criterion.status === "evidence_ready"
                  ? "bi-eye-fill"
                  : "bi-circle"
              }`}
            >
            </i>
            <span>
              {criterion.text}
              <small>{criterion.status.replaceAll("_", " ")}</small>
            </span>
          </li>
        ))}
      </ul>
      {run.retired_criteria?.length
        ? (
          <details className="goal-prior-criteria">
            <summary>
              Prior criteria and evidence ({run.retired_criteria.length})
            </summary>
            <ul className="goal-criteria-list">
              {run.retired_criteria.map((criterion) => (
                <li key={criterion.id}>
                  <i className="bi bi-archive"></i>
                  <span>
                    {criterion.text}
                    <small>
                      {criterion.status.replaceAll("_", " ")} · preserved
                    </small>
                  </span>
                </li>
              ))}
            </ul>
          </details>
        )
        : null}
      <h3>Budget</h3>
      <div className="goal-budget-row">
        <span>Generated tokens</span>
        <strong>
          {generatedTokens.toLocaleString()} /{" "}
          {run.budget.total_generated_tokens.toLocaleString()}
        </strong>
      </div>
      <div
        className="progress"
        role="progressbar"
        aria-label="Goal generated-token budget"
        aria-valuenow={tokenPercent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="progress-bar" style={{ width: `${tokenPercent}%` }}>
        </div>
      </div>
      <div className="goal-budget-row">
        <span>Strict workflows</span>
        <strong>{workflows} / {run.budget.max_workflows}</strong>
      </div>
      <div className="goal-budget-row">
        <span>Model invocations</span>
        <strong>
          {modelInvocations} / {run.budget.total_model_invocations}
        </strong>
      </div>
      <div className="goal-budget-row">
        <span>Milestone capacity</span>
        <strong>{progress.total} / {run.budget.max_milestones}</strong>
      </div>
      <div className="goal-budget-row">
        <span>Elapsed wall time</span>
        <strong>{wallMinutes} / {run.budget.wall_time_minutes} min</strong>
      </div>
      <p className="goal-authority-note">
        <i className="bi bi-shield-lock"></i> Local work only · no publishing
      </p>
    </div>
  );
}
