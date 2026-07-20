import type { GoalCheckpoint } from "../types";
import { goalProgress, goalStageLabel } from "../lib/goalUtils";

interface GoalModeBannerProps {
  goal: GoalCheckpoint;
  onDetails: () => void;
  onPause: () => void;
  onResume: () => void;
  onAccept: () => void;
  onEdit: () => void;
  onStop: () => void;
  busy?: boolean;
}

export function GoalModeBanner({
  goal,
  onDetails,
  onPause,
  onResume,
  onAccept,
  onEdit,
  onStop,
  busy = false,
}: GoalModeBannerProps) {
  const { run } = goal;
  const progress = goalProgress(run);
  const label = goalStageLabel(run.stage, run.pause_requested);
  const paused = run.stage === "paused" || run.stage === "blocked";
  const review = run.stage === "awaiting_user_review";
  const approval = run.stage === "awaiting_plan_approval";
  return (
    <section
      className="goal-mode-banner"
      aria-label={`${label}: ${run.objective}`}
    >
      <button className="goal-mode-main" type="button" onClick={onDetails}>
        <span className="goal-mode-icon">
          <i className="bi bi-bullseye"></i>
        </span>
        <span className="goal-mode-copy">
          <strong>{label}</strong>
          <span>{run.objective}</span>
        </span>
        <span className="goal-mode-count">
          {progress.completed}/{progress.total}
        </span>
        <span className="goal-mode-details">
          Details <i className="bi bi-chevron-right"></i>
        </span>
      </button>
      <div className="goal-mode-actions">
        {approval ? null : review
          ? (
            <button
              className="btn btn-primary btn-sm"
              type="button"
              disabled={busy}
              onClick={onAccept}
            >
              Accept goal
            </button>
          )
          : paused
          ? (
            <button
              className="btn btn-primary btn-sm"
              type="button"
              disabled={busy}
              onClick={onResume}
            >
              Resume
            </button>
          )
          : (
            <button
              className="btn btn-light btn-sm"
              type="button"
              disabled={busy || run.pause_requested}
              onClick={onPause}
            >
              {run.pause_requested ? "Pause requested" : "Pause"}
            </button>
          )}
        <details className="goal-mode-menu">
          <summary className="btn btn-light btn-sm" aria-label="Goal actions">
            <i className="bi bi-three-dots"></i>
          </summary>
          <div className="goal-mode-menu-popover">
            <button className="dropdown-item" type="button" onClick={onEdit}>
              {paused || review || approval ? "Edit goal" : "Pause and edit"}
            </button>
            <button
              className="dropdown-item text-danger"
              type="button"
              onClick={onStop}
            >
              Stop goal
            </button>
          </div>
        </details>
      </div>
    </section>
  );
}
