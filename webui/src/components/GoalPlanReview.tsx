import type { GoalCheckpoint } from "../types";

interface Props {
  goal: GoalCheckpoint;
  busy?: boolean;
  onApprove: (planSha256: string, amendmentId?: string) => void;
  onDiscardAmendment: (amendmentId: string) => void;
  onEdit: () => void;
  onStop: () => void;
}

export function GoalPlanReview({
  goal,
  busy = false,
  onApprove,
  onDiscardAmendment,
  onEdit,
  onStop,
}: Props) {
  const amendment = goal.run.pending_amendment;
  const milestones = amendment?.replacement_milestones ?? goal.run.milestones;
  const planSha256 = amendment?.replacement_plan_sha256 ?? goal.run.plan_sha256;
  return (
    <section className="goal-plan-review" role="status" aria-live="polite">
      <div className="goal-plan-heading">
        <div>
          <span className="goal-eyebrow">
            {amendment ? "Goal changes ready" : "Goal plan ready"}
          </span>
          <h2>{amendment?.objective ?? goal.run.objective}</h2>
        </div>
        <span className="goal-plan-count">
          {milestones.length} milestone{milestones.length === 1 ? "" : "s"}
        </span>
      </div>
      {amendment
        ? (
          <p className="goal-plan-notice">
            Review the replacement plan below. Completed milestone history
            remains preserved.
          </p>
        )
        : null}
      <ol className="goal-plan-milestones">
        {milestones.filter((milestone) => milestone.status !== "superseded")
          .map((milestone, index) => (
            <li key={milestone.id}>
              <span>{index + 1}</span>
              <div>
                <strong>{milestone.title}</strong>
                <p>{milestone.description}</p>
              </div>
            </li>
          ))}
      </ol>
      <div className="goal-plan-meta">
        <span>
          <i className="bi bi-arrow-repeat"></i>{" "}
          {(amendment?.continuation ?? goal.run.continuation).replaceAll(
            "_",
            " ",
          )}
        </span>
        <span>
          <i className="bi bi-shield-check"></i> No publishing
        </span>
      </div>
      <div className="goal-plan-actions">
        <button
          className="btn btn-primary"
          type="button"
          disabled={busy}
          onClick={() => onApprove(planSha256, amendment?.id)}
        >
          {amendment ? "Approve changes and resume" : "Approve and start"}
        </button>
        {!amendment
          ? (
            <button
              className="btn btn-light"
              type="button"
              disabled={busy}
              onClick={onEdit}
            >
              Edit goal
            </button>
          )
          : null}
        {amendment
          ? (
            <button
              className="btn btn-light"
              type="button"
              disabled={busy}
              onClick={() => onDiscardAmendment(amendment.id)}
            >
              Discard changes
            </button>
          )
          : null}
        <button
          className="btn btn-link text-danger"
          type="button"
          disabled={busy}
          onClick={onStop}
        >
          Stop goal
        </button>
      </div>
    </section>
  );
}
