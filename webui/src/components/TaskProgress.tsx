import type {
  MultiTaskCheckpoint,
  MultiTaskStage,
  TaskPlanRejected,
  TaskRun,
  TaskState,
} from "../types";
import type { ReactNode } from "react";

const successfulStates = new Set<TaskState>(["committed", "no_change"]);

export function taskStateLabel(state: TaskState): string {
  switch (state) {
    case "queued":
      return "Queued";
    case "running":
      return "In progress";
    case "committed":
      return "Committed";
    case "no_change":
      return "Verified no change";
    case "blocked":
      return "Blocked";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    case "superseded":
      return "Superseded";
  }
}

export function multiTaskStageLabel(stage: MultiTaskStage): string {
  switch (stage) {
    case "running_task":
      return "Delivering Tasks";
    case "evaluating":
      return "Checking Task readiness";
    case "paused":
      return "Tasks paused";
    case "blocked":
      return "Task blocked";
    case "ready":
      return "Tasks complete";
    case "failed":
      return "Tasks failed";
    case "cancelled":
      return "Tasks cancelled";
  }
}

function TaskStatusIcon({ task }: { task: TaskRun }) {
  const icon = successfulStates.has(task.state)
    ? "bi-check-lg"
    : task.state === "running"
    ? "bi-arrow-right"
    : task.state === "blocked" || task.state === "failed"
    ? "bi-exclamation-lg"
    : task.state === "cancelled" || task.state === "superseded"
    ? "bi-dash-lg"
    : "bi-circle";
  return <i className={`bi ${icon}`} aria-hidden="true" />;
}

export function TaskProgress(
  { checkpoint, activeTaskDetail }: {
    checkpoint: MultiTaskCheckpoint;
    activeTaskDetail?: ReactNode;
  },
) {
  const { run } = checkpoint;
  const currentTasks = run.tasks.filter((task) => task.state !== "superseded");
  const completed =
    currentTasks.filter((task) => successfulStates.has(task.state)).length;

  return (
    <section className="task-progress" aria-labelledby="task-progress-title">
      <header className="task-progress-header">
        <span className="task-progress-kicker">Tasks</span>
        <h2 id="task-progress-title">{run.plan.artifact.objective}</h2>
        <span className="task-progress-count">
          {completed} of {currentTasks.length}
        </span>
        <span className="visually-hidden">
          {multiTaskStageLabel(run.stage)}
        </span>
      </header>
      <ol className="task-progress-list">
        {currentTasks.map((task) => {
          const active = task.spec.id === run.active_task_id;
          const tokenPercent = Math.min(
            100,
            Math.round(
              (task.counters.generated_tokens /
                Math.max(1, task.spec.budget.total_generated_tokens)) * 100,
            ),
          );
          return (
            <li
              key={`${task.spec.id}-${task.revision}`}
              className={`task-progress-item task-state-${task.state}`}
              aria-current={active ? "true" : undefined}
            >
              <span className="task-progress-icon">
                <TaskStatusIcon task={task} />
              </span>
              <div className="task-progress-copy">
                <span className="task-progress-title-row">
                  <strong>{task.spec.title}</strong>
                  <span className={`task-kind task-kind-${task.spec.kind}`}>
                    {task.spec.kind === "goal" ? "Goal" : "Build"}
                  </span>
                </span>
                <span className="task-progress-detail">
                  {taskStateLabel(task.state)}
                  {task.attempts > 1 ? ` · attempt ${task.attempts}` : ""}
                  {task.result?.commits.length
                    ? ` · ${task.result.commits.length} commit${
                      task.result.commits.length === 1 ? "" : "s"
                    }`
                    : ""}
                </span>
                {(active || task.counters.generated_tokens > 0) && (
                  <span
                    className="task-budget-track"
                    role="progressbar"
                    aria-label={`${task.spec.title} token budget`}
                    aria-valuemin={0}
                    aria-valuemax={task.spec.budget.total_generated_tokens}
                    aria-valuenow={task.counters.generated_tokens}
                  >
                    <span style={{ width: `${tokenPercent}%` }} />
                  </span>
                )}
                {task.blocked_reason && (
                  <span className="task-progress-reason">
                    {task.blocked_reason}
                  </span>
                )}
                {active && activeTaskDetail
                  ? (
                    <div className="task-progress-active-detail">
                      {activeTaskDetail}
                    </div>
                  )
                  : null}
              </div>
            </li>
          );
        })}
      </ol>
      {run.reason && (
        <p className="task-progress-run-reason" role="status">
          {run.reason}
        </p>
      )}
    </section>
  );
}

export function TaskPlanningRecovery({
  rejection,
  busy,
  onRetry,
  onEdit,
  onRunAsBuild,
}: {
  rejection: TaskPlanRejected;
  busy: boolean;
  onRetry: () => void;
  onEdit: () => void;
  onRunAsBuild: () => void;
}) {
  const reason = rejection.failures.at(-1)?.reason ??
    "Task planning stopped before it could produce a valid plan.";
  return (
    <section
      className="task-planning-recovery"
      aria-labelledby="task-recovery-title"
    >
      <div>
        <span className="task-progress-kicker">Task planning</span>
        <h2 id="task-recovery-title">Planning needs a decision</h2>
        <p>{reason}</p>
      </div>
      <div className="task-planning-recovery-actions">
        <button
          className="btn btn-primary btn-sm"
          disabled={busy}
          onClick={onRetry}
        >
          Retry planning
        </button>
        <button
          className="btn btn-outline-secondary btn-sm"
          disabled={busy}
          onClick={onEdit}
        >
          Edit request
        </button>
        <button
          className="btn btn-outline-secondary btn-sm"
          disabled={busy}
          onClick={onRunAsBuild}
        >
          Run as one Build
        </button>
      </div>
    </section>
  );
}
