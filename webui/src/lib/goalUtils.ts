import type { GoalRun, GoalStage } from "../types";

export function goalStageLabel(
  stage: GoalStage,
  pauseRequested = false,
): string {
  if (pauseRequested) return "Pausing after current action";
  switch (stage) {
    case "planning":
      return "Planning goal";
    case "plan_review":
      return "Reviewing goal plan";
    case "plan_revision":
      return "Revising goal plan";
    case "awaiting_plan_approval":
      return "Review goal plan";
    case "running_milestone":
      return "Goal running";
    case "evaluating":
      return "Checking goal progress";
    case "awaiting_user_review":
      return "Goal ready for review";
    case "paused":
      return "Goal paused";
    case "blocked":
      return "Goal needs help";
    case "completed":
      return "Goal complete";
    case "failed":
      return "Goal stopped";
    case "cancelled":
      return "Goal cancelled";
  }
}

export function goalProgress(
  run: GoalRun,
): { completed: number; total: number } {
  const visible = run.milestones.filter((milestone) =>
    milestone.status !== "superseded"
  );
  return {
    completed:
      visible.filter((milestone) => milestone.status === "completed").length,
    total: visible.length,
  };
}

export function goalPercent(used: number, limit: number): number {
  if (limit <= 0) return 0;
  return Math.min(100, Math.round((used / limit) * 100));
}
