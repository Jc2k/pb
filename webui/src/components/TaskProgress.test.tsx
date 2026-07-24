import { renderToStaticMarkup } from "react-dom/server";
import { equal, ok } from "node:assert/strict";
import { describe, it } from "node:test";
import type { MultiTaskCheckpoint, TaskRun } from "../types";
import {
  multiTaskStageLabel,
  TaskPlanningDetails,
  TaskPlanningRecovery,
  TaskProgress,
  taskStateLabel,
} from "./TaskProgress";

function task(
  id: string,
  title: string,
  state: TaskRun["state"],
  kind: TaskRun["spec"]["kind"],
): TaskRun {
  return {
    spec: {
      id,
      title,
      description: title,
      requirement_ids: ["r1"],
      depends_on: [],
      acceptance_ids: ["a1"],
      scope_hints: [],
      effort: "small",
      kind,
      budget: {
        max_workflows: 1,
        stage_steps: 10,
        total_model_invocations: 10,
        total_generated_tokens: 1_000,
        advisory_calls: 2,
        plan_cycles: 1,
        repair_cycles: 1,
        wall_time_minutes: 10,
      },
    },
    revision: 1,
    state,
    attempts: state === "queued" ? 0 : 1,
    counters: {
      workflows: state === "queued" ? 0 : 1,
      stage_steps: 2,
      model_invocations: 2,
      generated_tokens: state === "queued" ? 0 : 200,
      advisory_calls: 0,
      plan_cycles: 1,
      repair_cycles: 0,
      elapsed_ms: 100,
    },
  };
}

const checkpoint: MultiTaskCheckpoint = {
  sha256: "a".repeat(64),
  run: {
    id: "multi-1",
    stage: "running_task",
    active_task_id: "t2",
    plan: {
      sha256: "b".repeat(64),
      artifact: {
        objective: "Ship durable imports",
        tasks: [],
      },
    },
    tasks: [
      task("t1", "Persist lifecycle", "committed", "build"),
      task("t2", "Prove recovery", "running", "goal"),
      task("t3", "Expose progress", "queued", "build"),
    ],
  },
};

describe("TaskProgress", () => {
  it("shows ordered Task state, kind, progress and the active Task", () => {
    const html = renderToStaticMarkup(
      <TaskProgress
        checkpoint={checkpoint}
        activeTaskDetail={<div>Existing Goal controls</div>}
      />,
    );
    ok(html.includes("Tasks"));
    ok(html.includes("1 of 3"));
    ok(html.includes("Persist lifecycle"));
    ok(html.includes("Prove recovery"));
    ok(html.includes("Goal"));
    ok(html.includes('aria-current="true"'));
    ok(html.includes('role="progressbar"'));
    ok(html.includes("Existing Goal controls"));
  });

  it("uses the Task product language consistently", () => {
    equal(taskStateLabel("no_change"), "Verified no change");
    equal(multiTaskStageLabel("ready"), "Tasks complete");
  });

  it("shows whole-request audit coverage at terminal readiness", () => {
    const html = renderToStaticMarkup(
      <TaskProgress
        checkpoint={{
          ...checkpoint,
          run: {
            ...checkpoint.run,
            stage: "ready",
            active_task_id: undefined,
            completion_audit: {
              plan_sha256: checkpoint.run.plan.sha256,
              completed_at_ms: 70,
              requirements: [{
                requirement_id: "r1",
                task_ids: ["t1"],
                acceptance_ids: ["a1"],
                evidence_refs: ["workflow:ready"],
                commits: ["a".repeat(40)],
              }],
            },
          },
        }}
      />,
    );
    ok(html.includes("Whole request audited"));
    ok(html.includes("1 requirement"));
  });
});

describe("TaskPlanningDetails", () => {
  it("shows the controller decision and every preserved constrained attempt", () => {
    const html = renderToStaticMarkup(
      <TaskPlanningDetails
        transcript={{
          decision: "one_build_planner_fallback",
          summary: "Task planning failed soft to the original Build",
          attempts: [{
            attempt: 1,
            stage: "planner",
            prompt: "Partition this Build",
            schema: { type: "object" },
            raw_output: "{}",
            failure: "tasks is required",
            prompt_tokens: 20,
            generated_tokens: 2,
            duration_ms: 5,
          }],
        }}
      />,
    );
    ok(html.includes("Task planning details"));
    ok(html.includes("failed soft to the original Build"));
    ok(html.includes("tasks is required"));
    ok(html.includes("Partition this Build"));
    ok(html.includes("Constrained output"));
  });
});

describe("TaskPlanningRecovery", () => {
  it("offers only the three explicit recovery decisions", () => {
    const html = renderToStaticMarkup(
      <TaskPlanningRecovery
        rejection={{
          outcome: "attempts_exhausted",
          attempts: 2,
          failures: [{
            attempt: 2,
            stage: "reviewer",
            reason: "The proposed Tasks are still too broad.",
          }],
          recovery_actions: [
            "retry_planning",
            "edit_request",
            "run_as_one_build",
          ],
        }}
        busy={false}
        onRetry={() => {}}
        onEdit={() => {}}
        onRunAsBuild={() => {}}
      />,
    );
    ok(html.includes("Retry planning"));
    ok(html.includes("Edit request"));
    ok(html.includes("Run as one Build"));
    ok(!html.includes("Continue automatically"));
  });
});
