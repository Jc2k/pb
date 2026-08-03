import type {
  EventEnvelope,
  ProjectEntry,
  ProjectSessionSnapshot,
  ProjectSessionTerminalTransition,
  SessionDetails,
  SessionItem,
  SessionStreamSnapshot,
} from "../types";

export const eventFields = {
  semantic_gate: [["receipt"], ["nesting_depth", "timestamp_ms"]],
  harness_experiment_configured: [["observation_rendering"], ["timestamp_ms"]],
  conversation_turn_started: [["turn_id", "intent", "task"], ["timestamp_ms"]],
  delivery_proposed: [["proposal_id", "source_turn_id", "task_summary"], [
    "timestamp_ms",
  ]],
  goal_proposed: [["proposal_id", "source_turn_id", "objective", "criteria"], [
    "timestamp_ms",
  ]],
  task_plan_accepted: [["multi_task_id", "plan_sha256", "task_count"], [
    "timestamp_ms",
  ]],
  task_plan_rejected: [["outcome", "attempts"], ["timestamp_ms"]],
  tasks_changed: [["multi_task_id", "stage", "checkpoint_sha256"], [
    "outcome",
    "active_task_id",
    "timestamp_ms",
  ]],
  goal_started: [["goal_id", "objective", "plan_sha256"], ["timestamp_ms"]],
  goal_plan_awaiting_approval: [["goal_id", "plan_sha256", "milestones"], [
    "timestamp_ms",
  ]],
  goal_plan_approved: [["goal_id", "plan_sha256"], ["timestamp_ms"]],
  goal_milestone_started: [["goal_id", "milestone_id", "title"], [
    "timestamp_ms",
  ]],
  goal_milestone_completed: [["goal_id", "milestone_id", "workflow_id"], [
    "timestamp_ms",
  ]],
  goal_pause_requested: [["goal_id"], ["timestamp_ms"]],
  goal_paused: [["goal_id"], ["timestamp_ms"]],
  goal_resumed: [["goal_id"], ["timestamp_ms"]],
  goal_amendment_requested: [[
    "goal_id",
    "amendment_id",
    "replacement_plan_sha256",
  ], ["timestamp_ms"]],
  goal_change_requested: [["goal_id", "kind", "summary"], ["timestamp_ms"]],
  goal_amendment_resolved: [["goal_id", "amendment_id", "accepted"], [
    "timestamp_ms",
  ]],
  goal_ready_for_review: [["goal_id", "checkpoint_sha256"], ["timestamp_ms"]],
  goal_completed: [[
    "goal_id",
    "outcome",
    "completion_basis",
    "checkpoint_sha256",
  ], ["timestamp_ms"]],
  goal_failed: [["goal_id", "outcome", "reason", "checkpoint_sha256"], [
    "timestamp_ms",
  ]],
  goal_cancelled: [["goal_id", "checkpoint_sha256"], ["timestamp_ms"]],
  workflow_started: [["workflow_id", "source_turn_id", "policy_sha256"], [
    "timestamp_ms",
  ]],
  workflow_resumed: [["workflow_id", "stage", "checkpoint_sha256"], [
    "timestamp_ms",
  ]],
  workflow_stage_started: [["workflow_id", "stage"], ["timestamp_ms"]],
  workflow_stage_completed: [["workflow_id", "stage"], ["timestamp_ms"]],
  workflow_artifact_accepted: [[
    "workflow_id",
    "artifact_kind",
    "artifact_id",
    "sha256",
  ], ["timestamp_ms"]],
  workflow_challenge_raised: [[
    "workflow_id",
    "challenge_id",
    "severity",
    "summary",
  ], ["timestamp_ms"]],
  workflow_evidence_invalidated: [[
    "workflow_id",
    "previous_fingerprint",
    "current_fingerprint",
    "reason",
  ], ["timestamp_ms"]],
  workflow_blocked: [["workflow_id", "outcome", "cause", "reason"], [
    "current_user",
    "timestamp_ms",
  ]],
  workflow_completed: [["workflow_id", "outcome", "checkpoint_sha256"], [
    "ready_evidence_sha256",
    "timestamp_ms",
  ]],
  started: [
    ["task", "model", "workspace", "branch", "attachments", "profile"],
    ["focus_root", "timestamp_ms"],
  ],
  step_started: [["step", "max_steps", "profile"], [
    "nesting_depth",
    "timestamp_ms",
  ]],
  model_loading: [["model", "profile"], ["nesting_depth", "timestamp_ms"]],
  reasoning: [["content", "profile"], ["nesting_depth", "timestamp_ms"]],
  tool_call: [["tool", "arguments", "call_id", "batch_id", "actor"], [
    "nesting_depth",
    "timestamp_ms",
  ]],
  controller_observation: [["receipt", "actor"], [
    "assisting_profile",
    "nesting_depth",
    "timestamp_ms",
  ]],
  controller_closure: [["workflow_id", "stage", "reason", "actor"], [
    "assisting_profile",
    "nesting_depth",
    "timestamp_ms",
  ]],
  controller_mutation: [["receipt", "actor"], [
    "assisting_profile",
    "nesting_depth",
    "timestamp_ms",
  ]],
  tool_batch: [[
    "call_count",
    "parallel_safe_count",
    "useful_count",
    "bookkeeping_only_count",
    "rejected_as_dependent",
  ], ["nesting_depth", "timestamp_ms"]],
  tool_result: [[
    "tool",
    "result",
    "call_id",
    "batch_id",
    "outcome",
    "actor",
    "duration_ms",
  ], [
    "energy_joules",
    "energy_kwh",
    "average_power_watts",
    "energy_shared_calls",
    "nesting_depth",
    "timestamp_ms",
  ]],
  executor_started: [["executor_id", "kind", "success", "detail"], [
    "timestamp_ms",
  ]],
  check_result: [[
    "check_id",
    "exit_status",
    "success",
    "timed_out",
    "output",
    "truncated",
    "duration_ms",
    "fingerprint",
    "dependency_outputs",
    "reused",
  ], [
    "command",
    "cwd",
    "executor",
    "source",
    "command_fingerprint",
    "output_fingerprint",
    "skip_reason",
    "nesting_depth",
    "timestamp_ms",
  ]],
  team_message: [["actor", "tone", "purpose", "message", "evidence"], [
    "handoff",
    "detail",
    "nesting_depth",
    "timestamp_ms",
  ]],
  handoff_summary: [["summary"], ["nesting_depth", "timestamp_ms"]],
  commit_result: [["success", "created", "reused", "changed_paths", "detail"], [
    "oid",
    "subject",
    "nesting_depth",
    "timestamp_ms",
  ]],
  user_question: [["question_id", "question", "choices", "profile"], [
    "timestamp_ms",
  ]],
  user_answer: [["question_id", "answer"], ["timestamp_ms"]],
  user_message: [["message_id", "message"], ["timestamp_ms"]],
  user_message_applied: [["message_id"], ["timestamp_ms"]],
  correction: [["message", "kind", "summary", "actor"], [
    "assisting_profile",
    "nesting_depth",
    "timestamp_ms",
  ]],
  sub_agent_started: [["profile", "task", "nesting_depth"], ["timestamp_ms"]],
  sub_agent_finished: [["profile", "result"], [
    "nesting_depth",
    "timestamp_ms",
  ]],
  diff: [["path", "diff"], ["nesting_depth", "timestamp_ms"]],
  final: [["content", "profile"], ["nesting_depth", "timestamp_ms"]],
  final_grace: [["status", "detail"], ["nesting_depth", "timestamp_ms"]],
  context_limit: [[
    "context_capacity",
    "reserved_generation_tokens",
    "safety_margin_tokens",
    "usable_prompt_capacity",
    "measured_prompt_tokens",
    "largest_sections",
  ], ["nesting_depth", "timestamp_ms"]],
  llm_invocation: [[
    "step",
    "purpose",
    "profile",
    "duration_ms",
    "prompt_tokens",
    "generated_tokens",
  ], [
    "workflow_stage",
    "prompt_cache",
    "context",
    "native",
    "energy_joules",
    "energy_kwh",
    "average_power_watts",
    "nesting_depth",
    "timestamp_ms",
  ]],
  session_metrics: [[
    "llm_invocations",
    "llm_runtime_ms",
    "prompt_tokens",
    "generated_tokens",
    "tool_calls",
    "tool_runtime_ms",
    "cache_persistence_queued_checkpoints",
    "cache_persistence_completed_checkpoints",
    "cache_persistence_wall_ms",
    "cache_persistence_failures",
    "wall_runtime_ms",
    "started_at_ms",
    "ended_at_ms",
    "display_energy_excluded",
    "idle_baseline_applied",
    "energy_complete",
    "energy_exclusive",
  ], [
    "llm_energy_joules",
    "llm_energy_kwh",
    "tool_energy_joules",
    "tool_energy_kwh",
    "total_energy_joules",
    "total_energy_kwh",
    "gross_energy_joules",
    "adjusted_energy_joules",
    "average_power_watts",
    "energy_measured_ms",
    "energy_coverage",
    "energy_source",
    "nesting_depth",
    "timestamp_ms",
  ]],
  session_title: [["title"], ["timestamp_ms"]],
  session_state_changed: [["status", "running", "paused"], ["timestamp_ms"]],
  session_summary: [[
    "branch",
    "commits",
    "reached_final",
    "contract_status",
    "verified_completed",
    "summary",
    "power_summary",
    "diff_stat",
    "diff",
  ], ["termination_reason", "handoff_outcome", "timestamp_ms"]],
  error: [["summary", "detail"], ["nesting_depth", "timestamp_ms"]],
} as const;

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function json(text: string, label: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function nonNegativeInteger(value: unknown, label: string): number {
  const number = finiteNumber(value, label);
  if (!Number.isSafeInteger(number) || number < 0) {
    throw new Error(`${label} must be a non-negative safe integer`);
  }
  return number;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || !value) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function exactKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  label: string,
): void {
  const allowedKeys = new Set(allowed);
  const unexpected = Object.keys(value).find((key) => !allowedKeys.has(key));
  if (unexpected) {
    throw new Error(`${label} contains unknown field ${unexpected}`);
  }
}

function nullableString(value: unknown, label: string): void {
  if (value !== null && typeof value !== "string") {
    throw new Error(`${label} must be a string or null`);
  }
}

function optionalString(value: unknown, label: string): void {
  if (value !== undefined && typeof value !== "string") {
    throw new Error(`${label} must be a string when present`);
  }
}

function enumValue(value: unknown, values: Set<string>, label: string): void {
  if (typeof value !== "string" || !values.has(value)) {
    throw new Error(`${label} is invalid`);
  }
}

function nullableEnum(
  value: unknown,
  values: Set<string>,
  label: string,
): void {
  if (value !== null) enumValue(value, values, label);
}

function optionalEnum(
  value: unknown,
  values: Set<string>,
  label: string,
): void {
  if (value !== undefined) enumValue(value, values, label);
}

function projectEntry(value: unknown, label: string): ProjectEntry {
  const project = record(value, label);
  exactKeys(
    project,
    ["id", "name", "path", "repository_root", "notify_on_finish"],
    label,
  );
  requiredString(project.id, `${label} id`);
  requiredString(project.name, `${label} name`);
  requiredString(project.path, `${label} path`);
  optionalString(project.repository_root, `${label} repository root`);
  if (typeof project.notify_on_finish !== "boolean") {
    throw new Error(`${label} notification setting must be boolean`);
  }
  return project as unknown as ProjectEntry;
}

function sessionProject(value: unknown, label: string): void {
  const project = record(value, label);
  exactKeys(project, ["id", "name", "path"], label);
  requiredString(project.id, `${label} id`);
  requiredString(project.name, `${label} name`);
  requiredString(project.path, `${label} path`);
}

function projectUsageStats(value: unknown, label: string): void {
  const stats = record(value, label);
  exactKeys(
    stats,
    ["tokens", "runtime_ms", "tool_calls", "energy_kwh", "energy_joules"],
    label,
  );
  nonNegativeInteger(stats.tokens, `${label} tokens`);
  nonNegativeInteger(stats.runtime_ms, `${label} runtime`);
  nonNegativeInteger(stats.tool_calls, `${label} tool calls`);
  for (const field of ["energy_kwh", "energy_joules"] as const) {
    const energy = stats[field];
    if (energy !== null && energy !== undefined) {
      if (finiteNumber(energy, `${label} ${field}`) < 0) {
        throw new Error(`${label} ${field} must be non-negative`);
      }
    }
  }
}

function projectUsageSummary(value: unknown, label: string): void {
  const summary = record(value, label);
  exactKeys(summary, ["total", "today"], label);
  projectUsageStats(summary.total, `${label} total`);
  projectUsageStats(summary.today, `${label} today`);
}

const handoffOutcomes = new Set([
  "pending",
  "ready",
  "no_change",
  "checks_failed",
  "executor_unavailable",
  "commit_blocked",
  "repair_exhausted",
  "incomplete",
]);

const sessionStatuses = new Set([
  "queued",
  "running",
  "paused",
  "completed",
  "failed",
]);
const turnIntents = new Set(["discuss", "deliver", "auto"]);
const workflowStages = new Set([
  "planning",
  "plan_review",
  "plan_revision",
  "implementing",
  "checking",
  "code_review",
  "repairing",
  "committing",
  "ready",
  "failed",
  "blocked",
  "cancelled",
]);
const workflowOutcomes = new Set([
  "ready",
  "no_change",
  "plan_rejected",
  "plan_cycles_exhausted",
  "checks_failed",
  "review_failed",
  "repair_cycles_exhausted",
  "contract_unsatisfied",
  "executor_unavailable",
  "commit_blocked",
  "step_limit",
  "invocation_limit",
  "token_limit",
  "context_limit",
  "engine_error",
  "cancelled",
]);
const goalStages = new Set([
  "planning",
  "plan_review",
  "plan_revision",
  "awaiting_plan_approval",
  "running_milestone",
  "evaluating",
  "awaiting_user_review",
  "paused",
  "blocked",
  "completed",
  "failed",
  "cancelled",
]);
const goalOutcomes = new Set([
  "complete",
  "budget_exhausted",
  "criteria_unsatisfied",
  "milestone_failed",
  "repeated_no_progress",
  "authority_denied",
  "context_limit",
  "engine_error",
  "cancelled",
]);
const goalCompletionBases = new Set(["machine_verified", "user_accepted"]);
const multiTaskStages = new Set([
  "running_task",
  "evaluating",
  "paused",
  "blocked",
  "ready",
  "failed",
  "cancelled",
]);
const multiTaskOutcomes = new Set([
  "ready",
  "task_blocked",
  "task_failed",
  "budget_exhausted",
  "cancelled",
]);

function goalSummary(value: unknown, label: string): void {
  const goal = record(value, label);
  exactKeys(
    goal,
    [
      "id",
      "objective",
      "stage",
      "outcome",
      "completion_basis",
      "completed_milestones",
      "total_milestones",
      "current_milestone_title",
      "generated_tokens",
      "generated_token_limit",
      "workflows",
      "workflow_limit",
      "active",
      "plan_sha256",
    ],
    label,
  );
  requiredString(goal.id, `${label} id`);
  requiredString(goal.objective, `${label} objective`);
  enumValue(goal.stage, goalStages, `${label} stage`);
  optionalEnum(goal.outcome, goalOutcomes, `${label} outcome`);
  optionalEnum(
    goal.completion_basis,
    goalCompletionBases,
    `${label} completion basis`,
  );
  nonNegativeInteger(
    goal.completed_milestones,
    `${label} completed milestones`,
  );
  nonNegativeInteger(goal.total_milestones, `${label} total milestones`);
  optionalString(goal.current_milestone_title, `${label} current milestone`);
  nonNegativeInteger(goal.generated_tokens, `${label} generated tokens`);
  nonNegativeInteger(goal.generated_token_limit, `${label} token limit`);
  nonNegativeInteger(goal.workflows, `${label} workflows`);
  nonNegativeInteger(goal.workflow_limit, `${label} workflow limit`);
  if (typeof goal.active !== "boolean") {
    throw new Error(`${label} active must be boolean`);
  }
  requiredString(goal.plan_sha256, `${label} plan digest`);
}

function multiTaskSummary(value: unknown, label: string): void {
  const multiTask = record(value, label);
  exactKeys(
    multiTask,
    [
      "id",
      "stage",
      "outcome",
      "completed_tasks",
      "total_tasks",
      "active_task_title",
    ],
    label,
  );
  requiredString(multiTask.id, `${label} id`);
  enumValue(multiTask.stage, multiTaskStages, `${label} stage`);
  optionalEnum(multiTask.outcome, multiTaskOutcomes, `${label} outcome`);
  nonNegativeInteger(multiTask.completed_tasks, `${label} completed tasks`);
  nonNegativeInteger(multiTask.total_tasks, `${label} total tasks`);
  optionalString(multiTask.active_task_title, `${label} active task title`);
}

function sessionItem(value: unknown, index: number): SessionItem {
  const label = `session ${index}`;
  const session = record(value, label);
  exactKeys(
    session,
    [
      "session_id",
      "task",
      "title",
      "running",
      "paused",
      "status",
      "intent",
      "branch",
      "workdir",
      "project",
      "handoff_outcome",
      "pending_question",
      "started_at_ms",
      "updated_at_ms",
      "workflow_id",
      "workflow_stage",
      "workflow_outcome",
      "strict_workflow",
      "goal",
      "active_goal",
      "multi_task",
      "active_multi_task",
    ],
    label,
  );
  requiredString(session.session_id, `${label} id`);
  requiredString(session.task, `${label} task`);
  nullableString(session.title, `${label} title`);
  if (
    typeof session.running !== "boolean" || typeof session.paused !== "boolean"
  ) {
    throw new Error(`${label} running and paused state must be boolean`);
  }
  enumValue(session.status, sessionStatuses, `${label} status`);
  nullableEnum(session.intent, turnIntents, `${label} intent`);
  nullableString(session.branch, `${label} branch`);
  nullableString(session.workdir, `${label} workdir`);
  if (session.project !== null) {
    sessionProject(session.project, `${label} project`);
  }
  nullableEnum(
    session.handoff_outcome,
    handoffOutcomes,
    `${label} handoff outcome`,
  );
  if (session.pending_question !== null) {
    const question = record(
      session.pending_question,
      `${label} pending question`,
    );
    exactKeys(
      question,
      ["question_id", "question", "choices"],
      `${label} pending question`,
    );
    requiredString(question.question_id, `${label} question id`);
    requiredString(question.question, `${label} question`);
    if (
      !Array.isArray(question.choices) ||
      question.choices.some((choice) => typeof choice !== "string")
    ) {
      throw new Error(`${label} question choices must be strings`);
    }
  }
  nonNegativeInteger(session.started_at_ms, `${label} start time`);
  nonNegativeInteger(session.updated_at_ms, `${label} update time`);
  nullableString(session.workflow_id, `${label} workflow id`);
  nullableEnum(
    session.workflow_stage,
    workflowStages,
    `${label} workflow stage`,
  );
  nullableEnum(
    session.workflow_outcome,
    workflowOutcomes,
    `${label} workflow outcome`,
  );
  if (
    typeof session.strict_workflow !== "boolean" ||
    typeof session.active_goal !== "boolean" ||
    typeof session.active_multi_task !== "boolean"
  ) {
    throw new Error(`${label} activity flags must be boolean`);
  }
  if (session.goal !== null) goalSummary(session.goal, `${label} goal`);
  if (session.multi_task !== null) {
    multiTaskSummary(session.multi_task, `${label} multi task`);
  }
  return session as unknown as SessionItem;
}

function terminalTransition(
  value: unknown,
): ProjectSessionTerminalTransition {
  const transition = record(value, "project session terminal transition");
  exactKeys(
    transition,
    [
      "entry_key",
      "revision",
      "session_id",
      "status",
      "task",
      "title",
      "handoff_outcome",
      "project",
    ],
    "project session terminal transition",
  );
  requiredString(transition.entry_key, "terminal transition entry key");
  nonNegativeInteger(transition.revision, "terminal transition revision");
  requiredString(transition.session_id, "terminal transition session id");
  if (transition.status !== "completed" && transition.status !== "failed") {
    throw new Error("terminal transition status must be completed or failed");
  }
  requiredString(transition.task, "terminal transition task");
  if (transition.title !== null && typeof transition.title !== "string") {
    throw new Error("terminal transition title must be a string or null");
  }
  if (
    transition.handoff_outcome !== null &&
    !handoffOutcomes.has(String(transition.handoff_outcome))
  ) {
    throw new Error("terminal transition handoff outcome is invalid");
  }
  if (transition.project !== null) {
    projectEntry(transition.project, "terminal transition project");
  }
  return transition as unknown as ProjectSessionTerminalTransition;
}

const agentProfiles = new Set([
  "build",
  "scout",
  "review",
  "explore",
  "plan",
  "ask",
  "research",
  "monitor",
]);
const teamMessageTones = new Set(["info", "success", "warning", "error"]);
const teamMessagePurposes = new Set([
  "general",
  "handoff_progress",
  "handoff_outcome",
]);
const chatterAudiences = new Set(["team", "current_user"]);
const transcriptVisibilities = new Set([
  "visible",
  "evidence_only",
  "activity",
]);
const transcriptKinds = new Set([
  "conversation",
  "activity",
  "evidence",
  "correction",
  "repeated_tool_correction",
  "no_progress_correction",
  "dependent_tool_batch_correction",
  "handoff_correction",
  "workflow_closure_checkpoint",
  "work_unit_progress",
  "workflow_blocked",
  "session_summary",
]);
const sessionRunningEffects = new Set(["unchanged", "running", "stopped"]);

function requiredKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  label: string,
): void {
  const missing = required.find((key) => !Object.hasOwn(value, key));
  if (missing) throw new Error(`${label} is missing field ${missing}`);
}

function teamActor(value: unknown, label: string): void {
  const actor = record(value, label);
  exactKeys(actor, ["kind", "id"], label);
  requiredKeys(actor, ["kind", "id"], label);
  if (actor.kind === "agent") {
    enumValue(actor.id, agentProfiles, `${label} profile`);
  } else if (actor.kind !== "automation" || actor.id !== "trinity") {
    throw new Error(`${label} is invalid`);
  }
}

function eventChatter(value: unknown, label: string): void {
  const chatter = record(value, label);
  exactKeys(
    chatter,
    ["actor", "tone", "headline", "message", "detail", "audience"],
    label,
  );
  requiredKeys(
    chatter,
    ["actor", "tone", "message", "detail", "audience"],
    label,
  );
  teamActor(chatter.actor, `${label} actor`);
  enumValue(chatter.tone, teamMessageTones, `${label} tone`);
  optionalString(chatter.headline, `${label} headline`);
  if (
    typeof chatter.message !== "string" || typeof chatter.detail !== "string"
  ) {
    throw new Error(`${label} message and detail must be strings`);
  }
  enumValue(chatter.audience, chatterAudiences, `${label} audience`);
}

function eventEvidence(value: unknown, label: string): void {
  const evidence = record(value, label);
  requiredKeys(evidence, ["kind", "value"], label);
  exactKeys(evidence, ["kind", "value"], label);
  const detail = record(evidence.value, `${label} value`);
  if (evidence.kind === "check") {
    exactKeys(
      detail,
      [
        "check_id",
        "exit_status",
        "success",
        "timed_out",
        "output",
        "duration_ms",
        "command",
        "cwd",
        "executor",
        "reused",
        "skip_reason",
      ],
      `${label} check`,
    );
    requiredKeys(
      detail,
      [
        "check_id",
        "exit_status",
        "success",
        "timed_out",
        "output",
        "duration_ms",
        "reused",
      ],
      `${label} check`,
    );
    requiredString(detail.check_id, `${label} check id`);
  } else if (evidence.kind === "commit") {
    exactKeys(
      detail,
      [
        "success",
        "created",
        "reused",
        "oid",
        "subject",
        "changed_paths",
        "detail",
      ],
      `${label} commit`,
    );
    requiredKeys(
      detail,
      ["success", "created", "reused", "changed_paths", "detail"],
      `${label} commit`,
    );
  } else {
    throw new Error(`${label} kind is invalid`);
  }
}

function transcriptMetadata(value: unknown): void {
  const transcript = record(value, "event transcript");
  exactKeys(
    transcript,
    [
      "sequence",
      "visibility",
      "kind",
      "entry_key",
      "supersedes",
      "tool_summary",
      "dedupe_key",
      "related_action_key",
      "summary_redundant",
      "session_effect",
    ],
    "event transcript",
  );
  requiredKeys(
    transcript,
    [
      "sequence",
      "visibility",
      "kind",
      "entry_key",
      "supersedes",
      "summary_redundant",
      "session_effect",
    ],
    "event transcript",
  );
  if (nonNegativeInteger(transcript.sequence, "event sequence") < 1) {
    throw new Error("event sequence must be positive");
  }
  requiredString(transcript.entry_key, "event entry key");
  enumValue(transcript.visibility, transcriptVisibilities, "event visibility");
  enumValue(transcript.kind, transcriptKinds, "event transcript kind");
  if (
    !Array.isArray(transcript.supersedes) ||
    transcript.supersedes.some((key) => typeof key !== "string")
  ) {
    throw new Error("event supersedes must be strings");
  }
  optionalString(transcript.tool_summary, "event tool summary");
  optionalString(transcript.dedupe_key, "event dedupe key");
  optionalString(transcript.related_action_key, "event related action key");
  if (typeof transcript.summary_redundant !== "boolean") {
    throw new Error("event summary_redundant must be boolean");
  }
  const effect = record(transcript.session_effect, "event session effect");
  exactKeys(
    effect,
    ["running", "reset_intent", "title"],
    "event session effect",
  );
  requiredKeys(effect, ["running", "reset_intent"], "event session effect");
  enumValue(effect.running, sessionRunningEffects, "event running effect");
  if (typeof effect.reset_intent !== "boolean") {
    throw new Error("event reset intent must be boolean");
  }
  optionalString(effect.title, "event title effect");
}

function agentEvent(value: unknown): void {
  const event = record(value, "event payload");
  if (typeof event.type !== "string" || !event.type) {
    throw new Error("event payload type must be a non-empty string");
  }
  const shape = eventFields[event.type as keyof typeof eventFields];
  if (!shape) {
    throw new Error(`event payload type '${event.type}' is unsupported`);
  }
  const [required, optional] = shape;
  requiredKeys(event, ["type", ...required], `event payload ${event.type}`);
  exactKeys(
    event,
    ["type", ...required, ...optional],
    `event payload ${event.type}`,
  );
  for (const field of ["timestamp_ms", "nesting_depth"] as const) {
    if (event[field] !== undefined) {
      nonNegativeInteger(event[field], `event payload ${event.type} ${field}`);
    }
  }
  if (event.actor !== undefined) {
    teamActor(event.actor, `event payload ${event.type} actor`);
  }
  if (event.assisting_profile !== undefined) {
    enumValue(
      event.assisting_profile,
      agentProfiles,
      `event payload ${event.type} assisting profile`,
    );
  }
  if (event.type === "session_state_changed") {
    enumValue(event.status, sessionStatuses, "session state status");
    if (
      typeof event.running !== "boolean" || typeof event.paused !== "boolean"
    ) {
      throw new Error("session state flags must be boolean");
    }
  } else if (event.type === "session_metrics") {
    const metrics = { ...event };
    Reflect.deleteProperty(metrics, "type");
    Reflect.deleteProperty(metrics, "nesting_depth");
    Reflect.deleteProperty(metrics, "timestamp_ms");
    sessionMetrics(metrics, "session metrics event");
  } else if (event.type === "session_title") {
    requiredString(event.title, "session title event title");
  }
}

function eventEnvelope(value: unknown): EventEnvelope {
  const envelope = record(value, "event envelope");
  exactKeys(
    envelope,
    ["version", "event", "chatter", "evidence", "transcript"],
    "event envelope",
  );
  if (envelope.version !== "v5") {
    throw new Error(
      `unsupported event schema '${String(envelope.version)}'; expected 'v5'`,
    );
  }
  requiredKeys(envelope, [
    "version",
    "event",
    "chatter",
    "evidence",
    "transcript",
  ], "event envelope");
  agentEvent(envelope.event);
  if (!Array.isArray(envelope.chatter) || !Array.isArray(envelope.evidence)) {
    throw new Error("event projections must be arrays");
  }
  envelope.chatter.forEach((value, index) =>
    eventChatter(value, `event chatter ${index}`)
  );
  const event = envelope.event as Record<string, unknown>;
  const chatterAudiences = envelope.chatter.map((value) =>
    record(value, "event chatter").audience
  );
  if (
    event.type === "correction" &&
    (chatterAudiences.length !== 1 || chatterAudiences[0] !== "team")
  ) {
    throw new Error(
      "correction event must include exactly one server-authored team chatter projection",
    );
  }
  if (
    event.type === "workflow_blocked" &&
    (chatterAudiences.length !== 2 ||
      chatterAudiences.filter((audience) => audience === "team").length !== 1 ||
      chatterAudiences.filter((audience) => audience === "current_user")
          .length !== 1)
  ) {
    throw new Error(
      "blocked workflow event must include team and current-user chatter projections",
    );
  }
  if (
    event.type !== "correction" && event.type !== "workflow_blocked" &&
    chatterAudiences.length !== 0
  ) {
    throw new Error(
      `${String(event.type)} event does not support chatter projections`,
    );
  }
  envelope.evidence.forEach((value, index) =>
    eventEvidence(value, `event evidence ${index}`)
  );
  if (event.type !== "team_message" && envelope.evidence.length !== 0) {
    throw new Error(
      `${String(event.type)} event does not support evidence projections`,
    );
  }
  transcriptMetadata(envelope.transcript);
  return envelope as unknown as EventEnvelope;
}

const sessionMetricRequiredFields = [
  "llm_invocations",
  "llm_runtime_ms",
  "prompt_tokens",
  "generated_tokens",
  "tool_calls",
  "tool_runtime_ms",
  "cache_persistence_queued_checkpoints",
  "cache_persistence_completed_checkpoints",
  "cache_persistence_wall_ms",
  "cache_persistence_failures",
  "wall_runtime_ms",
  "started_at_ms",
  "ended_at_ms",
  "display_energy_excluded",
  "idle_baseline_applied",
  "energy_complete",
  "energy_exclusive",
] as const;
const sessionMetricOptionalFields = [
  "llm_energy_joules",
  "llm_energy_kwh",
  "tool_energy_joules",
  "tool_energy_kwh",
  "total_energy_joules",
  "total_energy_kwh",
  "gross_energy_joules",
  "adjusted_energy_joules",
  "average_power_watts",
  "energy_measured_ms",
  "energy_coverage",
  "energy_source",
] as const;

function sessionMetrics(value: unknown, label: string): void {
  const metrics = record(value, label);
  exactKeys(metrics, [
    ...sessionMetricRequiredFields,
    ...sessionMetricOptionalFields,
  ], label);
  requiredKeys(metrics, sessionMetricRequiredFields, label);
  for (const field of sessionMetricRequiredFields.slice(0, 13)) {
    if (finiteNumber(metrics[field], `${label} ${field}`) < 0) {
      throw new Error(`${label} ${field} must be non-negative`);
    }
  }
  for (const field of sessionMetricRequiredFields.slice(13)) {
    if (typeof metrics[field] !== "boolean") {
      throw new Error(`${label} ${field} must be boolean`);
    }
  }
  for (const field of sessionMetricOptionalFields) {
    const metric = metrics[field];
    if (metric === undefined) continue;
    if (field === "energy_source") {
      if (typeof metric !== "string") {
        throw new Error(`${label} energy source must be a string`);
      }
    } else if (finiteNumber(metric, `${label} ${field}`) < 0) {
      throw new Error(`${label} ${field} must be non-negative`);
    }
  }
}

function workflowSummary(
  value: unknown,
  label: string,
): Record<string, unknown> {
  const workflow = record(value, label);
  exactKeys(
    workflow,
    [
      "id",
      "source_turn_id",
      "task",
      "stage",
      "outcome",
      "policy_sha256",
      "commit_oid",
      "ready_evidence",
      "paused_stage",
      "blocked_reason",
      "blocked_cause",
      "recovery",
      "plan",
      "plan_review",
    ],
    label,
  );
  requiredKeys(workflow, [
    "id",
    "source_turn_id",
    "task",
    "stage",
    "policy_sha256",
  ], label);
  requiredString(workflow.id, `${label} id`);
  enumValue(workflow.stage, workflowStages, `${label} stage`);
  optionalEnum(workflow.outcome, workflowOutcomes, `${label} outcome`);
  optionalEnum(workflow.paused_stage, workflowStages, `${label} paused stage`);
  return workflow;
}

function checkpoint(
  value: unknown,
  label: string,
  kind: "goal" | "multi_task",
): void {
  const valueRecord = record(value, label);
  exactKeys(valueRecord, ["sha256", "run"], label);
  requiredKeys(valueRecord, ["sha256", "run"], label);
  requiredString(valueRecord.sha256, `${label} digest`);
  const run = record(valueRecord.run, `${label} run`);
  const required = kind === "goal"
    ? [
      "version",
      "id",
      "session_id",
      "objective",
      "stage",
      "plan_version",
      "plan_sha256",
      "policy",
      "budget",
      "authority",
      "continuation",
      "criteria",
      "milestones",
      "counters",
      "pause_requested",
      "created_at_ms",
      "updated_at_ms",
    ]
    : ["id", "stage", "tasks", "plan"];
  const optional = kind === "goal"
    ? [
      "retired_criteria",
      "active_milestone_id",
      "paused_stage",
      "blocked_reason",
      "pending_amendment",
      "outcome",
      "completion_basis",
    ]
    : [
      "active_task_id",
      "outcome",
      "reason",
      "planning_transcript",
      "completion_audit",
    ];
  exactKeys(run, [...required, ...optional], `${label} run`);
  requiredKeys(run, required, `${label} run`);
  requiredString(run.id, `${label} id`);
  enumValue(
    run.stage,
    kind === "goal" ? goalStages : multiTaskStages,
    `${label} stage`,
  );
  optionalEnum(
    run.outcome,
    kind === "goal" ? goalOutcomes : multiTaskOutcomes,
    `${label} outcome`,
  );
  if (kind === "goal") {
    optionalEnum(
      run.completion_basis,
      goalCompletionBases,
      `${label} completion basis`,
    );
    optionalEnum(run.paused_stage, goalStages, `${label} paused stage`);
  }
}

function planningTranscript(value: unknown, label: string): void {
  const transcript = record(value, label);
  exactKeys(transcript, ["decision", "summary", "attempts"], label);
  requiredKeys(transcript, ["decision", "summary", "attempts"], label);
  enumValue(
    transcript.decision,
    new Set([
      "multi_task",
      "one_build_single_task",
      "one_build_planner_fallback",
      "one_build_budget_fallback",
      "cancelled",
      "rejected",
    ]),
    `${label} decision`,
  );
  if (!Array.isArray(transcript.attempts)) {
    throw new Error(`${label} attempts must be an array`);
  }
  transcript.attempts.forEach((value, index) => {
    const attempt = record(value, `${label} attempt ${index}`);
    exactKeys(
      attempt,
      [
        "attempt",
        "stage",
        "prompt",
        "schema",
        "raw_output",
        "normalized_output",
        "failure",
        "prompt_tokens",
        "generated_tokens",
        "duration_ms",
      ],
      `${label} attempt ${index}`,
    );
    requiredKeys(
      attempt,
      [
        "attempt",
        "stage",
        "prompt",
        "schema",
        "prompt_tokens",
        "generated_tokens",
        "duration_ms",
      ],
      `${label} attempt ${index}`,
    );
    enumValue(
      attempt.stage,
      new Set(["planner", "reviewer"]),
      `${label} attempt stage`,
    );
  });
}

function rejectedTaskPlan(value: unknown, label: string): void {
  const rejected = record(value, label);
  exactKeys(rejected, [
    "outcome",
    "attempts",
    "failures",
    "recovery_actions",
    "transcript",
  ], label);
  requiredKeys(rejected, [
    "outcome",
    "attempts",
    "failures",
    "recovery_actions",
  ], label);
  enumValue(
    rejected.outcome,
    new Set([
      "attempts_exhausted",
      "budget_exhausted",
      "cancelled",
      "qualification_mismatch",
    ]),
    `${label} outcome`,
  );
  if (rejected.transcript !== undefined) {
    planningTranscript(rejected.transcript, `${label} transcript`);
  }
}

function proposal(
  value: unknown,
  label: string,
  kind: "delivery" | "goal" | "change",
): void {
  const detail = record(value, label);
  const fields = kind === "delivery"
    ? ["id", "source_turn_id", "task_summary"]
    : kind === "goal"
    ? ["id", "source_turn_id", "objective", "criteria"]
    : ["goal_id", "kind", "summary"];
  exactKeys(detail, fields, label);
  requiredKeys(detail, fields, label);
  if (kind === "change") {
    enumValue(detail.kind, new Set(["amendment", "budget"]), `${label} kind`);
  } else if (kind === "goal" && !Array.isArray(detail.criteria)) {
    throw new Error(`${label} criteria must be an array`);
  }
}

function sessionDetails(value: unknown): SessionDetails {
  const session = record(value, "session snapshot");
  const fields = [
    "session_id",
    "task",
    "title",
    "running",
    "paused",
    "cancel_requested",
    "status",
    "intent",
    "branch",
    "workdir",
    "project",
    "handoff_outcome",
    "pending_question",
    "events",
    "started_at_ms",
    "updated_at_ms",
    "metrics",
    "usage_records",
    "workflow",
    "strict_workflow",
    "goal",
    "active_goal",
    "multi_task",
    "active_multi_task",
    "task_plan_rejected",
    "task_planning_transcript",
    "pending_delivery_proposal",
    "pending_goal_proposal",
    "pending_goal_change",
    "revision",
  ] as const;
  exactKeys(session, fields, "session snapshot");
  requiredKeys(session, fields, "session snapshot");
  nonNegativeInteger(session.revision, "session revision");
  requiredString(session.session_id, "session id");
  requiredString(session.task, "session task");
  nullableString(session.title, "session title");
  if (
    typeof session.running !== "boolean" ||
    typeof session.paused !== "boolean" ||
    typeof session.cancel_requested !== "boolean"
  ) {
    throw new Error(
      "session running, paused, and cancellation state must be boolean",
    );
  }
  enumValue(session.status, sessionStatuses, "session status");
  nullableEnum(session.intent, turnIntents, "session intent");
  nullableString(session.branch, "session branch");
  nullableString(session.workdir, "session workdir");
  if (session.project !== null) {
    sessionProject(session.project, "session project");
  }
  nullableEnum(
    session.handoff_outcome,
    handoffOutcomes,
    "session handoff outcome",
  );
  if (session.pending_question !== null) {
    const question = record(
      session.pending_question,
      "session pending question",
    );
    exactKeys(
      question,
      ["question_id", "question", "choices"],
      "session pending question",
    );
    requiredKeys(
      question,
      ["question_id", "question", "choices"],
      "session pending question",
    );
    requiredString(question.question_id, "session pending question id");
    requiredString(question.question, "session pending question");
    if (
      !Array.isArray(question.choices) ||
      question.choices.some((choice) => typeof choice !== "string")
    ) {
      throw new Error("session pending question choices must be strings");
    }
  }
  if (!Array.isArray(session.events)) {
    throw new Error("session events must be an array");
  }
  session.events.forEach(eventEnvelope);
  nonNegativeInteger(session.started_at_ms, "session start time");
  nonNegativeInteger(session.updated_at_ms, "session update time");
  if (session.metrics !== null) {
    sessionMetrics(session.metrics, "session metrics");
  }
  if (!Array.isArray(session.usage_records)) {
    throw new Error("session usage records must be an array");
  }
  session.usage_records.forEach((value, index) =>
    sessionMetrics(value, `session usage record ${index}`)
  );
  if (session.workflow !== null) {
    workflowSummary(session.workflow, "session workflow");
  }
  for (
    const flag of [
      "strict_workflow",
      "active_goal",
      "active_multi_task",
    ] as const
  ) {
    if (typeof session[flag] !== "boolean") {
      throw new Error(`session ${flag} must be boolean`);
    }
  }
  if (session.goal !== null) checkpoint(session.goal, "session goal", "goal");
  if (session.multi_task !== null) {
    checkpoint(session.multi_task, "session multi task", "multi_task");
  }
  if (session.task_plan_rejected !== null) {
    rejectedTaskPlan(session.task_plan_rejected, "session rejected task plan");
  }
  if (session.task_planning_transcript !== null) {
    planningTranscript(
      session.task_planning_transcript,
      "session task planning transcript",
    );
  }
  if (session.pending_delivery_proposal !== null) {
    proposal(
      session.pending_delivery_proposal,
      "session delivery proposal",
      "delivery",
    );
  }
  if (session.pending_goal_proposal !== null) {
    proposal(session.pending_goal_proposal, "session goal proposal", "goal");
  }
  if (session.pending_goal_change !== null) {
    proposal(session.pending_goal_change, "session goal change", "change");
  }
  return session as unknown as SessionDetails;
}

export function parseEventEnvelopeJson(text: string): EventEnvelope {
  return eventEnvelope(json(text, "event envelope"));
}

export function parseSessionDetailsJson(text: string): SessionDetails {
  return sessionDetails(json(text, "session snapshot"));
}

export function parseSessionStreamSnapshotJson(
  text: string,
): SessionStreamSnapshot {
  const snapshot = record(
    json(text, "session stream snapshot"),
    "session stream snapshot",
  );
  exactKeys(
    snapshot,
    ["session", "reset_history"],
    "session stream snapshot",
  );
  requiredKeys(
    snapshot,
    ["session", "reset_history"],
    "session stream snapshot",
  );
  if (typeof snapshot.reset_history !== "boolean") {
    throw new Error("session reset_history must be a boolean");
  }
  return {
    session: sessionDetails(snapshot.session),
    reset_history: snapshot.reset_history,
  };
}

export function parseProjectSessionSnapshotJson(
  text: string,
): ProjectSessionSnapshot {
  const snapshot = record(
    json(text, "project session snapshot"),
    "project session snapshot",
  );
  exactKeys(
    snapshot,
    [
      "stream_id",
      "revision",
      "usage_window_start_ms",
      "usage_window_end_ms",
      "terminal_transition_floor",
      "terminal_transitions",
      "projects",
      "sessions",
      "overall_usage",
      "project_usage",
    ],
    "project session snapshot",
  );
  if (typeof snapshot.stream_id !== "string" || !snapshot.stream_id) {
    throw new Error("project session stream id must be a non-empty string");
  }
  const revision = nonNegativeInteger(
    snapshot.revision,
    "project session revision",
  );
  const usageWindowStart = nonNegativeInteger(
    snapshot.usage_window_start_ms,
    "project usage window start",
  );
  const usageWindowEnd = nonNegativeInteger(
    snapshot.usage_window_end_ms,
    "project usage window end",
  );
  if (
    usageWindowEnd <= usageWindowStart ||
    usageWindowEnd - usageWindowStart > 172_800_000
  ) {
    throw new Error("project usage window must be a positive 48-hour interval");
  }
  const transitionFloor = nonNegativeInteger(
    snapshot.terminal_transition_floor,
    "project session terminal transition floor",
  );
  if (transitionFloor > revision) {
    throw new Error("terminal transition floor exceeds the snapshot revision");
  }
  if (!Array.isArray(snapshot.terminal_transitions)) {
    throw new Error("project session terminal transitions must be an array");
  }
  const transitions = snapshot.terminal_transitions.map(terminalTransition);
  if (
    transitions.some((transition) =>
      transition.revision <= transitionFloor ||
      transition.revision > revision
    )
  ) {
    throw new Error("terminal transition falls outside the snapshot delta");
  }
  if (
    new Set(transitions.map((transition) => transition.entry_key)).size !==
      transitions.length
  ) {
    throw new Error("project session terminal transitions contain duplicates");
  }
  if (
    transitions.some((transition, index) =>
      index > 0 && transitions[index - 1].revision >= transition.revision
    )
  ) {
    throw new Error("project session terminal transitions are out of order");
  }
  if (!Array.isArray(snapshot.projects) || !Array.isArray(snapshot.sessions)) {
    throw new Error("project session collections must be arrays");
  }
  const projects = snapshot.projects.map((value, index) =>
    projectEntry(value, `project ${index}`)
  );
  const sessions = snapshot.sessions.map(sessionItem);
  const projectIds = projects.map((project) => project.id);
  const sessionIds = sessions.map((session) => session.session_id);
  if (new Set(projectIds).size !== projectIds.length) {
    throw new Error("project session snapshot contains duplicate projects");
  }
  if (new Set(sessionIds).size !== sessionIds.length) {
    throw new Error("project session snapshot contains duplicate sessions");
  }
  projectUsageSummary(snapshot.overall_usage, "overall usage summary");
  const projectUsage = record(snapshot.project_usage, "project usage snapshot");
  const usageIds = Object.keys(projectUsage);
  if (
    usageIds.length !== projectIds.length ||
    projectIds.some((projectId) => !Object.hasOwn(projectUsage, projectId))
  ) {
    throw new Error(
      "project usage snapshot must contain every project exactly once",
    );
  }
  for (const projectId of usageIds) {
    if (!projectIds.includes(projectId)) {
      throw new Error(
        `project usage snapshot contains unknown project ${projectId}`,
      );
    }
    projectUsageSummary(projectUsage[projectId], `project ${projectId} usage`);
  }
  return {
    ...snapshot,
    projects,
    sessions,
  } as unknown as ProjectSessionSnapshot;
}
