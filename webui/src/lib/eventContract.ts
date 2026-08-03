import type {
  EventEnvelope,
  ProjectEntry,
  ProjectSessionSnapshot,
  ProjectSessionTerminalTransition,
  SessionDetails,
  SessionItem,
  SessionStreamSnapshot,
} from "../types";

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
      "revision",
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
  nonNegativeInteger(session.revision, `${label} revision`);
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

function eventEnvelope(value: unknown): EventEnvelope {
  const envelope = record(value, "event envelope");
  if (envelope.version !== "v5") {
    throw new Error(
      `unsupported event schema '${String(envelope.version)}'; expected 'v5'`,
    );
  }
  const event = record(envelope.event, "event payload");
  if (typeof event.type !== "string" || !event.type) {
    throw new Error("event payload type must be a non-empty string");
  }
  if (!Array.isArray(envelope.chatter) || !Array.isArray(envelope.evidence)) {
    throw new Error("event projections must be arrays");
  }
  const transcript = record(envelope.transcript, "event transcript");
  if (nonNegativeInteger(transcript.sequence, "event sequence") < 1) {
    throw new Error("event sequence must be positive");
  }
  if (typeof transcript.entry_key !== "string" || !transcript.entry_key) {
    throw new Error("event entry key must be a non-empty string");
  }
  record(transcript.session_effect, "event session effect");
  return envelope as unknown as EventEnvelope;
}

function sessionDetails(value: unknown): SessionDetails {
  const session = record(value, "session snapshot");
  nonNegativeInteger(session.revision, "session revision");
  if (typeof session.session_id !== "string" || !session.session_id) {
    throw new Error("session id must be a non-empty string");
  }
  if (!Array.isArray(session.events)) {
    throw new Error("session events must be an array");
  }
  session.events.forEach(eventEnvelope);
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
