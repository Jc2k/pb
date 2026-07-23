export type AgentEvent =
  | {
    type: "harness_experiment_configured";
    observation_rendering: "native" | "controller_block";
    timestamp_ms?: number;
  }
  | {
    type: "conversation_turn_started";
    turn_id: string;
    intent: TurnIntent;
    task: string;
    timestamp_ms?: number;
  }
  | {
    type: "delivery_proposed";
    proposal_id: string;
    source_turn_id: string;
    task_summary: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_proposed";
    proposal_id: string;
    source_turn_id: string;
    objective: string;
    criteria: GoalCriterionInput[];
    timestamp_ms?: number;
  }
  | {
    type: "goal_started";
    goal_id: string;
    objective: string;
    plan_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_plan_awaiting_approval";
    goal_id: string;
    plan_sha256: string;
    milestones: number;
    timestamp_ms?: number;
  }
  | {
    type: "goal_plan_approved";
    goal_id: string;
    plan_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_milestone_started";
    goal_id: string;
    milestone_id: string;
    title: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_milestone_completed";
    goal_id: string;
    milestone_id: string;
    workflow_id: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_pause_requested" | "goal_paused" | "goal_resumed";
    goal_id: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_amendment_requested";
    goal_id: string;
    amendment_id: string;
    replacement_plan_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_change_requested";
    goal_id: string;
    kind: string;
    summary: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_amendment_resolved";
    goal_id: string;
    amendment_id: string;
    accepted: boolean;
    timestamp_ms?: number;
  }
  | {
    type: "goal_ready_for_review";
    goal_id: string;
    checkpoint_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_completed";
    goal_id: string;
    outcome: GoalOutcome;
    completion_basis: GoalCompletionBasis;
    checkpoint_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_failed";
    goal_id: string;
    outcome: GoalOutcome;
    reason: string;
    checkpoint_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "goal_cancelled";
    goal_id: string;
    checkpoint_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_started";
    workflow_id: string;
    source_turn_id: string;
    policy_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_resumed";
    workflow_id: string;
    stage: WorkflowStage;
    checkpoint_sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_stage_started" | "workflow_stage_completed";
    workflow_id: string;
    stage: WorkflowStage;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_artifact_accepted";
    workflow_id: string;
    artifact_kind: string;
    artifact_id: string;
    sha256: string;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_challenge_raised";
    workflow_id: string;
    challenge_id: string;
    severity: "p0" | "p1" | "p2" | "p3";
    summary: string;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_evidence_invalidated";
    workflow_id: string;
    previous_fingerprint: string;
    current_fingerprint: string;
    reason: string;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_blocked";
    workflow_id: string;
    outcome: WorkflowOutcome;
    reason: string;
    timestamp_ms?: number;
  }
  | {
    type: "workflow_completed";
    workflow_id: string;
    outcome: WorkflowOutcome;
    checkpoint_sha256: string;
    ready_evidence_sha256?: string;
    timestamp_ms?: number;
  }
  | {
    type: "started";
    task: string;
    model: string;
    workspace: string;
    focus_root?: string;
    branch: string;
    attachments?: SessionAttachment[];
    profile: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "step_started";
    step: number;
    max_steps: number;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "model_loading";
    model: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "reasoning";
    content: string;
    profile: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "tool_call";
    tool: string;
    arguments: unknown;
    actor?: TeamActor;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "controller_observation";
    receipt: {
      version: number;
      action_id: string;
      actual_origin: "controller";
      prompt_representation: "controller_block";
      stage: WorkflowStage;
      work_unit_id?: string;
      operation: "read_file" | "inspect_change";
      path: string;
      workspace_fingerprint: string;
      path_fingerprint: string;
      content_sha256: string;
      coverage: "full" | "ranges" | "metadata_only" | "none";
      observed_bytes: number;
      prompt_bytes: number;
      included_ranges: Array<{
        start_byte: number;
        end_byte: number;
        sha256: string;
      }>;
      included_in_prompt: boolean;
      authority_effects: Array<
        "prompt_context" | "read_before_write" | "review_coverage"
      >;
      fallback_reason?: string;
    };
    actor?: TeamActor;
    assisting_profile?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "controller_closure";
    workflow_id: string;
    stage: WorkflowStage;
    reason: string;
    actor?: TeamActor;
    assisting_profile?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "controller_mutation";
    receipt: {
      version: number;
      action_id: string;
      actual_origin: "controller";
      stage: WorkflowStage;
      work_unit_id: string;
      operation: "delete";
      path: string;
      before_workspace_fingerprint: string;
      before_path_fingerprint: string;
      before_content_sha256: string;
      after_workspace_fingerprint: string;
      tracked: boolean;
      adopted: boolean;
      recovery: string;
    };
    actor?: TeamActor;
    assisting_profile?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "tool_batch";
    call_count: number;
    parallel_safe_count: number;
    useful_count: number;
    bookkeeping_only_count: number;
    rejected_as_dependent: boolean;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "tool_result";
    tool: string;
    result: string;
    actor?: TeamActor;
    duration_ms?: number;
    energy_joules?: number;
    energy_kwh?: number;
    average_power_watts?: number;
    energy_shared_calls?: number;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "executor_started";
    executor_id: string;
    kind: string;
    success: boolean;
    detail?: string;
    timestamp_ms?: number;
  }
  | {
    type: "check_result";
    check_id: string;
    exit_status: number;
    success: boolean;
    timed_out: boolean;
    output: string;
    truncated: boolean;
    duration_ms: number;
    fingerprint: string;
    command?: string;
    cwd?: string;
    executor?: string;
    source?: string;
    command_fingerprint?: string;
    dependency_outputs?: Record<string, string>;
    output_fingerprint?: string;
    reused?: boolean;
    skip_reason?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "team_message";
    actor: TeamActor;
    tone: TeamMessageTone;
    message: string;
    detail?: string;
    evidence_ids?: string[];
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "handoff_summary";
    summary: HandoffSummary;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "commit_result";
    success: boolean;
    created: boolean;
    reused: boolean;
    oid?: string;
    subject?: string;
    changed_paths?: string[];
    detail?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "user_question";
    question_id: string;
    question: string;
    choices?: string[];
    profile: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "user_answer";
    question_id: string;
    answer: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "correction";
    message: string;
    summary?: string;
    actor?: TeamActor;
    assisting_profile?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "sub_agent_started";
    profile: string;
    task: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "sub_agent_finished";
    profile: string;
    result: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "diff";
    path: string;
    diff: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "final";
    content: string;
    profile: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "final_grace";
    status: "started" | "accepted" | "rejected";
    detail: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "context_limit";
    context_capacity: number;
    reserved_generation_tokens: number;
    safety_margin_tokens: number;
    usable_prompt_capacity: number;
    measured_prompt_tokens: number;
    largest_sections: Array<{ label: string; chars: number }>;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "llm_invocation";
    step: number;
    duration_ms: number;
    prompt_tokens: number;
    generated_tokens: number;
    prompt_cache?: {
      source: string;
      cached_tokens: number;
      prefilled_tokens: number;
      restore_ms?: number;
    };
    context?: {
      context_capacity: number;
      reserved_generation_tokens: number;
      safety_margin_tokens: number;
      usable_prompt_capacity: number;
      preflight_prompt_tokens: number;
      prompt_utilization_bps: number;
      message_chars: number;
      tool_count: number;
      tool_schema_chars: number;
      tool_schema_tokens?: number;
      thinking_enabled?: boolean;
      retry_reason?:
        | "thinking_off_after_truncation"
        | "compact_mutation_after_truncation"
        | "larger_token_cap_after_truncation";
      compacted_messages: number;
      omitted_tool_result_chars: number;
      read_cache_hits: number;
      closure_checkpoints: number;
      mutation_payload_char_limit?: number;
      serialized_action_chars: number;
      carried_evidence_entries: number;
      carried_evidence_bytes: number;
    };
    native?: {
      fresh_prefill_tokens: number;
      cached_tokens: number;
      prefill_wall_ms: number;
      prefill_tokens_per_second: number;
      prefill_metal_commands: number;
      prefill_host_upload_bytes: number;
      prefill_host_readback_bytes: number;
      decode_tokens: number;
      decode_wall_ms: number;
      decode_tokens_per_second: number;
      model_family: string;
      active_experts_per_token?: number;
      expert_strategy: string;
      prefill_command_kind: string;
      thinking_enabled: boolean;
      tool_constraint_mode?: string;
      tool_schema_sha256?: string;
      rejected_constraint_candidates: number;
      constraint_terminal_state?: string;
    };
    energy_joules?: number;
    energy_kwh?: number;
    average_power_watts?: number;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "session_metrics";
    llm_invocations: number;
    llm_runtime_ms: number;
    prompt_tokens: number;
    generated_tokens: number;
    tool_calls: number;
    tool_runtime_ms: number;
    llm_energy_joules?: number;
    llm_energy_kwh?: number;
    tool_energy_joules?: number;
    tool_energy_kwh?: number;
    wall_runtime_ms?: number;
    started_at_ms?: number;
    ended_at_ms?: number;
    total_energy_joules?: number;
    total_energy_kwh?: number;
    gross_energy_joules?: number;
    adjusted_energy_joules?: number;
    average_power_watts?: number;
    energy_measured_ms?: number;
    energy_coverage?: number;
    energy_source?: string;
    display_energy_excluded?: boolean;
    idle_baseline_applied?: boolean;
    energy_complete?: boolean;
    energy_exclusive?: boolean;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "session_title";
    title: string;
    timestamp_ms?: number;
  }
  | {
    type: "session_summary";
    branch: string;
    commits: string;
    reached_final?: boolean;
    contract_status?: "unspecified" | "unsatisfied" | "satisfied";
    verified_completed?: boolean;
    termination_reason?: string;
    handoff_outcome?: HandoffOutcome;
    summary?: string;
    power_summary?: string;
    diff_stat?: string;
    diff?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "error";
    message: string;
    summary?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  };

export type TeamActor =
  | { kind: "agent"; id: string }
  | { kind: "automation"; id: "trinity" | "handoff" };

export type TeamMessageTone = "info" | "success" | "warning" | "error";

export type HandoffOutcome =
  | "pending"
  | "ready"
  | "no_change"
  | "checks_failed"
  | "executor_unavailable"
  | "commit_blocked"
  | "repair_exhausted"
  | "incomplete";

export interface HandoffSummary {
  outcome: HandoffOutcome;
  affected_components: string[];
  checks: { check_id: string; status: string }[];
  commit?: { oid: string; subject: string };
  changed_paths: string[];
  detail?: string;
}

export interface SessionAttachment {
  name: string;
  mime: string;
  base64?: string;
  id?: string;
  path?: string;
  size?: number;
}

export interface SessionMetricsSnapshot {
  llm_invocations: number;
  llm_runtime_ms: number;
  prompt_tokens: number;
  generated_tokens: number;
  tool_calls: number;
  tool_runtime_ms: number;
  llm_energy_joules?: number;
  llm_energy_kwh?: number;
  tool_energy_joules?: number;
  tool_energy_kwh?: number;
  wall_runtime_ms?: number;
  started_at_ms?: number;
  ended_at_ms?: number;
  total_energy_joules?: number;
  total_energy_kwh?: number;
  gross_energy_joules?: number;
  adjusted_energy_joules?: number;
  average_power_watts?: number;
  energy_measured_ms?: number;
  energy_coverage?: number;
  energy_source?: string;
  display_energy_excluded?: boolean;
  idle_baseline_applied?: boolean;
  energy_complete?: boolean;
  energy_exclusive?: boolean;
}

export interface EventEnvelope {
  version: string;
  event: AgentEvent;
}

export type SessionStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "failed";
export type TurnIntent = "discuss" | "deliver" | "auto";
export type ComposerMode = "discuss" | "deliver" | "goal";
export type GoalStage =
  | "planning"
  | "plan_review"
  | "plan_revision"
  | "awaiting_plan_approval"
  | "running_milestone"
  | "evaluating"
  | "awaiting_user_review"
  | "paused"
  | "blocked"
  | "completed"
  | "failed"
  | "cancelled";
export type GoalOutcome =
  | "complete"
  | "budget_exhausted"
  | "criteria_unsatisfied"
  | "milestone_failed"
  | "repeated_no_progress"
  | "authority_denied"
  | "context_limit"
  | "engine_error"
  | "cancelled";
export type GoalCompletionBasis = "machine_verified" | "user_accepted";
export type GoalContinuationPolicy =
  | "review_plan_then_automatic"
  | "manual_milestones"
  | "automatic_within_limits";
export type GoalVerifier =
  | "workflow_ready"
  | "review_required"
  | "user_confirmation";
export type GoalCriterionStatus =
  | "pending"
  | "evidence_ready"
  | "machine_verified"
  | "user_accepted"
  | "unsatisfied";
export type GoalMilestoneStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "superseded";

export interface GoalBudget {
  max_milestones: number;
  max_workflows: number;
  total_model_invocations: number;
  total_generated_tokens: number;
  wall_time_minutes: number;
}

export interface GoalCriterionInput {
  text: string;
  verifier?: GoalVerifier;
}

export interface GoalCriterionState {
  id: string;
  text: string;
  verifier: GoalVerifier;
  status: GoalCriterionStatus;
  evidence_ids: string[];
}

export interface GoalMilestone {
  id: string;
  title: string;
  description: string;
  criterion_ids: string[];
  plan_version: number;
  status: GoalMilestoneStatus;
  attempts: number;
  workflow?: {
    sha256: string;
    run: {
      counters: {
        model_invocations: number;
        generated_tokens: number;
        advisory_calls: number;
      };
    };
  };
  workflow_summary?: WorkflowSummary;
}

export interface GoalAmendmentDraft {
  id: string;
  base_goal_sha256: string;
  objective: string;
  criteria: GoalCriterionInput[];
  continuation: GoalContinuationPolicy;
  budget: GoalBudget;
  replacement_milestones: GoalMilestone[];
  replacement_plan_sha256: string;
  resume_stage: GoalStage;
}

export interface GoalRun {
  version: number;
  id: string;
  session_id: string;
  objective: string;
  stage: GoalStage;
  plan_version: number;
  plan_sha256: string;
  policy: {
    version: number;
    limits: GoalBudget;
    sha256: string;
  };
  budget: GoalBudget;
  authority: {
    workdir: string;
    publication: false;
  };
  continuation: GoalContinuationPolicy;
  criteria: GoalCriterionState[];
  retired_criteria?: GoalCriterionState[];
  milestones: GoalMilestone[];
  active_milestone_id?: string;
  counters: {
    workflows: number;
    model_invocations: number;
    generated_tokens: number;
    advisory_calls: number;
  };
  pause_requested: boolean;
  paused_stage?: GoalStage;
  blocked_reason?: string;
  pending_amendment?: GoalAmendmentDraft;
  outcome?: GoalOutcome;
  completion_basis?: GoalCompletionBasis;
  created_at_ms: number;
  updated_at_ms: number;
}

export interface GoalCheckpoint {
  sha256: string;
  run: GoalRun;
}

export interface GoalSummary {
  id: string;
  objective: string;
  stage: GoalStage;
  outcome?: GoalOutcome;
  completion_basis?: GoalCompletionBasis;
  completed_milestones: number;
  total_milestones: number;
  current_milestone_title?: string;
  generated_tokens: number;
  generated_token_limit: number;
  workflows: number;
  workflow_limit: number;
  active: boolean;
  plan_sha256: string;
}
export type WorkflowStage =
  | "planning"
  | "plan_review"
  | "plan_revision"
  | "implementing"
  | "checking"
  | "code_review"
  | "repairing"
  | "committing"
  | "ready"
  | "failed"
  | "blocked"
  | "cancelled";
export type WorkflowOutcome =
  | "ready"
  | "no_change"
  | "plan_rejected"
  | "plan_cycles_exhausted"
  | "checks_failed"
  | "review_failed"
  | "repair_cycles_exhausted"
  | "contract_unsatisfied"
  | "executor_unavailable"
  | "commit_blocked"
  | "step_limit"
  | "invocation_limit"
  | "token_limit"
  | "context_limit"
  | "engine_error"
  | "cancelled";

export interface ReadyEvidenceBundle {
  workflow_id: string;
  commit_oid: string;
  plan_sha256: string;
  review_sha256: string;
  check_evidence_ids: string[];
  repository_remote?: string;
}

export interface WorkflowSummary {
  id: string;
  source_turn_id: string;
  task: string;
  stage: WorkflowStage;
  outcome?: WorkflowOutcome;
  policy_sha256: string;
  commit_oid?: string;
  ready_evidence?: ReadyEvidenceBundle;
}

export interface SessionItem {
  session_id: string;
  task: string;
  title?: string | null;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  intent?: TurnIntent;
  branch?: string;
  workdir?: string;
  handoff_outcome?: HandoffOutcome;
  pending_question?: {
    question_id: string;
    question: string;
    choices?: string[];
  };
  updated_at_ms: number;
  metrics?: SessionMetricsSnapshot | null;
  usage_records?: SessionMetricsSnapshot[];
  workflow_id?: string;
  workflow_stage?: WorkflowStage;
  workflow_outcome?: WorkflowOutcome;
  strict_workflow?: boolean;
  goal?: GoalSummary;
  active_goal?: boolean;
}

export interface SessionDetails {
  session_id: string;
  task: string;
  title?: string | null;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  intent?: TurnIntent;
  branch?: string;
  workdir?: string;
  handoff_outcome?: HandoffOutcome;
  pending_question?: {
    question_id: string;
    question: string;
    choices?: string[];
  };
  events: EventEnvelope[];
  updated_at_ms: number;
  metrics?: SessionMetricsSnapshot | null;
  usage_records?: SessionMetricsSnapshot[];
  workflow?: WorkflowSummary;
  strict_workflow?: boolean;
  goal?: GoalCheckpoint;
  active_goal?: boolean;
}

export interface ProjectEntry {
  name: string;
  path: string;
  repository_root?: string;
  notify_on_finish: boolean;
}

export interface ProjectUsageStats {
  tokens: number;
  runtime_ms: number;
  tool_calls: number;
  energy_kwh?: number | null;
  energy_joules?: number | null;
}

export type IntegrationKind = "mcp" | "lsp";

export interface MarketplaceIntegration {
  name: string;
  kind: IntegrationKind;
  description: string;
  icon_url: string;
  repo_url: string;
  container_image: string;
}

export interface InstalledIntegration {
  name: string;
  kind: IntegrationKind;
  container_image: string;
  env?: Record<string, string>;
  disabled: boolean;
}

export type JsonSchemaProperty = {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: string | number | boolean;
  enum?: Array<string | number | boolean>;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
};

export type IntegrationJsonSchema = {
  title?: string;
  description?: string;
  type?: string;
  required?: string[];
  properties?: Record<string, JsonSchemaProperty>;
};

export interface LspPackageServerConfig {
  args: string[];
  language_ids: string[];
  initialization_options?: unknown;
  workspace_access: "read_only";
  network_access: "none";
  cache_ids: string[];
}

export interface LspPackageManifest {
  version: number;
  kind: "lsp";
  server: LspPackageServerConfig;
}

export interface IntegrationConfigSchemaResponse {
  container_image: string;
  annotation: string;
  schema?: IntegrationJsonSchema | null;
  lsp_manifest_annotation: string;
  lsp_manifest?: LspPackageManifest | null;
}

export interface PendingIntegrationInstall {
  kind: IntegrationKind;
  containerImage: string;
  name?: string;
  installed?: boolean;
  env?: Record<string, string>;
}
