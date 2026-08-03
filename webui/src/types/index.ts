export type AgentProfile =
  | "build"
  | "scout"
  | "review"
  | "explore"
  | "plan"
  | "ask"
  | "research"
  | "monitor";

export type AgentEvent =
  | {
    type: "semantic_gate";
    receipt: SemanticGateReceipt;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
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
    type: "task_plan_accepted";
    multi_task_id: string;
    plan_sha256: string;
    task_count: number;
    timestamp_ms?: number;
  }
  | {
    type: "task_plan_rejected";
    outcome: TaskPlanRejected["outcome"];
    attempts: number;
    timestamp_ms?: number;
  }
  | {
    type: "tasks_changed";
    multi_task_id: string;
    stage: MultiTaskStage;
    outcome?: MultiTaskOutcome;
    active_task_id?: string;
    checkpoint_sha256: string;
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
    kind: GoalChangeKind;
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
    cause: WorkflowBlockCause;
    reason: string;
    current_user?: string;
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
    attachments: SessionAttachment[];
    profile: AgentProfile;
    timestamp_ms?: number;
  }
  | {
    type: "step_started";
    step: number;
    max_steps: number;
    profile: AgentProfile;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "model_loading";
    model: string;
    profile: AgentProfile;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "reasoning";
    content: string;
    profile: AgentProfile;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "tool_call";
    tool: string;
    arguments: unknown;
    call_id: string;
    batch_id: string;
    actor: TeamActor;
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
    actor: TeamActor;
    assisting_profile?: AgentProfile;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "controller_closure";
    workflow_id: string;
    stage: WorkflowStage;
    reason: string;
    actor: TeamActor;
    assisting_profile?: AgentProfile;
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
    actor: TeamActor;
    assisting_profile?: AgentProfile;
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
    call_id: string;
    batch_id: string;
    outcome:
      | "succeeded"
      | "failed"
      | "rejected"
      | "timed_out"
      | "cancelled"
      | "cache_replay";
    actor: TeamActor;
    duration_ms: number;
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
    detail: string;
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
    dependency_outputs: Record<string, string>;
    output_fingerprint?: string;
    reused: boolean;
    skip_reason?: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "team_message";
    actor: TeamActor;
    tone: TeamMessageTone;
    purpose: TeamMessagePurpose;
    handoff?: HandoffSummary;
    message: string;
    detail?: string;
    evidence: EvidenceRef[];
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
    changed_paths: string[];
    detail: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "user_question";
    question_id: string;
    question: string;
    choices: string[];
    profile: AgentProfile;
    timestamp_ms?: number;
  }
  | {
    type: "user_answer";
    question_id: string;
    answer: string;
    timestamp_ms?: number;
  }
  | {
    type: "user_message";
    message_id: string;
    message: string;
    timestamp_ms?: number;
  }
  | {
    type: "user_message_applied";
    message_id: string;
    timestamp_ms?: number;
  }
  | {
    type: "correction";
    message: string;
    kind: CorrectionKind;
    summary: string;
    actor: TeamActor;
    assisting_profile?: AgentProfile;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "sub_agent_started";
    profile: AgentProfile;
    task: string;
    nesting_depth: number;
    timestamp_ms?: number;
  }
  | {
    type: "sub_agent_finished";
    profile: AgentProfile;
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
    profile: AgentProfile;
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
    purpose:
      | "unclassified"
      | "conversation"
      | "task_partitioning"
      | "workflow_planning"
      | "workflow_review"
      | "workflow_evidence"
      | "workflow_mutation"
      | "workflow_closure"
      | "workflow_recovery";
    workflow_stage?: WorkflowStage;
    profile: AgentProfile;
    duration_ms: number;
    prompt_tokens: number;
    generated_tokens: number;
    prompt_cache?: {
      source: string;
      cached_tokens: number;
      prefilled_tokens: number;
      restore_ms: number;
      miss_reason?:
        | "cache_disabled"
        | "cold_session"
        | "prompt_diverged"
        | "stable_prefix_unavailable"
        | "cache_unreadable"
        | "context_reset"
        | "runtime_unsupported";
      lookup_detail?:
        | "session_checkpoint_missing"
        | "session_checkpoint_diverged"
        | "exact_root_checkpoint_missing"
        | "session_diverged_root_missing"
        | "session_diverged_root_hit";
      root?: {
        descriptor_version: number;
        backend: string;
        cache_format_version: string;
        model_namespace_sha256: string;
        rendered_token_sha256: string;
        tokens: number;
        reused_tokens: number;
        system_instruction_version?: string;
        workflow_stage?: WorkflowStage;
        authority_class:
          | "unclassified"
          | "conversation"
          | "task_artifact"
          | "planning"
          | "planning_evidence"
          | "planning_closure"
          | "plan_review"
          | "plan_review_evidence"
          | "plan_review_closure"
          | "implementation_read"
          | "implementation_mutation"
          | "implementation_closure"
          | "repair_read"
          | "repair_mutation"
          | "repair_closure"
          | "code_review"
          | "code_review_evidence"
          | "code_review_closure";
        tool_schema_sha256?: string;
        output_constraint_mode?: string;
      };
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
        | "bounded_read_after_mutation_dead_end"
        | "expanded_mutation_after_payload_limit"
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
      prefill_command_reason: string;
      thinking_enabled: boolean;
      refill?: {
        cache_lookup_wall_ms: number;
        disk_read_decode_wall_ms: number;
        cpu_state_validation_allocation_wall_ms: number;
        state_hydration_wall_ms: number;
        fresh_suffix_prefill_wall_ms: number;
        snapshot_capture_wall_ms: number;
        persistence_queue_wall_ms: number;
      };
      tool_constraint_mode?: string;
      tool_constraint_dialect?: string;
      tool_schema_sha256?: string;
      rejected_constraint_candidates: number;
      mutation_constraint_rejections: Record<string, number>;
      mutation_snapshot_files: number;
      mutation_snapshot_bytes: number;
      constraint_terminal_state?: string;
      constraint_guarantee_rung?: string;
      semantic_boundary?: {
        probes: number;
        allows: number;
        rejects: number;
        defers: number;
        wall_millis: number;
        receipt?: SemanticGateReceipt;
      };
      decode_recovery:
        | "candidate_probe_only"
        | "replay_from_boundary"
        | "snapshot_and_restore";
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
    cache_persistence_queued_checkpoints: number;
    cache_persistence_completed_checkpoints: number;
    cache_persistence_wall_ms: number;
    cache_persistence_failures: number;
    llm_energy_joules?: number;
    llm_energy_kwh?: number;
    tool_energy_joules?: number;
    tool_energy_kwh?: number;
    wall_runtime_ms: number;
    started_at_ms: number;
    ended_at_ms: number;
    total_energy_joules?: number;
    total_energy_kwh?: number;
    gross_energy_joules?: number;
    adjusted_energy_joules?: number;
    average_power_watts?: number;
    energy_measured_ms?: number;
    energy_coverage?: number;
    energy_source?: string;
    display_energy_excluded: boolean;
    idle_baseline_applied: boolean;
    energy_complete: boolean;
    energy_exclusive: boolean;
    nesting_depth?: number;
    timestamp_ms?: number;
  }
  | {
    type: "session_title";
    title: string;
    timestamp_ms?: number;
  }
  | {
    type: "session_state_changed";
    status: SessionStatus;
    running: boolean;
    paused: boolean;
    timestamp_ms?: number;
  }
  | {
    type: "session_summary";
    branch: string;
    commits: HandoffCommitSummary[];
    reached_final: boolean;
    contract_status: "unspecified" | "unsatisfied" | "satisfied";
    verified_completed: boolean;
    termination_reason?: TerminationReason;
    handoff_outcome?: HandoffOutcome;
    summary: string;
    power_summary: string;
    diff_stat: string;
    diff: string;
    timestamp_ms?: number;
  }
  | {
    type: "error";
    summary: string;
    detail: string;
    nesting_depth?: number;
    timestamp_ms?: number;
  };

export type TeamActor =
  | { kind: "agent"; id: AgentProfile }
  | { kind: "automation"; id: "trinity" };

export type TeamMessageTone = "info" | "success" | "warning" | "error";

export type TeamMessagePurpose =
  | "general"
  | "handoff_progress"
  | "handoff_outcome";

export type CorrectionKind =
  | "artifact_validation"
  | "repository_evidence"
  | "contract_evidence"
  | "work_unit"
  | "runtime_fallback"
  | "repeated_tool"
  | "no_progress"
  | "dependent_tool_batch"
  | "stage_submission"
  | "invalid_action"
  | "step_limit"
  | "advisory_budget"
  | "missing_evidence"
  | "truncated_action"
  | "mutation_recovery"
  | "lifecycle"
  | "task_planning_recovery"
  | "tool_unavailable"
  | "requirements_remain"
  | "handoff"
  | "diagnostics"
  | "workflow_closure"
  | "tool_failure";

export type WorkflowBlockCause =
  | "other"
  | "planning_rejected"
  | "git_control_changed"
  | "repository_content_changed"
  | "deterministic_repeat_limit"
  | "executor_unavailable"
  | "commit_blocked"
  | "cancelled";

export type TerminationReason =
  | "final"
  | "step_limit"
  | "gate_loop"
  | "parse_loop"
  | "contract_unsatisfied"
  | "context_limit"
  | "resource_limit"
  | "invocation_limit"
  | "token_limit"
  | "engine_error"
  | "checks_failed"
  | "executor_unavailable"
  | "repair_exhausted"
  | "commit_blocked"
  | "cancelled";

export type ChatterAudience = "team" | "current_user";

export interface EventChatter {
  actor: TeamActor;
  tone: TeamMessageTone;
  headline?: string;
  message: string;
  detail: string;
  audience: ChatterAudience;
}

export type EvidenceRef =
  | { kind: "check"; check_id: string }
  | { kind: "commit"; oid: string };

export interface CheckEvidence {
  check_id: string;
  exit_status: number;
  success: boolean;
  timed_out: boolean;
  output: string;
  duration_ms: number;
  command?: string;
  cwd?: string;
  executor?: string;
  reused: boolean;
  skip_reason?: string;
}

export interface CommitEvidence {
  success: boolean;
  created: boolean;
  reused: boolean;
  oid?: string;
  subject?: string;
  changed_paths: string[];
  detail: string;
}

export type EventEvidence =
  | { kind: "check"; value: CheckEvidence }
  | { kind: "commit"; value: CommitEvidence };

export type TranscriptVisibility = "visible" | "evidence_only" | "activity";

export type TranscriptKind =
  | "conversation"
  | "activity"
  | "evidence"
  | "correction"
  | "repeated_tool_correction"
  | "no_progress_correction"
  | "dependent_tool_batch_correction"
  | "handoff_correction"
  | "workflow_closure_checkpoint"
  | "work_unit_progress"
  | "workflow_blocked"
  | "session_summary";

export interface TranscriptMetadata {
  sequence: number;
  visibility: TranscriptVisibility;
  kind: TranscriptKind;
  entry_key: string;
  supersedes: string[];
  tool_summary?: string;
  dedupe_key?: string;
  related_action_key?: string;
  summary_redundant: boolean;
  session_effect: SessionEffect;
}

export type SessionRunningEffect = "unchanged" | "running" | "stopped";

export interface SessionEffect {
  running: SessionRunningEffect;
  reset_intent: boolean;
  title?: string;
}

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
  commit?: HandoffCommitSummary;
  changed_paths: string[];
  detail?: string;
}

export interface HandoffCommitSummary {
  oid: string;
  subject: string;
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
  cache_persistence_queued_checkpoints: number;
  cache_persistence_completed_checkpoints: number;
  cache_persistence_wall_ms: number;
  cache_persistence_failures: number;
  llm_energy_joules?: number;
  llm_energy_kwh?: number;
  tool_energy_joules?: number;
  tool_energy_kwh?: number;
  wall_runtime_ms: number;
  started_at_ms: number;
  ended_at_ms: number;
  total_energy_joules?: number;
  total_energy_kwh?: number;
  gross_energy_joules?: number;
  adjusted_energy_joules?: number;
  average_power_watts?: number;
  energy_measured_ms?: number;
  energy_coverage?: number;
  energy_source?: string;
  display_energy_excluded: boolean;
  idle_baseline_applied: boolean;
  energy_complete: boolean;
  energy_exclusive: boolean;
}

export interface EventEnvelope {
  version: "v5";
  event: AgentEvent;
  chatter: EventChatter[];
  evidence: EventEvidence[];
  transcript: TranscriptMetadata;
}

export interface SemanticGateReceipt {
  contract_version: number;
  stage: "generation_boundary" | "final_executor";
  scope: "document" | "affected_targets" | "complete_project";
  workspace_sha256: string;
  affected_documents: number;
  providers: SemanticProviderEvidence[];
  viability: "valid" | "repairable" | "impossible" | "unknown";
  closure: "allow" | "reject" | "defer";
  definite_errors: string[];
  unknown_reasons: string[];
  wall_millis: number;
  budget_millis: number;
}

export interface SemanticProviderEvidence {
  provider: string;
  provider_version: string;
  world_sha256: string;
  configuration_sha256: string;
  dependency_sha256: string;
  baseline: "complete" | "incomplete";
  document_count: number;
  introduced_diagnostics: number;
  resolved_diagnostics: number;
  unchanged_diagnostics: number;
  authoritative: boolean;
  definite_errors: string[];
  unknown_reasons: string[];
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

export type TaskKind = "build" | "goal";
export type TaskState =
  | "queued"
  | "running"
  | "committed"
  | "no_change"
  | "blocked"
  | "failed"
  | "cancelled"
  | "superseded";
export type MultiTaskStage =
  | "running_task"
  | "evaluating"
  | "paused"
  | "blocked"
  | "ready"
  | "failed"
  | "cancelled";
export type MultiTaskOutcome =
  | "ready"
  | "task_blocked"
  | "task_failed"
  | "budget_exhausted"
  | "cancelled";

export interface TaskBudget {
  max_workflows: number;
  stage_steps: number;
  total_model_invocations: number;
  total_generated_tokens: number;
  advisory_calls: number;
  plan_cycles: number;
  repair_cycles: number;
  wall_time_minutes: number;
}

export interface TaskCounters {
  workflows: number;
  stage_steps: number;
  model_invocations: number;
  generated_tokens: number;
  advisory_calls: number;
  plan_cycles: number;
  repair_cycles: number;
  elapsed_ms: number;
}

export interface PlannedTask {
  id: string;
  title: string;
  description: string;
  requirement_ids: string[];
  depends_on: string[];
  acceptance_ids: string[];
  scope_hints: string[];
  effort: "small" | "medium" | "large";
  kind: TaskKind;
  budget: TaskBudget;
}

export interface TaskRun {
  spec: PlannedTask;
  revision: number;
  state: TaskState;
  attempts: number;
  counters: TaskCounters;
  blocked_reason?: string;
  result?: {
    commits: string[];
    no_change: boolean;
    summary: string;
  };
}

export interface MultiTaskCheckpoint {
  sha256: string;
  run: {
    id: string;
    stage: MultiTaskStage;
    active_task_id?: string;
    tasks: TaskRun[];
    plan: {
      sha256: string;
      artifact: {
        objective: string;
        tasks: PlannedTask[];
      };
    };
    outcome?: MultiTaskOutcome;
    reason?: string;
    planning_transcript?: TaskPlanningTranscript;
    completion_audit?: TaskCompletionAudit;
  };
}

export type TaskPlanningDecision =
  | "multi_task"
  | "one_build_single_task"
  | "one_build_planner_fallback"
  | "one_build_budget_fallback"
  | "cancelled"
  | "rejected";

export interface TaskPlanningTranscript {
  decision: TaskPlanningDecision;
  summary: string;
  attempts: Array<{
    attempt: number;
    stage: "planner" | "reviewer";
    prompt: string;
    schema: unknown;
    raw_output?: string;
    normalized_output?: unknown;
    failure?: string;
    prompt_tokens: number;
    generated_tokens: number;
    duration_ms: number;
  }>;
}

export interface TaskCompletionAudit {
  plan_sha256: string;
  requirements: Array<{
    requirement_id: string;
    task_ids: string[];
    acceptance_ids: string[];
    evidence_refs: string[];
    commits: string[];
  }>;
  completed_at_ms: number;
}

export interface MultiTaskSummary {
  id: string;
  stage: MultiTaskStage;
  outcome?: MultiTaskOutcome;
  completed_tasks: number;
  total_tasks: number;
  active_task_title?: string;
}

export interface TaskPlanRejected {
  outcome:
    | "attempts_exhausted"
    | "budget_exhausted"
    | "cancelled"
    | "qualification_mismatch";
  attempts: number;
  failures: Array<{
    attempt: number;
    stage: "planner" | "reviewer";
    reason: string;
  }>;
  recovery_actions: Array<
    "retry_planning" | "edit_request" | "run_as_one_build"
  >;
  transcript?: TaskPlanningTranscript;
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

export interface WorkflowArtifactEnvelope<T> {
  id: string;
  sha256: string;
  artifact: T;
}

export interface WorkflowPlanArtifact {
  summary: string;
  requirements: Array<{
    id: string;
    description: string;
    source: string;
  }>;
  steps: Array<{
    id: string;
    requirement_ids: string[];
    component_ids?: string[];
    paths: Array<{
      path: string;
      change: "create" | "modify" | "delete";
    }>;
    description: string;
  }>;
  acceptance: Array<{
    id: string;
    requirement_ids: string[];
    check_ids?: string[];
    description: string;
  }>;
  risks?: Array<{
    id: string;
    description: string;
    mitigation: string;
  }>;
  assumptions?: string[];
  open_questions?: string[];
  resolved_challenge_ids?: string[];
}

export interface WorkflowPlanReviewArtifact {
  plan_id: string;
  plan_sha256: string;
  assessments: Array<{
    kind: string;
    status: "pass" | "concern" | "fail";
    explanation?: string;
  }>;
  challenges: Array<{
    id: string;
    severity: "p0" | "p1" | "p2" | "p3";
    requirement_ids?: string[];
    description: string;
  }>;
  verdict: "pass" | "revise";
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
  paused_stage?: WorkflowStage;
  blocked_reason?: string;
  blocked_cause?: WorkflowBlockCause;
  recovery?: "resume" | "restart_from_current_files";
  plan?: WorkflowArtifactEnvelope<WorkflowPlanArtifact>;
  plan_review?: WorkflowArtifactEnvelope<WorkflowPlanReviewArtifact>;
}

export interface SessionItem {
  session_id: string;
  task: string;
  title: string | null;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  intent: TurnIntent | null;
  branch: string | null;
  workdir: string | null;
  project: SessionProject | null;
  handoff_outcome: HandoffOutcome | null;
  pending_question: {
    question_id: string;
    question: string;
    choices: string[];
  } | null;
  started_at_ms: number;
  updated_at_ms: number;
  metrics: SessionMetricsSnapshot | null;
  workflow_id: string | null;
  workflow_stage: WorkflowStage | null;
  workflow_outcome: WorkflowOutcome | null;
  strict_workflow: boolean;
  goal: GoalSummary | null;
  active_goal: boolean;
  multi_task: MultiTaskSummary | null;
  active_multi_task: boolean;
  revision: number;
}

export interface SessionDetails {
  session_id: string;
  task: string;
  title: string | null;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  intent: TurnIntent | null;
  branch: string | null;
  workdir: string | null;
  project: SessionProject | null;
  handoff_outcome: HandoffOutcome | null;
  pending_question: {
    question_id: string;
    question: string;
    choices: string[];
  } | null;
  events: EventEnvelope[];
  started_at_ms: number;
  updated_at_ms: number;
  metrics: SessionMetricsSnapshot | null;
  usage_records: SessionMetricsSnapshot[];
  workflow: WorkflowSummary | null;
  strict_workflow: boolean;
  goal: GoalCheckpoint | null;
  active_goal: boolean;
  multi_task: MultiTaskCheckpoint | null;
  active_multi_task: boolean;
  task_plan_rejected: TaskPlanRejected | null;
  task_planning_transcript: TaskPlanningTranscript | null;
  pending_delivery_proposal: DeliveryProposal | null;
  pending_goal_proposal: GoalProposal | null;
  pending_goal_change: PendingGoalChange | null;
  revision: number;
}

export interface SessionStreamSnapshot {
  session: SessionDetails;
  reset_history: boolean;
}

export interface SessionProject {
  id: string;
  name: string;
  path: string;
}

export interface DeliveryProposal {
  id: string;
  source_turn_id: string;
  task_summary: string;
}

export interface GoalProposal {
  id: string;
  source_turn_id: string;
  objective: string;
  criteria: GoalCriterionInput[];
}

export interface PendingGoalChange {
  goal_id: string;
  kind: GoalChangeKind;
  summary: string;
}

export type GoalChangeKind = "amendment" | "budget";

export interface ProjectEntry {
  id: string;
  name: string;
  path: string;
  repository_root?: string;
  notify_on_finish: boolean;
}

export interface ProjectSessionSnapshot {
  stream_id: string;
  revision: number;
  usage_window_start_ms: number;
  usage_window_end_ms: number;
  terminal_transition_floor: number;
  terminal_transitions: ProjectSessionTerminalTransition[];
  projects: ProjectEntry[];
  sessions: SessionItem[];
  overall_usage: ProjectUsageSummary;
  project_usage: Record<string, ProjectUsageSummary>;
}

export interface ProjectSessionTerminalTransition {
  entry_key: string;
  revision: number;
  session_id: string;
  status: "completed" | "failed";
  task: string;
  title: string | null;
  handoff_outcome: HandoffOutcome | null;
  project: ProjectEntry | null;
}

export interface ProjectUsageStats {
  tokens: number;
  runtime_ms: number;
  tool_calls: number;
  energy_kwh?: number | null;
  energy_joules?: number | null;
}

export interface ProjectUsageSummary {
  total: ProjectUsageStats;
  today: ProjectUsageStats;
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
  source_container_image?: string;
  verified_manifest_digest?: string;
  env?: Record<string, string>;
  disabled: boolean;
  status?: "ready" | "disabled" | "unavailable" | "legacy_unverified";
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
  source_container_image?: string;
  manifest_digest?: string;
  annotation: string;
  schema?: IntegrationJsonSchema | null;
  lsp_manifest_annotation: string;
  lsp_manifest?: LspPackageManifest | null;
}

export interface PendingIntegrationInstall {
  kind: IntegrationKind;
  containerImage: string;
  sourceContainerImage?: string;
  name?: string;
  installed?: boolean;
  operation?: "install" | "configure" | "upgrade";
  env?: Record<string, string>;
}
