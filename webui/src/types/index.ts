export type AgentEvent =
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
  | { type: "reasoning"; content: string; profile: string; nesting_depth?: number; timestamp_ms?: number }
  | {
      type: "tool_call";
      tool: string;
      arguments: unknown;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "tool_result";
      tool: string;
      result: string;
      duration_ms?: number;
      energy_joules?: number;
      energy_kwh?: number;
      average_power_watts?: number;
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
  | { type: "diff"; path: string; diff: string; nesting_depth?: number; timestamp_ms?: number }
  | { type: "final"; content: string; profile: string; nesting_depth?: number; timestamp_ms?: number }
  | {
      type: "final_grace";
      status: "started" | "accepted" | "rejected";
      detail: string;
      nesting_depth?: number;
      timestamp_ms?: number;
    }
  | {
      type: "llm_invocation";
      step: number;
      duration_ms: number;
      prompt_tokens: number;
      generated_tokens: number;
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
  | { kind: "automation"; id: "handoff" };

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
}

export interface EventEnvelope {
  version: string;
  event: AgentEvent;
}

export type SessionStatus = "queued" | "running" | "paused" | "completed" | "failed";

export interface SessionItem {
  session_id: string;
  task: string;
  title?: string | null;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  branch?: string;
  workdir?: string;
  handoff_outcome?: HandoffOutcome;
  pending_question?: { question_id: string; question: string; choices?: string[] };
  updated_at_ms: number;
  metrics?: SessionMetricsSnapshot | null;
}

export interface SessionDetails {
  session_id: string;
  task: string;
  title?: string | null;
  running: boolean;
  paused: boolean;
  status: SessionStatus;
  branch?: string;
  workdir?: string;
  handoff_outcome?: HandoffOutcome;
  pending_question?: { question_id: string; question: string; choices?: string[] };
  events: EventEnvelope[];
  updated_at_ms: number;
  metrics?: SessionMetricsSnapshot | null;
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

export interface IntegrationConfigSchemaResponse {
  container_image: string;
  annotation: string;
  schema?: IntegrationJsonSchema | null;
}

export interface PendingIntegrationInstall {
  kind: IntegrationKind;
  containerImage: string;
  name?: string;
  installed?: boolean;
  env?: Record<string, string>;
}
