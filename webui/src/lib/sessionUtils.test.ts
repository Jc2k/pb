/// <reference lib="deno.ns" />
import { deepEqual, equal, ok } from "node:assert/strict";
import type { EventEnvelope } from "../types/index";
import {
  buildActionTimeline,
  buildToolSummaries,
  chatEventsWithOnlyLatestStep,
  errorSummary,
  getToolDetail,
  harnessEfficiencyStats,
  latestAssistantProfile,
  trustedSessionSummaryCommitLines,
} from "./sessionUtils.ts";

Deno.test("harness efficiency uses durable help and prevention evidence", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "controller_observation",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
        receipt: {
          version: 1,
          action_id: "observe-1",
          actual_origin: "controller",
          prompt_representation: "controller_block",
          stage: "code_review",
          operation: "inspect_change",
          path: "src/lib.rs",
          workspace_fingerprint: "workspace",
          path_fingerprint: "path",
          content_sha256: "content",
          coverage: "full",
          observed_bytes: 10,
          prompt_bytes: 10,
          included_ranges: [],
          included_in_prompt: true,
          authority_effects: ["review_coverage"],
        },
      },
    },
    {
      version: "1",
      event: {
        type: "llm_invocation",
        step: 1,
        duration_ms: 100,
        prompt_tokens: 10,
        generated_tokens: 2,
        native: {
          fresh_prefill_tokens: 10,
          cached_tokens: 0,
          prefill_wall_ms: 50,
          prefill_tokens_per_second: 200,
          prefill_metal_commands: 1,
          prefill_host_upload_bytes: 0,
          prefill_host_readback_bytes: 0,
          decode_tokens: 2,
          decode_wall_ms: 50,
          decode_tokens_per_second: 40,
          model_family: "test",
          expert_strategy: "test",
          prefill_command_kind: "test",
          thinking_enabled: false,
          rejected_constraint_candidates: 7,
          mutation_constraint_rejections: { invalid_syntax: 3 },
        },
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "repeated_tool_correction",
      },
      event: {
        type: "correction",
        summary: "Repeated tool call blocked",
        message: "duplicate",
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "dependent_tool_batch_correction",
      },
      event: {
        type: "correction",
        summary: "Dependent tool batch rejected",
        message: "dependent",
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "no_progress_correction",
      },
      event: {
        type: "correction",
        summary: "No-progress tool outcome detected",
        message: "loop",
      },
    },
  ];

  deepEqual(harnessEfficiencyStats(events), {
    proactiveActions: 1,
    proactiveReads: 0,
    proactiveInspections: 1,
    collarCandidatesFiltered: 7,
    mutationCandidatesFiltered: 3,
    duplicateActionsPrevented: 1,
    dependentBatchesPrevented: 1,
    noProgressLoopsStopped: 1,
  });
});

Deno.test("buildActionTimeline preserves chronology and actor provenance", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "tool_call",
        tool: "read_file",
        arguments: { path: "src/lib.rs" },
        actor: { kind: "agent", id: "review" },
      },
    },
    {
      version: "1",
      event: {
        type: "controller_observation",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
        receipt: {
          version: 1,
          action_id: "observe-1",
          actual_origin: "controller",
          prompt_representation: "controller_block",
          stage: "code_review",
          operation: "inspect_change",
          path: "src/lib.rs",
          workspace_fingerprint: "workspace",
          path_fingerprint: "path",
          content_sha256: "content",
          coverage: "full",
          observed_bytes: 10,
          prompt_bytes: 10,
          included_ranges: [],
          included_in_prompt: true,
          authority_effects: ["review_coverage"],
        },
      },
    },
    {
      version: "1",
      event: {
        type: "tool_result",
        tool: "read_file",
        result: "contents",
        actor: { kind: "agent", id: "review" },
      },
    },
  ];

  const timeline = buildActionTimeline(events);
  equal(timeline.length, 2);
  equal(timeline[0].actor?.kind, "agent");
  equal(timeline[0].result?.event.type, "tool_result");
  equal(timeline[1].actor?.kind, "automation");
  equal(timeline[1].assistingProfile, "review");
});

Deno.test("strict workflow summaries require typed commit evidence", () => {
  const workflow: EventEnvelope = {
    version: "1",
    event: {
      type: "workflow_started",
      workflow_id: "run-1",
      source_turn_id: "turn-1",
      policy_sha256: "policy",
    },
  };
  const commit: EventEnvelope = {
    version: "1",
    event: {
      type: "commit_result",
      success: true,
      created: true,
      reused: false,
      oid: "abc123",
    },
  };

  deepEqual(
    trustedSessionSummaryCommitLines("abc old repository history", [workflow]),
    [],
  );
  deepEqual(
    trustedSessionSummaryCommitLines("abc delivery commit", [workflow, commit]),
    ["abc delivery commit"],
  );
  deepEqual(
    trustedSessionSummaryCommitLines("abc legacy summary", []),
    ["abc legacy summary"],
  );
});

Deno.test("getToolDetail shows session_title call title", () => {
  const call: EventEnvelope = {
    version: "1",
    transcript: {
      visibility: "visible",
      kind: "conversation",
      tool_summary: "Wire title tool",
    },
    event: {
      type: "tool_call",
      tool: "session_title",
      arguments: { title: "Wire title tool" },
    },
  };

  equal(getToolDetail(call), "Wire title tool");
});

Deno.test("getToolDetail keeps search scope and summarizes workflow submissions", () => {
  const search: EventEnvelope = {
    version: "1",
    transcript: {
      visibility: "visible",
      kind: "conversation",
      tool_summary: "branch.*selector · in webui",
    },
    event: {
      type: "tool_call",
      tool: "search",
      arguments: { pattern: "branch.*selector", path: "webui" },
    },
  };
  const changes: EventEnvelope = {
    version: "1",
    transcript: {
      visibility: "visible",
      kind: "conversation",
      tool_summary: "Recent sessions and changes",
    },
    event: {
      type: "tool_call",
      tool: "session_changes",
      arguments: {},
    },
  };
  const plan: EventEnvelope = {
    version: "1",
    transcript: {
      visibility: "visible",
      kind: "conversation",
      tool_summary: "1 requirement · 1 step · 1 acceptance check",
    },
    event: {
      type: "tool_call",
      tool: "submit_plan",
      arguments: {
        requirements: [{ id: "r1" }],
        steps: [{ id: "s1" }],
        acceptance: [{ id: "a1" }],
      },
    },
  };
  const incompletePlan: EventEnvelope = {
    version: "1",
    transcript: {
      visibility: "visible",
      kind: "conversation",
      tool_summary: "Incomplete plan · missing required sections",
    },
    event: {
      type: "tool_call",
      tool: "submit_plan",
      arguments: { requirements: [], steps: [], acceptance: [] },
    },
  };

  equal(getToolDetail(search), "branch.*selector · in webui");
  equal(getToolDetail(changes), "Recent sessions and changes");
  equal(
    getToolDetail(plan),
    "1 requirement · 1 step · 1 acceptance check",
  );
  equal(
    getToolDetail(incompletePlan),
    "Incomplete plan · missing required sections",
  );
});

Deno.test("getToolDetail summarizes Trinity's proactive LSP pass", () => {
  const call: EventEnvelope = {
    version: "1",
    event: {
      type: "tool_call",
      tool: "lsp_proactive_diagnostics",
      arguments: { mode: "settled", paths: ["src/lib.rs", "src/main.rs"] },
      actor: { kind: "automation", id: "trinity" },
    },
  };
  const result: EventEnvelope = {
    version: "1",
    transcript: {
      visibility: "visible",
      kind: "conversation",
      tool_summary: "settled · 1 blocking diagnostic in 2 files · 3 deferred",
    },
    event: {
      type: "tool_result",
      tool: "lsp_proactive_diagnostics",
      result: JSON.stringify({
        scanned_paths: ["src/lib.rs", "src/main.rs"],
        diagnostics: [{ path: "src/lib.rs" }],
        failures: [],
        omitted_paths: 3,
        stale: false,
      }),
      actor: { kind: "automation", id: "trinity" },
    },
  };

  equal(
    getToolDetail(call, result),
    "settled · 1 blocking diagnostic in 2 files · 3 deferred",
  );
});

Deno.test("getToolDetail never presents partial LSP coverage as clean", () => {
  const call: EventEnvelope = {
    version: "1",
    event: {
      type: "tool_call",
      tool: "lsp_proactive_diagnostics",
      arguments: { mode: "settled", paths: ["src/lib.rs"] },
      call_id: "lsp-1",
    },
  };
  const result: EventEnvelope = {
    version: "1",
    transcript: {
      visibility: "visible",
      kind: "conversation",
      tool_summary: "settled · incomplete evidence · 1/2 server/file targets",
    },
    event: {
      type: "tool_result",
      tool: "lsp_proactive_diagnostics",
      result: JSON.stringify({
        scanned_paths: ["src/lib.rs"],
        diagnostics: [],
        failures: [],
        omitted_paths: 0,
        stale: false,
        complete: false,
        requested_targets: [{ server: "rust", path: "src/lib.rs" }, {
          server: "second",
          path: "src/lib.rs",
        }],
        completed_targets: [{ server: "rust", path: "src/lib.rs" }],
      }),
      call_id: "lsp-1",
      outcome: "failed",
    },
  };

  equal(
    getToolDetail(call, result),
    "settled · incomplete evidence · 1/2 server/file targets",
  );
});

Deno.test("buildToolSummaries includes session_title parameters in drawer details", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "conversation",
        tool_summary: "Wire title tool",
      },
      event: {
        type: "tool_call",
        tool: "session_title",
        arguments: { title: "Wire title tool" },
        timestamp_ms: 1_782_735_600_000,
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "conversation",
        tool_summary: "Wire title tool",
      },
      event: {
        type: "tool_result",
        tool: "session_title",
        result: "session title set: Wire title tool",
        duration_ms: 4,
      },
    },
  ];

  deepEqual(buildToolSummaries(events), [
    {
      toolName: "session_title",
      friendlyName: "Set title",
      icon: "bi bi-type",
      count: 1,
      items: [
        {
          detail: "Wire title tool · 4 ms",
          timestampMs: 1_782_735_600_000,
        },
      ],
    },
  ]);
});

Deno.test("buildToolSummaries shows joules, power, and parallel measurement scope", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "conversation",
        tool_summary: "power",
      },
      event: {
        type: "tool_call",
        tool: "web_search",
        arguments: { query: "power" },
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "conversation",
        tool_summary: "power",
      },
      event: {
        type: "tool_result",
        tool: "web_search",
        result: "done",
        duration_ms: 1_500,
        energy_joules: 42,
        average_power_watts: 28,
        energy_shared_calls: 3,
      },
    },
  ];

  equal(
    buildToolSummaries(events)[0].items[0].detail,
    "power · 1.5 s · 42.0 J at 28.0 W across 3 parallel calls",
  );
});

Deno.test("chatEventsWithOnlyLatestStep keeps only the current activity indicator", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "started",
        task: "Implement loading state",
        model: "/models/local.gguf",
        workspace: "/repo",
        branch: "feat-loading-state",
        profile: "build",
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "activity",
        kind: "activity",
      },
      event: {
        type: "model_loading",
        model: "/models/local.gguf",
        profile: "build",
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "activity",
        kind: "activity",
      },
      event: {
        type: "step_started",
        step: 1,
        max_steps: 20,
        profile: "build",
      },
    },
  ];

  deepEqual(
    chatEventsWithOnlyLatestStep(events).map((event) => event.event.type),
    ["started", "step_started"],
  );
});

Deno.test("running user messages stay in chat while delivery acknowledgements stay internal", () => {
  const events: EventEnvelope[] = [
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "conversation",
        entry_key: "handoff-progress",
      },
      event: {
        type: "user_message",
        message_id: "message-1",
        message: "Keep the API stable.",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "evidence_only",
        kind: "evidence",
      },
      event: {
        type: "user_message_applied",
        message_id: "message-1",
      },
    },
  ];

  deepEqual(
    chatEventsWithOnlyLatestStep(events).map((event) => event.event.type),
    ["user_message"],
  );
});

Deno.test("chatEventsWithOnlyLatestStep removes session summary text duplicated by final message", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "final",
        content: "Fixed the bug.",
        profile: "build",
      },
    },
    {
      version: "1",
      transcript: {
        visibility: "visible",
        kind: "session_summary",
        summary_redundant: true,
      },
      event: {
        type: "session_summary",
        branch: "fix-duplicate-summary",
        commits: "abc123 fix: avoid duplicate summary",
        summary: " Fixed the bug. ",
      },
    },
  ];

  const summaryEvent = chatEventsWithOnlyLatestStep(events)[1];

  equal(summaryEvent.event.type, "session_summary");
  if (summaryEvent.event.type === "session_summary") {
    equal(summaryEvent.event.summary, undefined);
    equal(summaryEvent.event.commits, "abc123 fix: avoid duplicate summary");
  }
});

Deno.test("chat hides internal closure checkpoints and deduplicates blocked delivery text", () => {
  const events: EventEnvelope[] = [
    {
      version: "v1",
      transcript: {
        visibility: "evidence_only",
        kind: "workflow_closure_checkpoint",
      },
      event: {
        type: "correction",
        summary: "Workflow closure checkpoint",
        message: '{"type":"workflow_closure_checkpoint","stage":"planning"}',
        actor: { kind: "automation", id: "trinity" },
      },
    },
    {
      version: "v1",
      event: {
        type: "workflow_blocked",
        workflow_id: "workflow-1",
        outcome: "plan_rejected",
        reason: "The planning submission was rejected three times.",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "session_summary",
        summary_redundant: true,
      },
      event: {
        type: "session_summary",
        branch: "main",
        commits: "",
        summary: "The planning submission was rejected three times.",
      },
    },
  ];

  const visible = chatEventsWithOnlyLatestStep(events);
  deepEqual(visible.map((event) => event.event.type), [
    "workflow_blocked",
    "session_summary",
  ]);
  const summary = visible[1].event;
  equal(summary.type, "session_summary");
  if (summary.type === "session_summary") equal(summary.summary, undefined);
});

Deno.test("latestAssistantProfile falls back to the started profile for early activity", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "started",
        task: "Implement loading state",
        model: "/models/local.gguf",
        workspace: "/repo",
        branch: "feat-loading-state",
        profile: "build",
      },
    },
    {
      version: "1",
      event: {
        type: "step_started",
        step: 1,
        max_steps: 20,
        profile: "build",
      },
    },
  ];

  equal(latestAssistantProfile(events), "build");
});

Deno.test("handoff progress is replaced by the teammate result while raw evidence stays out of chat", () => {
  const events: EventEnvelope[] = [
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "conversation",
        entry_key: "handoff-progress",
      },
      event: {
        type: "team_message",
        actor: { kind: "automation", id: "handoff" },
        tone: "info",
        purpose: "handoff_progress",
        message: "I’m checking the API tests.",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "evidence_only",
        kind: "evidence",
      },
      event: {
        type: "check_result",
        check_id: "api-test",
        exit_status: 1,
        success: false,
        timed_out: false,
        output: "failed",
        truncated: false,
        duration_ms: 10,
        fingerprint: "input",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "conversation",
        supersedes: ["handoff-progress"],
      },
      event: {
        type: "team_message",
        actor: { kind: "automation", id: "handoff" },
        tone: "warning",
        purpose: "handoff_outcome",
        handoff: {
          outcome: "checks_failed",
          affected_components: ["api"],
          checks: [{ check_id: "api-test", status: "failed" }],
          changed_paths: [],
        },
        message: "The API tests failed. I sent that back to Kate.",
      },
    },
  ];

  deepEqual(
    chatEventsWithOnlyLatestStep(events).map((event) => event.event.type),
    ["team_message"],
  );
  const message = chatEventsWithOnlyLatestStep(events)[0].event;
  equal(message.type, "team_message");
  if (message.type === "team_message") equal(message.tone, "warning");
});

Deno.test("errorSummary prefers explicit error summaries", () => {
  equal(
    errorSummary({
      type: "error",
      summary: "Invalid pb JSON action on step 10/12",
      message:
        "Invalid pb JSON action on step 10/12: failed to parse agent JSON action",
    }),
    "Invalid pb JSON action on step 10/12",
  );
});

Deno.test("terminal repeat errors stay in evidence but collapse into Trinity feedback in chat", () => {
  const events: EventEnvelope[] = [
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "repeated_tool_detected",
        entry_key: "repeat-detected-1",
      },
      event: {
        type: "correction",
        summary: "Repeated tool call detected",
        message: "Choose another action.",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "build",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "repeated_tool_correction",
        entry_key: "repeat-correction-1",
      },
      event: {
        type: "correction",
        summary: "Eugene reached the repeat limit",
        message: "The duplicate was blocked.",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "repeated_tool_detected",
        entry_key: "repeat-detected-2",
      },
      event: {
        type: "correction",
        summary: "Repeated tool call detected",
        message: "Use a different action.",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
      },
    },
    ...Array.from({ length: 6 }, (_, index): EventEnvelope => ({
      version: "v1",
      transcript: {
        visibility: "activity",
        kind: "activity",
      },
      event: {
        type: "step_started",
        step: index + 3,
        max_steps: 8,
        profile: "review",
      },
    })),
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "repeated_tool_correction",
        entry_key: "repeat-correction-2",
      },
      event: {
        type: "correction",
        summary: "Eugene reached the repeat limit",
        message: "The final duplicate was blocked before execution.",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "terminal_tool_loop_error",
        entry_key: "repeat-error",
      },
      event: {
        type: "error",
        summary: "Kate reached the repeat limit",
        message: "No further model turn ran.",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "workflow_blocked",
        supersedes: [
          "repeat-detected-1",
          "repeat-correction-1",
          "repeat-detected-2",
          "repeat-correction-2",
          "repeat-error",
        ],
      },
      event: {
        type: "workflow_blocked",
        workflow_id: "workflow-1",
        outcome: "commit_blocked",
        reason: "model stage Implementing changed Git control state (HEAD)",
      },
    },
  ];

  const visible = chatEventsWithOnlyLatestStep(events);
  deepEqual(visible.map((event) => event.event.type), ["workflow_blocked"]);
});

Deno.test("a repeated failed action keeps one explanation and one terminal outcome", () => {
  const failedRead = JSON.stringify({
    type: "tool_failure",
    tool: "read_file",
    message: "failed to resolve path 'webui/src/components/SessionRows.tsx'",
  });
  const repeatedFailedRead = JSON.stringify({
    type: "tool_failure",
    tool: "read_file",
    message:
      "failed to resolve path 'webui/src/components/SessionRows.tsx': attempt 2",
    action_id: "read-2",
  });
  const events: EventEnvelope[] = [
    {
      version: "v1",
      event: {
        type: "tool_call",
        tool: "read_file",
        arguments: { path: "webui/src/components/SessionRows.tsx" },
        actor: { kind: "agent", id: "review" },
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "correction",
        entry_key: "failed-read-1",
        dedupe_key:
          "tool_failure:read_file:missing:webui/src/components/SessionRows.tsx",
        related_action_key: "tool:read_file:same-path",
      },
      event: {
        type: "correction",
        summary: "read_file failed",
        message: failedRead,
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
      },
    },
    {
      version: "v1",
      event: {
        type: "tool_call",
        tool: "read_file",
        arguments: { path: "webui/src/components/SessionRows.tsx" },
        actor: { kind: "agent", id: "review" },
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "correction",
        entry_key: "failed-read-2",
        dedupe_key:
          "tool_failure:read_file:missing:webui/src/components/SessionRows.tsx",
        related_action_key: "tool:read_file:same-path",
      },
      event: {
        type: "correction",
        summary: "read_file tool call was not executed successfully",
        message: repeatedFailedRead,
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "workflow_blocked",
        supersedes: ["failed-read-2"],
      },
      event: {
        type: "workflow_blocked",
        workflow_id: "workflow-1",
        outcome: "step_limit",
        reason:
          "Eugene stopped making progress and reached a deterministic repeat limit.",
      },
    },
  ];

  const visible = chatEventsWithOnlyLatestStep(events);
  deepEqual(visible.map((event) => event.event.type), [
    "tool_call",
    "correction",
    "tool_call",
    "workflow_blocked",
  ]);
});

Deno.test("no-progress loop errors collapse into the terminal Trinity message", () => {
  const events: EventEnvelope[] = [
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "no_progress_correction",
        entry_key: "no-progress",
      },
      event: {
        type: "correction",
        summary: "No-progress tool outcome detected",
        message: "Use a different action that changes the work unit.",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "build",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "terminal_tool_loop_error",
        entry_key: "no-progress-error",
      },
      event: {
        type: "error",
        summary: "No-progress tool loop",
        message:
          "The same read result reached its deterministic stop threshold.",
      },
    },
    {
      version: "v1",
      transcript: {
        visibility: "visible",
        kind: "workflow_blocked",
        supersedes: ["no-progress", "no-progress-error"],
      },
      event: {
        type: "workflow_blocked",
        workflow_id: "workflow-1",
        outcome: "step_limit",
        reason:
          "Kate stopped making progress in the Implementing stage and reached a deterministic repeat limit.",
      },
    },
  ];

  const visible = chatEventsWithOnlyLatestStep(events);
  deepEqual(visible.map((event) => event.event.type), ["workflow_blocked"]);
});

Deno.test("work-unit progress credits do not split adjacent action runs", () => {
  const visible = chatEventsWithOnlyLatestStep([{
    version: "v1",
    transcript: {
      visibility: "evidence_only",
      kind: "work_unit_progress",
    },
    event: {
      type: "correction",
      summary: "Work-unit progress earned one bounded turn",
      message: "Internal bounded turn accounting.",
      actor: { kind: "automation", id: "trinity" },
    },
  }]);
  deepEqual(visible, []);
});
