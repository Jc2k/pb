/// <reference lib="deno.ns" />
import { deepEqual, equal } from "node:assert/strict";
import type { EventEnvelope } from "../types/index";
import {
  buildActionTimeline,
  buildToolSummaries,
  chatEventsWithOnlyLatestStep,
  errorSummary,
  getToolDetail,
  latestAssistantProfile,
  trustedSessionSummaryCommitLines,
} from "./sessionUtils.ts";

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
    event: {
      type: "tool_call",
      tool: "search",
      arguments: { pattern: "branch.*selector", path: "webui" },
    },
  };
  const changes: EventEnvelope = {
    version: "1",
    event: {
      type: "tool_call",
      tool: "session_changes",
      arguments: {},
    },
  };
  const plan: EventEnvelope = {
    version: "1",
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
      event: {
        type: "tool_call",
        tool: "session_title",
        arguments: { title: "Wire title tool" },
        timestamp_ms: 1_782_735_600_000,
      },
    },
    {
      version: "1",
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
      event: {
        type: "tool_call",
        tool: "web_search",
        arguments: { query: "power" },
      },
    },
    {
      version: "1",
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
      event: {
        type: "model_loading",
        model: "/models/local.gguf",
      },
    },
    {
      version: "1",
      event: {
        type: "step_started",
        step: 1,
        max_steps: 20,
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
      event: {
        type: "user_message",
        message_id: "message-1",
        message: "Keep the API stable.",
      },
    },
    {
      version: "v1",
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
      },
    },
  ];

  equal(latestAssistantProfile(events), "build");
});

Deno.test("handoff progress is replaced by the teammate result while raw evidence stays out of chat", () => {
  const events: EventEnvelope[] = [
    {
      version: "v1",
      event: {
        type: "team_message",
        actor: { kind: "automation", id: "handoff" },
        tone: "info",
        message: "I’m checking the API tests.",
      },
    },
    {
      version: "v1",
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
      event: {
        type: "team_message",
        actor: { kind: "automation", id: "handoff" },
        tone: "warning",
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
      event: {
        type: "correction",
        summary: "Kate repeated the same action",
        message: "The duplicate was blocked.",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "build",
      },
    },
    {
      version: "v1",
      event: {
        type: "error",
        summary: "Kate reached the repeat limit",
        message: "No further model turn ran.",
      },
    },
    {
      version: "v1",
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

Deno.test("work-unit progress credits do not split adjacent action runs", () => {
  const visible = chatEventsWithOnlyLatestStep([{
    version: "v1",
    event: {
      type: "correction",
      summary: "Work-unit progress earned one bounded turn",
      message: "Internal bounded turn accounting.",
      actor: { kind: "automation", id: "trinity" },
    },
  }]);
  deepEqual(visible, []);
});
