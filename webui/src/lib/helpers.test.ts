/// <reference lib="deno.ns" />
import { equal } from "node:assert/strict";
import type { EventEnvelope } from "../types/index";
import {
  buildChatPresentation,
  CHAT_TIME_GAP_MS,
  getAvatarForProfile,
  groupActionEvents,
  handoffNotificationTitle,
  projectName,
  projectSettingsPath,
  sessionPageDocumentTitle,
  sessionTitle,
  toolResultForCall,
  usageStatsForToday,
} from "./helpers.ts";

Deno.test("buildChatPresentation groups consecutive speakers and marks time gaps", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "reasoning",
        content: "First thought",
        profile: "review",
        timestamp_ms: 1_000,
      },
    },
    {
      version: "1",
      event: {
        type: "llm_invocation",
        step: 1,
        profile: "review",
        duration_ms: 500,
        prompt_tokens: 10,
        generated_tokens: 5,
        timestamp_ms: 2_000,
      },
    },
    {
      version: "1",
      event: {
        type: "final",
        content: "Same speaker",
        profile: "review",
        timestamp_ms: 3_000,
      },
    },
    {
      version: "1",
      event: {
        type: "correction",
        message: "Specific guidance",
        summary: "Repeated tool call detected",
        actor: { kind: "automation", id: "trinity" },
        timestamp_ms: 3_000 + CHAT_TIME_GAP_MS,
      },
    },
    {
      version: "1",
      event: {
        type: "correction",
        message: "More specific guidance",
        summary: "No-progress tool outcome detected",
        actor: { kind: "automation", id: "trinity" },
        timestamp_ms: 4_000 + CHAT_TIME_GAP_MS,
      },
    },
  ];

  const presentation = buildChatPresentation(events);
  equal(presentation[0].showIdentity, true);
  equal(presentation[2].showIdentity, false);
  equal(presentation[3].showIdentity, true);
  equal(presentation[3].timeDividerMs, 3_000 + CHAT_TIME_GAP_MS);
  equal(presentation[4].showIdentity, false);
});

Deno.test("sessionTitle prefers a trimmed title and falls back to the task", () => {
  equal(
    sessionTitle({ title: "  Fix login  ", task: "Investigate auth" }),
    "Fix login",
  );
  equal(
    sessionTitle({ title: "   ", task: "Investigate auth" }),
    "Investigate auth",
  );
  equal(
    sessionTitle({ title: null, task: "Investigate auth" }),
    "Investigate auth",
  );
});

Deno.test("handoff notifications use team outcome language", () => {
  equal(
    handoffNotificationTitle("ready", "completed"),
    "The team wrapped this up",
  );
  equal(
    handoffNotificationTitle("no_change", "completed"),
    "The team left the code untouched",
  );
  equal(
    handoffNotificationTitle("checks_failed", "failed"),
    "This needs another pass",
  );
  equal(
    handoffNotificationTitle("executor_unavailable", "failed"),
    "The team needs help to continue",
  );
  equal(
    handoffNotificationTitle(undefined, "failed"),
    "The task stopped before handoff",
  );
});

Deno.test("sessionPageDocumentTitle follows updated session title", () => {
  equal(
    sessionPageDocumentTitle({ title: "New heading", task: "Old prompt" }),
    "pb session: New heading",
  );
  equal(
    sessionPageDocumentTitle({ title: null, task: "Original prompt" }),
    "pb session: Original prompt",
  );
});

Deno.test("projectName extracts the final path segment across platforms", () => {
  equal(projectName("/workspace/pb"), "pb");
  equal(projectName("C:\\Users\\agent\\project"), "project");
  equal(projectName(), "Unknown project");
});

Deno.test("getAvatarForProfile returns profile avatars only for known profiles", () => {
  equal(getAvatarForProfile("build"), "/static/images/avatar-build.png");
  equal(getAvatarForProfile("monitor"), "/static/images/avatar-monitor.png");
  equal(getAvatarForProfile("unknown"), "/static/images/avatar.png");
});

Deno.test("every named agent profile has a packaged avatar", async () => {
  for (
    const profile of [
      "build",
      "scout",
      "review",
      "explore",
      "plan",
      "ask",
      "research",
      "monitor",
    ]
  ) {
    const avatar = await Deno.stat(
      `public/static/images/avatar-${profile}.png`,
    );
    equal(avatar.isFile, true);
  }
});

Deno.test("groupActionEvents separates profile and steward actions", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: { type: "reasoning", content: "thinking", profile: "build" },
    },
    {
      version: "1",
      event: {
        type: "tool_call",
        tool: "read_file",
        arguments: { path: "Cargo.toml" },
        actor: { kind: "agent", id: "build" },
      },
    },
    {
      version: "1",
      event: {
        type: "tool_result",
        tool: "read_file",
        result: "[package]",
        actor: { kind: "agent", id: "build" },
      },
    },
    {
      version: "1",
      event: {
        type: "controller_closure",
        workflow_id: "workflow-1",
        stage: "implementing",
        reason: "Trusted no-change contract",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "build",
      },
    },
    {
      version: "1",
      event: { type: "final", content: "done", profile: "build" },
    },
  ];

  const grouped = groupActionEvents(events);

  equal(grouped.length, 4);
  equal((grouped[1] as { type: string }).type, "action_group");
  equal(
    (grouped[1] as { actor: { id: string } }).actor.id,
    "build",
  );
  equal((grouped[1] as { toolCalls: EventEnvelope[] }).toolCalls.length, 1);
  equal((grouped[1] as { toolResults: EventEnvelope[] }).toolResults.length, 1);
  equal(
    (grouped[2] as { controllerActions: EventEnvelope[] }).controllerActions
      .length,
    1,
  );
  equal(
    (grouped[2] as { actor: { id: string } }).actor.id,
    "trinity",
  );
  equal((grouped[2] as { assistingProfile: string }).assistingProfile, "build");
});

Deno.test("groupActionEvents presents proactive LSP work as Trinity's routine action", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "tool_call",
        tool: "lsp_proactive_diagnostics",
        arguments: { mode: "syntax", paths: ["src/lib.rs"] },
        actor: { kind: "automation", id: "trinity" },
      },
    },
    {
      version: "1",
      event: {
        type: "tool_result",
        tool: "lsp_proactive_diagnostics",
        result: JSON.stringify({ diagnostics: [] }),
        actor: { kind: "automation", id: "trinity" },
      },
    },
  ];

  const grouped = groupActionEvents(events);
  equal(grouped.length, 1);
  equal((grouped[0] as { actor: { id: string } }).actor.id, "trinity");
  equal((grouped[0] as { toolCalls: EventEnvelope[] }).toolCalls.length, 1);
  equal((grouped[0] as { toolResults: EventEnvelope[] }).toolResults.length, 1);
});

Deno.test("groupActionEvents folds adjacent tool-only inferences by the same teammate into one run", () => {
  const actor = { kind: "agent" as const, id: "plan" };
  const inference = (step: number): EventEnvelope => ({
    version: "1",
    event: {
      type: "llm_invocation",
      step,
      profile: "plan",
      purpose: "workflow_evidence",
      duration_ms: 1000,
      prompt_tokens: 100,
      generated_tokens: 10,
    },
  });
  const events: EventEnvelope[] = [
    inference(1),
    {
      version: "1",
      event: {
        type: "tool_call",
        tool: "glob",
        arguments: { pattern: "webui/**/*.tsx" },
        call_id: "one",
        actor,
      },
    },
    {
      version: "1",
      event: {
        type: "tool_result",
        tool: "glob",
        result: "ProjectsPage.tsx",
        call_id: "one",
        actor,
      },
    },
    {
      version: "1",
      event: {
        type: "tool_batch",
        call_count: 1,
        parallel_safe_count: 1,
        useful_count: 1,
        bookkeeping_only_count: 0,
        rejected_as_dependent: false,
      },
    },
    inference(2),
    {
      version: "1",
      event: {
        type: "tool_call",
        tool: "read_file",
        arguments: { path: "webui/src/pages/ProjectsPage.tsx" },
        call_id: "two",
        actor,
      },
    },
    {
      version: "1",
      event: {
        type: "tool_result",
        tool: "read_file",
        result: "source",
        call_id: "two",
        actor,
      },
    },
    {
      version: "1",
      event: {
        type: "correction",
        message: "Submit the plan now",
        actor: { kind: "automation", id: "trinity" },
      },
    },
  ];

  const grouped = groupActionEvents(events);

  equal(grouped.length, 2);
  const run = grouped[0];
  if (!("type" in run) || run.type !== "action_group") {
    throw new Error("expected one action run");
  }
  equal(run.toolCalls.length, 2);
  equal(run.toolResults.length, 2);
  equal(run.inferenceEvents.length, 2);
  equal((grouped[1] as EventEnvelope).event.type, "correction");
});

Deno.test("groupActionEvents places inference timing after chat-only model work", () => {
  const inference: EventEnvelope = {
    version: "1",
    event: {
      type: "llm_invocation",
      step: 1,
      profile: "review",
      duration_ms: 1000,
      prompt_tokens: 100,
      generated_tokens: 10,
    },
  };
  const reasoning: EventEnvelope = {
    version: "1",
    event: {
      type: "reasoning",
      content: "I found the issue.",
      profile: "review",
    },
  };
  const final: EventEnvelope = {
    version: "1",
    event: {
      type: "final",
      content: "The issue is confirmed.",
      profile: "review",
    },
  };

  const grouped = groupActionEvents([inference, reasoning, final]);

  equal(grouped.length, 3);
  equal((grouped[0] as EventEnvelope).event.type, "reasoning");
  equal((grouped[1] as EventEnvelope).event.type, "final");
  equal((grouped[2] as EventEnvelope).event.type, "llm_invocation");
});

Deno.test("groupActionEvents hides timing when a model call produced no visible work", () => {
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "llm_invocation",
        step: 4,
        profile: "review",
        duration_ms: 30_000,
        prompt_tokens: 100,
        generated_tokens: 10,
      },
    },
    {
      version: "1",
      event: {
        type: "correction",
        summary: "Repeated tool call detected",
        message: "The duplicate action was blocked.",
        actor: { kind: "automation", id: "trinity" },
        assisting_profile: "review",
      },
    },
  ];

  const grouped = groupActionEvents(events);
  equal(grouped.length, 1);
  equal((grouped[0] as EventEnvelope).event.type, "correction");
});

Deno.test("groupActionEvents keeps teammate reasoning visible before its action run", () => {
  const actor = { kind: "agent" as const, id: "build" };
  const events: EventEnvelope[] = [
    {
      version: "1",
      event: {
        type: "llm_invocation",
        step: 1,
        profile: "build",
        duration_ms: 1000,
        prompt_tokens: 100,
        generated_tokens: 10,
      },
    },
    {
      version: "1",
      event: {
        type: "reasoning",
        content: "I will inspect the exact target first.",
        profile: "build",
      },
    },
    {
      version: "1",
      event: {
        type: "tool_call",
        tool: "read_file",
        arguments: { path: "webui/src/pages/ProjectsPage.tsx" },
        call_id: "one",
        actor,
      },
    },
    {
      version: "1",
      event: {
        type: "tool_result",
        tool: "read_file",
        result: "source",
        call_id: "one",
        actor,
      },
    },
  ];

  const grouped = groupActionEvents(events);
  equal(grouped.length, 2);
  equal((grouped[0] as EventEnvelope).event.type, "reasoning");
  const run = grouped[1];
  if (!("type" in run) || run.type !== "action_group") {
    throw new Error("expected one action run");
  }
  equal(run.inferenceEvents.length, 1);
  equal(run.toolCalls.length, 1);
  equal(run.toolResults.length, 1);
});

Deno.test("groupActionEvents correlates reordered identical tools across intervening messages", () => {
  const actor = { kind: "agent" as const, id: "build" };
  const callA: EventEnvelope = {
    version: "1",
    event: {
      type: "tool_call",
      tool: "read_file",
      arguments: { path: "a.rs" },
      call_id: "a",
      batch_id: "batch",
      actor,
    },
  };
  const callB: EventEnvelope = {
    version: "1",
    event: {
      type: "tool_call",
      tool: "read_file",
      arguments: { path: "b.rs" },
      call_id: "b",
      batch_id: "batch",
      actor,
    },
  };
  const correction: EventEnvelope = {
    version: "1",
    event: {
      type: "correction",
      message: "keep going",
      actor: { kind: "automation", id: "trinity" },
    },
  };
  const resultB: EventEnvelope = {
    version: "1",
    event: {
      type: "tool_result",
      tool: "read_file",
      result: "B",
      call_id: "b",
      batch_id: "batch",
      outcome: "succeeded",
      actor,
    },
  };
  const resultA: EventEnvelope = {
    version: "1",
    event: {
      type: "tool_result",
      tool: "read_file",
      result: "A",
      call_id: "a",
      batch_id: "batch",
      outcome: "succeeded",
      actor,
    },
  };

  const grouped = groupActionEvents([
    callA,
    callB,
    correction,
    resultB,
    resultA,
  ]);
  const actions = grouped[0];
  if (!("type" in actions) || actions.type !== "action_group") {
    throw new Error("expected action group");
  }
  const matchedA = toolResultForCall(callA, actions.toolResults);
  const matchedB = toolResultForCall(callB, actions.toolResults);
  equal(matchedA?.event.type, "tool_result");
  equal(
    matchedA?.event.type === "tool_result" ? matchedA.event.result : undefined,
    "A",
  );
  equal(
    matchedB?.event.type === "tool_result" ? matchedB.event.result : undefined,
    "B",
  );
});

Deno.test("projectSettingsPath encodes project names under the project URL", () => {
  equal(projectSettingsPath("my project"), "/projects/my%20project/settings");
  equal(projectSettingsPath("owner/repo"), "/projects/owner%2Frepo/settings");
});

Deno.test("usageStatsForToday sums metrics for sessions updated today", () => {
  const today = new Date("2026-06-26T12:00:00");
  const sessions = [
    {
      session_id: "today",
      task: "Current work",
      running: false,
      paused: false,
      status: "completed" as const,
      updated_at_ms: new Date("2026-06-26T08:30:00").getTime(),
      metrics: {
        llm_invocations: 1,
        llm_runtime_ms: 1000,
        prompt_tokens: 120,
        generated_tokens: 30,
        tool_calls: 2,
        tool_runtime_ms: 500,
        llm_energy_kwh: 0.001,
        tool_energy_kwh: 0.002,
      },
    },
    {
      session_id: "yesterday",
      task: "Old work",
      running: false,
      paused: false,
      status: "completed" as const,
      updated_at_ms: new Date("2026-06-25T23:59:59").getTime(),
      metrics: {
        llm_invocations: 1,
        llm_runtime_ms: 2000,
        prompt_tokens: 400,
        generated_tokens: 100,
        tool_calls: 5,
        tool_runtime_ms: 1000,
      },
    },
  ];

  const stats = usageStatsForToday(sessions, today);

  equal(stats.tokens, 150);
  equal(stats.runtime_ms, 1500);
  equal(stats.tool_calls, 2);
  equal(stats.energy_kwh, 0.003);
});

Deno.test("usageStatsForToday uses per-turn windows and apportions midnight overlap", () => {
  const record = {
    llm_invocations: 1,
    llm_runtime_ms: 120_000,
    prompt_tokens: 80,
    generated_tokens: 20,
    tool_calls: 4,
    tool_runtime_ms: 0,
    wall_runtime_ms: 120_000,
    started_at_ms: new Date("2026-06-25T23:59:00").getTime(),
    ended_at_ms: new Date("2026-06-26T00:01:00").getTime(),
    total_energy_joules: 120,
  };
  const stats = usageStatsForToday([{
    session_id: "midnight",
    task: "Cross midnight",
    running: false,
    paused: false,
    status: "completed",
    updated_at_ms: record.ended_at_ms,
    metrics: record,
    usage_records: [record],
  }], new Date("2026-06-26T12:00:00"));

  equal(stats.tokens, 50);
  equal(stats.runtime_ms, 60_000);
  equal(stats.tool_calls, 2);
  equal(stats.energy_joules, 60);
});
