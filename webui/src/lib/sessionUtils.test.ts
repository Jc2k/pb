/// <reference lib="deno.ns" />
import { deepEqual, equal } from "node:assert/strict";
import type { EventEnvelope } from "../types/index";
import {
  buildToolSummaries,
  chatEventsWithOnlyLatestStep,
  errorSummary,
  getToolDetail,
  latestAssistantProfile,
} from "./sessionUtils.ts";

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
    { version: "1", event: { type: "tool_call", tool: "web_search", arguments: { query: "power" } } },
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
      message: "Invalid pb JSON action on step 10/12: failed to parse agent JSON action",
    }),
    "Invalid pb JSON action on step 10/12",
  );
});
