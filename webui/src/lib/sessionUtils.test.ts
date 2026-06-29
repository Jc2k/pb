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
