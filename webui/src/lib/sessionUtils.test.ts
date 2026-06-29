/// <reference lib="deno.ns" />
import { deepEqual, equal } from "node:assert/strict";
import type { EventEnvelope } from "../types/index";
import { buildToolSummaries, getToolDetail } from "./sessionUtils.ts";

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
