/// <reference lib="deno.ns" />
import { equal } from "node:assert/strict";
import type { EventEnvelope } from "../types/index";
import {
  getAvatarForProfile,
  groupToolEvents,
  projectName,
  projectSettingsPath,
  sessionPageDocumentTitle,
  sessionTitle,
  usageStatsForToday,
} from "./helpers.ts";

Deno.test("sessionTitle prefers a trimmed title and falls back to the task", () => {
  equal(sessionTitle({ title: "  Fix login  ", task: "Investigate auth" }), "Fix login");
  equal(sessionTitle({ title: "   ", task: "Investigate auth" }), "Investigate auth");
  equal(sessionTitle({ title: null, task: "Investigate auth" }), "Investigate auth");
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
  equal(getAvatarForProfile("build"), "/avatar-build.png");
  equal(getAvatarForProfile("unknown"), "/avatar.png");
});

Deno.test("groupToolEvents groups contiguous tool calls with their results", () => {
  const events: EventEnvelope[] = [
    { version: "1", event: { type: "reasoning", content: "thinking", profile: "build" } },
    { version: "1", event: { type: "tool_call", tool: "read_file", arguments: { path: "Cargo.toml" } } },
    { version: "1", event: { type: "tool_result", tool: "read_file", result: "[package]" } },
    { version: "1", event: { type: "final", content: "done", profile: "build" } },
  ];

  const grouped = groupToolEvents(events);

  equal(grouped.length, 3);
  equal((grouped[1] as { type: string }).type, "tool_group");
  equal((grouped[1] as { toolCalls: EventEnvelope[] }).toolCalls.length, 1);
  equal((grouped[1] as { toolResults: EventEnvelope[] }).toolResults.length, 1);
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
