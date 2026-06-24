import { equal } from "node:assert/strict";
import type { EventEnvelope } from "../types";
import {
  getAvatarForProfile,
  groupToolEvents,
  projectName,
  sessionTitle,
} from "./helpers.ts";

Deno.test("sessionTitle prefers a trimmed title and falls back to the task", () => {
  equal(sessionTitle({ title: "  Fix login  ", task: "Investigate auth" }), "Fix login");
  equal(sessionTitle({ title: "   ", task: "Investigate auth" }), "Investigate auth");
  equal(sessionTitle({ title: null, task: "Investigate auth" }), "Investigate auth");
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
