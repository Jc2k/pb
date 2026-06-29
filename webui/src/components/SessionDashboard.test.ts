import { equal } from "node:assert/strict";
import {
  formatRuntime,
  formatUsageValue,
  sessionCounts,
} from "./SessionDashboard.tsx";
import type { SessionItem } from "../types/index.ts";

function session(status: SessionItem["status"]): SessionItem {
  return {
    session_id: `${status}-${Math.random()}`,
    task: "Test session",
    title: null,
    running: status === "running",
    paused: status === "paused",
    status,
    branch: "main",
    workdir: "/tmp/project",
    updated_at_ms: Date.now(),
  };
}

Deno.test("sessionCounts includes aggregate and per-status totals", () => {
  const counts = sessionCounts([
    session("running"),
    session("queued"),
    session("paused"),
    session("completed"),
    session("completed"),
    session("failed"),
  ]);

  equal(counts.all, 6);
  equal(counts.running, 1);
  equal(counts.queued, 1);
  equal(counts.paused, 1);
  equal(counts.completed, 2);
  equal(counts.failed, 1);
});

Deno.test("usage formatters match dashboard display units", () => {
  equal(formatUsageValue(1_240_000), "1.2m");
  equal(formatUsageValue(1284), "1k");
  equal(formatRuntime(8_040_000), "2h 14m");
});
