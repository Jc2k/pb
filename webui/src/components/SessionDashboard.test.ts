import { equal, ok } from "node:assert/strict";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import {
  formatRuntime,
  formatUsageValue,
  SESSION_ROW_BATCH_SIZE,
  sessionCounts,
  SessionRows,
} from "./SessionDashboard.tsx";
import type { SessionItem } from "../types/index.ts";

function session(status: SessionItem["status"]): SessionItem {
  return {
    session_id: `${status}-${Math.random()}`,
    task: "Test session",
    title: null,
    status,
    intent: null,
    branch: "main",
    workdir: "/tmp/project",
    project: { id: "project-1", name: "project", path: "/tmp/project" },
    handoff_outcome: null,
    pending_question: null,
    started_at_ms: Date.now(),
    updated_at_ms: Date.now(),
    workflow_id: null,
    workflow_stage: null,
    workflow_outcome: null,
    strict_workflow: false,
    goal: null,
    active_goal: false,
    multi_task: null,
    active_multi_task: false,
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

Deno.test("session rows reveal long histories in bounded batches", () => {
  const sessions = Array.from(
    { length: SESSION_ROW_BATCH_SIZE + 2 },
    (_, index) => ({
      ...session("completed"),
      session_id: `session-${index}`,
      title: `Session ${index}`,
    }),
  );
  const html = renderToString(createElement(SessionRows, {
    sessions,
    emptyText: "No sessions",
    onOpenSession: () => {},
  }));
  const readableHtml = html.replaceAll("<!-- -->", "");

  equal(
    html.match(/class="session-row project-session-row"/g)?.length,
    SESSION_ROW_BATCH_SIZE,
  );
  ok(readableHtml.includes("Show 2 more sessions"));
  ok(readableHtml.includes("2 remaining"));
});

Deno.test("session rows never invent a default branch", async () => {
  const source = await Deno.readTextFile(
    "webui/src/components/SessionDashboard.tsx",
  );

  ok(source.includes('session.branch || "Managed workspace"'));
  ok(!source.includes('defaultBranch = "main"'));
});

Deno.test("home workspace keeps primary actions focused across breakpoints", async () => {
  const page = await Deno.readTextFile("webui/src/pages/HomePage.tsx");
  const css = await Deno.readTextFile("webui/src/app.css");

  ok(page.includes("counts.all > 0"));
  ok(page.includes('className="quick-action-row"'));
  ok(page.includes('className="quick-action-row secondary-quick-actions"'));
  ok(page.includes('placeholder="Describe the work…"'));
  ok(css.includes(".home-composer-card .secondary-quick-actions"));
  ok(css.includes(".project-detail-wrap .quick-actions"));
  ok(css.includes("overflow-x: visible;"));
  ok(css.includes(".sidebar .nav-pills .nav-link.active"));
  ok(css.includes("background: rgba(0, 122, 255, 0.09);"));
});
