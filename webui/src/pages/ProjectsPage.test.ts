/// <reference lib="deno.ns" />
import { equal, ok } from "node:assert/strict";
import { nextProjectNotificationPreference } from "./ProjectsPage.tsx";

Deno.test("nextProjectNotificationPreference flips the project notification setting", () => {
  equal(nextProjectNotificationPreference({ notify_on_finish: false }), true);
  equal(nextProjectNotificationPreference({ notify_on_finish: true }), false);
});

Deno.test("project index uses the shared workspace frame", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");

  ok(source.includes('<PageShell contentClassName="projects-index-wrap">'));
  ok(source.includes('<span className="chevron" aria-hidden="true">'));
  ok(!source.includes('className="chevron text-decoration-none"'));
});

Deno.test("project sessions leave branch selection to the managed workspace", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");

  ok(!source.includes("setBranch"));
  ok(!source.includes('className="branch-picker"'));
  ok(!source.includes("feature/ui-refresh"));
  ok(!source.includes("branch: defaultBranch"));
  ok(source.includes('projectSessions[0]?.branch || "Managed automatically"'));
  ok(source.includes("Latest branch"));
  ok(!source.includes("Default branch"));
});

Deno.test("project pages share live session data and finish notifications", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");

  equal(source.match(/useProjectSessionData\(\)/g)?.length, 2);
  ok(!source.includes("useProjectFinishNotifications"));
  ok(!source.includes('fetch("/api/sessions")'));
});
