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

Deno.test("project pages distinguish loading and API failures from empty state", async () => {
  const [page, hooks] = await Promise.all([
    Deno.readTextFile("webui/src/pages/ProjectsPage.tsx"),
    Deno.readTextFile("webui/src/lib/hooks.ts"),
  ]);

  ok(page.includes("projectsLoading"));
  ok(page.includes("projectsError"));
  ok(page.includes("sessionsLoading"));
  ok(page.includes("sessionsError"));
  ok(page.includes("usageLoading"));
  ok(page.includes("usageError"));
  ok(page.includes("submitError"));
  ok(page.indexOf("projectsLoading") < page.indexOf("Project not found."));
  ok(page.includes("Projects may be out of date"));
  ok(page.includes("Project settings may be out of date"));
  ok(page.includes("Loading project settings"));
  ok(page.includes("projectsError && projects.length > 0"));
  ok(hooks.includes("Project request failed"));
  ok(hooks.includes("Session request failed"));
});
