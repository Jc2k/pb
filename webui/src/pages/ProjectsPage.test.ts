/// <reference lib="deno.ns" />
import { equal, ok } from "node:assert/strict";
import { LatestRequest } from "../lib/hooks.ts";
import { nextProjectNotificationPreference } from "./ProjectsPage.tsx";

Deno.test("nextProjectNotificationPreference flips the project notification setting", () => {
  equal(nextProjectNotificationPreference({ notify_on_finish: false }), true);
  equal(nextProjectNotificationPreference({ notify_on_finish: true }), false);
});

Deno.test("latest request ownership aborts and rejects stale responses", () => {
  const requests = new LatestRequest();
  const first = requests.start();
  const second = requests.start();

  equal(first.signal.aborted, true);
  equal(requests.owns(first), false);
  equal(requests.owns(second), true);
  requests.abort();
  equal(second.signal.aborted, true);
  equal(requests.owns(second), false);
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
  ok(source.includes("project_id: project.id"));
  ok(!source.includes("project_name: project.name"));
  ok(!source.includes("workdir: project.path"));
});

Deno.test("project pages share live session data and finish notifications", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");

  equal(source.match(/useProjectSessionData\(\)/g)?.length, 2);
  ok(source.includes("sessionBelongsToProject(session, project)"));
  ok(!source.includes("session.workdir === project.path"));
  ok(!source.includes("useProjectFinishNotifications"));
  ok(!source.includes('fetch("/api/sessions")'));
});

Deno.test("project routes reset drafts and invalidate project-scoped requests", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");
  const projectReset = source.slice(
    source.indexOf('setTask("");'),
    source.indexOf("}, [projectId]);", source.indexOf('setTask("");')),
  );
  const settingsStart = source.indexOf(
    "useEffect(() => {\n    invalidateSchemaRequest();",
    source.indexOf("export function ProjectSettingsPage"),
  );
  const settingsReset = source.slice(
    settingsStart,
    source.indexOf("}, [projectId]);", settingsStart),
  );

  for (
    const reset of [
      'setTask("")',
      'setIntent("discuss")',
      "setGoalOpen(false)",
      "setVoiceInputActive(false)",
      "setImages([])",
      'setFilter("all")',
      'setActiveDetailsTab("usage")',
      "startRequest.current.abort()",
    ]
  ) {
    ok(projectReset.includes(reset), `missing project route reset: ${reset}`);
  }
  for (
    const reset of [
      "invalidateSchemaRequest()",
      "setPendingInstall(null)",
      "setConfigSchema(null)",
      'setSchemaError("")',
      'setIntegrationMutationError("")',
      "notificationRequest.current.abort()",
      "integrationMutationRequest.current.abort()",
    ]
  ) {
    ok(settingsReset.includes(reset), `missing settings route reset: ${reset}`);
  }
  ok(source.includes("disabled={notificationMutationPending}"));
  ok(source.includes("startRequest.current.owns(controller)"));
  ok(source.includes("notificationRequest.current.owns(controller)"));
  ok(source.includes("integrationMutationRequest.current.owns(controller)"));
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
  ok(hooks.includes("Project data request failed"));
  ok(hooks.includes('fetch("/api/project-sessions"'));
  ok(hooks.includes("dataRequest.current.start()"));
  ok(hooks.includes("snapshot.projects"));
  ok(hooks.includes("snapshot.sessions"));
  ok(hooks.includes("window.setInterval(() => void fetchData(), pollMs)"));
  ok(!hooks.includes('fetch("/api/sessions"'));
  ok(page.includes("usageRequest.current.start()"));
  ok(page.includes("marketplaceRequest.current.start()"));
  ok(page.includes("installedRequest.current.start()"));
  ok(page.includes('setInstalledError("")'));
  ok(page.includes('setMarketplaceError("")'));
  ok(page.includes("window.setInterval(() => void fetchProjects(), 5000)"));
  ok(page.includes("encodeURIComponent(project.id)"));
  ok(!page.includes("encodeURIComponent(project.name)"));
});
