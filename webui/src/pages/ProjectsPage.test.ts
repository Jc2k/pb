/// <reference lib="deno.ns" />
import { equal, ok } from "node:assert/strict";
import { LatestRequest, ProjectSessionStreamCursor } from "../lib/hooks.ts";
import type {
  ProjectSessionSnapshot,
  ProjectSessionTerminalTransition,
} from "../types/index.ts";
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

function terminalTransition(
  entryKey: string,
  revision: number,
): ProjectSessionTerminalTransition {
  return {
    entry_key: entryKey,
    revision,
    session_id: `session-${entryKey}`,
    status: "completed",
    task: "finish the boundary",
    title: "Boundary complete",
    handoff_outcome: "ready",
    project: {
      id: "project-1",
      name: "pb",
      path: "/workspace/pb",
      notify_on_finish: true,
    },
  };
}

function projectSnapshot(
  streamId: string,
  revision: number,
  terminalTransitionFloor = revision,
  terminalTransitions: ProjectSessionTerminalTransition[] = [],
): ProjectSessionSnapshot {
  return {
    stream_id: streamId,
    revision,
    terminal_transition_floor: terminalTransitionFloor,
    terminal_transitions: terminalTransitions,
    projects: [],
    sessions: [],
    project_usage: {},
  };
}

Deno.test("project stream authority rejects stale process generations", () => {
  const cursor = new ProjectSessionStreamCursor();
  equal(
    cursor.accept(projectSnapshot("process-a", 8), "http")?.applyData,
    true,
  );
  equal(
    cursor.accept(projectSnapshot("process-b", 1), "stream")?.applyData,
    true,
  );
  equal(cursor.accept(projectSnapshot("process-a", 9), "http"), null);
  equal(
    cursor.accept(projectSnapshot("process-b", 2), "http")?.applyData,
    true,
  );

  const startup = new ProjectSessionStreamCursor();
  equal(
    startup.accept(projectSnapshot("process-b", 1), "http")?.applyData,
    true,
  );
  equal(
    startup.accept(projectSnapshot("process-a", 4), "stream")?.applyData,
    true,
  );
  equal(
    startup.accept(projectSnapshot("process-b", 2), "stream")?.applyData,
    true,
  );
  equal(startup.accept(projectSnapshot("process-a", 5), "stream"), null);
});

Deno.test("project stream consumes only server-authored terminal transitions after its floor", () => {
  const cursor = new ProjectSessionStreamCursor();
  const preexisting = terminalTransition("old", 3);
  const racedWithInitialSnapshot = terminalTransition("fast", 4);
  const initial = cursor.accept(
    projectSnapshot("process-a", 4, 3, [
      preexisting,
      racedWithInitialSnapshot,
    ]),
    "stream",
  );
  equal(initial?.terminalTransitions.length, 1);
  equal(initial?.terminalTransitions[0].entry_key, "fast");

  const duplicate = cursor.accept(
    projectSnapshot("process-a", 4, 3, [racedWithInitialSnapshot]),
    "stream",
  );
  equal(duplicate?.applyData, false);
  equal(duplicate?.terminalTransitions.length, 0);

  const afterReconnect = terminalTransition("reconnected", 5);
  const replay = cursor.accept(
    projectSnapshot("process-a", 5, 4, [afterReconnect]),
    "stream",
  );
  equal(replay?.terminalTransitions[0].entry_key, "reconnected");
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
  ok(source.includes("useProjectSessionData({ finishNotifications: false })"));
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

  ok(page.includes("dataLoading"));
  ok(page.includes("dataError"));
  ok(page.includes("usageLoading"));
  ok(page.includes("usageError"));
  ok(page.includes("submitError"));
  ok(page.indexOf("dataLoading") < page.indexOf("Project not found."));
  ok(page.includes("Projects may be out of date"));
  ok(page.includes("Project settings may be out of date"));
  ok(page.includes("Loading project settings"));
  ok(page.includes("dataError && projects.length > 0"));
  ok(hooks.includes("Project data request failed"));
  ok(hooks.includes('fetch("/api/project-sessions"'));
  ok(hooks.includes("dataRequest.current.start()"));
  ok(hooks.includes("snapshot.projects"));
  ok(hooks.includes("snapshot.sessions"));
  ok(hooks.includes('new EventSource("/api/project-sessions/events")'));
  ok(hooks.includes('addEventListener("project_session_snapshot"'));
  ok(!hooks.includes("    void fetchData();"));
  ok(hooks.includes("ProjectSessionStreamCursor"));
  ok(hooks.includes("snapshot.terminal_transitions.filter"));
  ok(
    hooks.includes("transition.revision <= snapshot.terminal_transition_floor"),
  );
  ok(!hooks.includes("previous !== session.status"));
  ok(!hooks.includes("setInterval"));
  ok(hooks.includes("dataLoading"));
  ok(hooks.includes("dataError"));
  ok(hooks.includes("refresh: fetchData"));
  ok(!hooks.includes("sessionsLoading"));
  ok(!hooks.includes("projectsLoading"));
  ok(!hooks.includes("sessionsError"));
  ok(!hooks.includes("projectsError"));
  ok(!hooks.includes("refreshSessions"));
  ok(!hooks.includes("refreshProjects"));
  ok(!hooks.includes('fetch("/api/sessions"'));
  ok(page.includes("projectUsage[project.id]"));
  ok(
    !page.includes(
      "fetch(`/api/projects/${encodeURIComponent(project.id)}/usage`",
    ),
  );
  ok(page.includes("marketplaceRequest.current.start()"));
  ok(page.includes("installedRequest.current.start()"));
  ok(page.includes('setInstalledError("")'));
  ok(page.includes('setMarketplaceError("")'));
  ok(!page.includes("setInterval"));
  ok(page.includes("encodeURIComponent(project.id)"));
  ok(!page.includes("encodeURIComponent(project.name)"));
});

Deno.test("project integration mutations apply their authoritative response", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");
  const mutations = source.slice(
    source.indexOf("const removeIntegration"),
    source.indexOf("const cancelIntegration"),
  );

  equal(mutations.match(/projectMcpIntegrations/g)?.length, 2);
  equal(mutations.match(/setInstalled\(nextInstalled\)/g)?.length, 2);
  ok(!mutations.includes("fetchInstalledIntegrations"));
});
