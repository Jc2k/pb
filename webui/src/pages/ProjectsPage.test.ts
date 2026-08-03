/// <reference lib="deno.ns" />
import { deepEqual, equal, ok } from "node:assert/strict";
import {
  currentUsageWindow,
  LatestRequest,
  LatestSubscription,
  ProjectSessionStreamCursor,
  projectSnapshotApplicationScope,
  projectSnapshotMatchesUsageWindow,
} from "../lib/hooks.ts";
import type {
  ProjectSessionSnapshot,
  ProjectSessionTerminalTransition,
} from "../types/index.ts";
import {
  applyProjectSessionMutationResponse,
  nextProjectNotificationPreference,
  projectUsageAvailability,
  recoverProjectSettings,
} from "./ProjectsPage.tsx";

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

Deno.test("latest stream subscription rejects callbacks queued by a closed source", () => {
  const subscriptions = new LatestSubscription();
  const oldSource = subscriptions.start();
  const currentSource = subscriptions.start();

  equal(subscriptions.owns(oldSource), false);
  equal(subscriptions.owns(currentSource), true);
  subscriptions.close(oldSource);
  equal(subscriptions.owns(currentSource), true);
  subscriptions.close(currentSource);
  equal(subscriptions.owns(currentSource), false);
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
    usage_window_start_ms: 1_767_225_600_000,
    usage_window_end_ms: 1_767_312_000_000,
    terminal_transition_floor: terminalTransitionFloor,
    terminal_transitions: terminalTransitions,
    projects: [],
    sessions: [],
    overall_usage: {
      total: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
      today: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
    },
    project_usage: {},
    warnings: [],
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

Deno.test("project stream retains only a bounded set of stale generations", () => {
  const cursor = new ProjectSessionStreamCursor();
  for (let generation = 0; generation < 20; generation += 1) {
    cursor.accept(projectSnapshot(`process-${generation}`, 1), "stream");
  }
  const state = cursor as unknown as { retiredStreamIds: string[] };
  equal(state.retiredStreamIds.length, 8);
  equal(cursor.accept(projectSnapshot("process-18", 2), "stream"), null);
});

Deno.test("project stream trusts server-authored per-connection terminal deltas", () => {
  const cursor = new ProjectSessionStreamCursor();
  const racedWithInitialSnapshot = terminalTransition("fast", 4);
  const initial = cursor.accept(
    projectSnapshot("process-a", 4, 3, [racedWithInitialSnapshot]),
    "stream",
  );
  equal(initial?.terminalTransitions.length, 1);
  equal(initial?.terminalTransitions[0].entry_key, "fast");
  equal(cursor.resumeCursor(), "process-a:4");

  const overlappingConnection = cursor.accept(
    projectSnapshot("process-a", 4, 4),
    "stream",
  );
  equal(overlappingConnection?.terminalTransitions.length, 0);

  const afterReconnect = terminalTransition("reconnected", 5);
  const replay = cursor.accept(
    projectSnapshot("process-a", 5, 4, [afterReconnect]),
    "stream",
  );
  equal(replay?.terminalTransitions[0].entry_key, "reconnected");
});

Deno.test("stale project snapshots cannot replace revisioned data or warnings", () => {
  const cursor = new ProjectSessionStreamCursor();
  equal(
    cursor.accept(projectSnapshot("process-a", 7), "stream")?.applyData,
    true,
  );
  equal(
    cursor.accept(projectSnapshot("process-a", 6), "http")?.applyData,
    false,
  );
});

Deno.test("project usage windows follow the browser's local calendar day", () => {
  const window = currentUsageWindow(new Date(2026, 5, 26, 12));
  const start = new Date(window.start_ms);
  const end = new Date(window.end_ms);
  equal(start.getHours(), 0);
  equal(end.getHours(), 0);
  equal(end.getDate(), start.getDate() + 1);
});

Deno.test("project snapshots cannot cross the requested usage window", () => {
  const snapshot = projectSnapshot("process-a", 4);
  equal(
    projectSnapshotMatchesUsageWindow(snapshot, {
      start_ms: snapshot.usage_window_start_ms,
      end_ms: snapshot.usage_window_end_ms,
    }),
    true,
  );
  equal(
    projectSnapshotMatchesUsageWindow(snapshot, {
      start_ms: snapshot.usage_window_start_ms + 86_400_000,
      end_ms: snapshot.usage_window_end_ms + 86_400_000,
    }),
    false,
  );
  const nextWindow = {
    start_ms: snapshot.usage_window_start_ms + 86_400_000,
    end_ms: snapshot.usage_window_end_ms + 86_400_000,
  };
  equal(projectSnapshotApplicationScope(snapshot, nextWindow, false), "reject");
  equal(
    projectSnapshotApplicationScope(snapshot, nextWindow, true),
    "collection",
  );
});

Deno.test("project settings recovery clears the presented mutation error before refresh", async () => {
  const events: string[] = [];
  await recoverProjectSettings(
    async () => {
      events.push("refresh");
    },
    (message) => events.push(`clear:${message}`),
  );
  deepEqual(events, ["clear:", "refresh"]);
});

Deno.test("project mutations apply their revisioned server snapshot", async () => {
  const applied: string[] = [];
  await applyProjectSessionMutationResponse(
    new Response('{"stream_id":"process-a","revision":3}'),
    (snapshot) => applied.push(snapshot),
  );
  deepEqual(applied, ['{"stream_id":"process-a","revision":3}']);
});

Deno.test("project usage remains visible while a loaded stream reconnects", () => {
  deepEqual(
    projectUsageAvailability(
      true,
      false,
      "Live project updates are temporarily unavailable",
    ),
    { loading: false, error: "" },
  );
  deepEqual(projectUsageAvailability(false, true, ""), {
    loading: true,
    error: "",
  });
  deepEqual(projectUsageAvailability(false, false, "Project request failed"), {
    loading: false,
    error: "Project request failed",
  });
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
  ok(source.includes("currentStatus={projectSessions[0]?.status ?? null}"));
  ok(source.includes(': "No sessions"'));
  ok(!source.includes('projectSessions[0]?.status || "queued"'));
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
  ok(hooks.includes('projectSessionUrl("/api/project-sessions"'));
  ok(hooks.includes("dataRequest.current.start()"));
  ok(hooks.includes("snapshot.projects"));
  ok(hooks.includes("snapshot.sessions"));
  ok(hooks.includes("snapshot.overall_usage"));
  ok(hooks.includes('"/api/project-sessions/events"'));
  ok(hooks.includes('addEventListener("project_session_snapshot"'));
  ok(!hooks.includes("    void fetchData();"));
  ok(hooks.includes("ProjectSessionStreamCursor"));
  ok(hooks.includes("? snapshot.terminal_transitions"));
  ok(!hooks.includes("seenTerminalEntries"));
  ok(!hooks.includes("snapshot.terminal_transitions.filter"));
  ok(!hooks.includes("previous !== session.status"));
  ok(!hooks.includes("setInterval"));
  ok(hooks.includes("dataLoading"));
  ok(hooks.includes("dataError"));
  ok(hooks.includes('setDataError(snapshot.warnings.join(" "))'));
  ok(
    hooks.indexOf("if (!decision.applyData) return;") <
      hooks.indexOf('setDataError(snapshot.warnings.join(" "))'),
  );
  ok(hooks.includes("refresh: fetchData"));
  ok(!hooks.includes("sessionsLoading"));
  ok(!hooks.includes("projectsLoading"));
  ok(!hooks.includes("sessionsError"));
  ok(!hooks.includes("projectsError"));
  ok(!hooks.includes("refreshSessions"));
  ok(!hooks.includes("refreshProjects"));
  ok(!hooks.includes('fetch("/api/sessions"'));
  ok(page.includes("projectUsage[project.id]"));
  ok(!page.includes("usageStatsForToday"));
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
  ok(page.includes("recoverProjectSettings(fetchProjects"));
});

Deno.test("project integration mutations apply their authoritative response", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");
  const mutations = source.slice(
    source.indexOf("const removeIntegration"),
    source.indexOf("const cancelIntegration"),
  );

  equal(mutations.match(/projectMcpIntegrations/g)?.length, 2);
  equal(mutations.match(/setInstalled\(nextInstalled\)/g)?.length, 2);
  equal(mutations.match(/installedRequest\.current\.abort\(\)/g)?.length, 2);
  equal(
    mutations.match(/parseInstalledIntegrationsJson/g)?.length,
    2,
  );
  ok(!mutations.includes("fetchInstalledIntegrations"));
  ok(!mutations.includes(".json()"));
});
