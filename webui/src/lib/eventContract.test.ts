/// <reference lib="deno.ns" />
import { deepEqual, throws } from "node:assert/strict";
import {
  eventFields,
  parseDeleteSessionMutationResponseJson,
  parseEventEnvelopeJson,
  parseGoalResponseJson,
  parseProjectSessionSnapshotJson,
  parseSessionDetailsJson,
  parseSessionResponseJson,
  parseSessionStreamSnapshotJson,
} from "./eventContract.ts";

function rustEnumVariants(source: string, enumName: string): string[] {
  const declaration = `pub enum ${enumName}`;
  const start = source.indexOf(declaration);
  if (start < 0) throw new Error(`missing Rust enum ${enumName}`);
  const bodyStart = source.indexOf("{", start);
  let depth = 0;
  const variants: string[] = [];
  for (const line of source.slice(bodyStart).split("\n")) {
    const openingDepth = depth;
    depth += (line.match(/\{/g) || []).length;
    depth -= (line.match(/\}/g) || []).length;
    if (openingDepth === 1) {
      const variant = line.match(/^\s{4}([A-Z][A-Za-z0-9_]*)\b/)?.[1];
      if (variant) variants.push(variant);
    }
    if (bodyStart >= 0 && depth === 0) break;
  }
  return variants;
}

function snakeCase(value: string): string {
  return value.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

function typescriptEventVariants(source: string): string[] {
  const start = source.indexOf("export type AgentEvent =");
  const end = source.indexOf("export interface", start);
  const union = source.slice(start, end);
  return [...union.matchAll(/type:\s*((?:"[^"]+"\s*\|\s*)*"[^"]+")/g)]
    .flatMap((match) => [...match[1].matchAll(/"([^"]+)"/g)])
    .map((match) => match[1]);
}

function rustStructFields(source: string, name: string): string[] {
  const start = source.indexOf(`pub struct ${name}`);
  if (start < 0) throw new Error(`missing Rust struct ${name}`);
  const end = source.indexOf("\n}", start);
  return [
    ...source.slice(start, end).matchAll(/^    pub ([a-z_][a-z0-9_]*):/gm),
  ]
    .map((match) => match[1]);
}

function typescriptInterfaceFields(source: string, name: string): string[] {
  const start = source.indexOf(`export interface ${name}`);
  if (start < 0) throw new Error(`missing TypeScript interface ${name}`);
  const end = source.indexOf("\n}", start);
  return [...source.slice(start, end).matchAll(/^  ([a-z_][a-z0-9_]*)\??:/gm)]
    .map((match) => match[1]);
}

function rustEventFields(source: string): Map<string, string[]> {
  const declaration = "pub enum AgentEvent";
  const start = source.indexOf(declaration);
  if (start < 0) throw new Error("missing Rust enum AgentEvent");
  const bodyStart = source.indexOf("{", start);
  const shapes = new Map<string, string[]>();
  let depth = 1;
  let current: string | undefined;
  for (const line of source.slice(bodyStart + 1).split("\n")) {
    if (depth === 1) {
      const variant = line.match(/^    ([A-Z][A-Za-z0-9_]*)\s*\{/)?.[1];
      if (variant) {
        current = snakeCase(variant);
        shapes.set(current, []);
      }
    } else if (depth === 2 && current) {
      const field = line.match(/^        ([a-z_][a-z0-9_]*)\s*:/)?.[1];
      if (field) shapes.get(current)!.push(field);
    }
    depth += (line.match(/\{/g) || []).length;
    depth -= (line.match(/\}/g) || []).length;
    if (depth === 1) current = undefined;
    if (depth === 0) break;
  }
  return shapes;
}

function rustEventOptionalFields(source: string): Map<string, string[]> {
  const declaration = "pub enum AgentEvent";
  const start = source.indexOf(declaration);
  const bodyStart = source.indexOf("{", start);
  const shapes = new Map<string, string[]>();
  let depth = 1;
  let current: string | undefined;
  for (const line of source.slice(bodyStart + 1).split("\n")) {
    if (depth === 1) {
      const variant = line.match(/^    ([A-Z][A-Za-z0-9_]*)\s*\{/)?.[1];
      if (variant) {
        current = snakeCase(variant);
        shapes.set(current, []);
      }
    } else if (depth === 2 && current) {
      const field = line.match(
        /^        ([a-z_][a-z0-9_]*)\s*:\s*Option</,
      )?.[1];
      if (field) shapes.get(current)!.push(field);
    }
    depth += (line.match(/\{/g) || []).length;
    depth -= (line.match(/\}/g) || []).length;
    if (depth === 1) current = undefined;
    if (depth === 0) break;
  }
  return shapes;
}

function typescriptEventFields(source: string): Map<string, string[]> {
  const start = source.indexOf("export type AgentEvent =");
  const end = source.indexOf("export type TeamActor", start);
  if (start < 0 || end < 0) throw new Error("missing TypeScript AgentEvent");
  const shapes = new Map<string, string[]>();
  for (const block of source.slice(start, end).split("\n  | {").slice(1)) {
    const typeDeclaration = block.slice(0, block.indexOf(";"));
    const variants = [...typeDeclaration.matchAll(/"([a-z0-9_]+)"/g)].map(
      (match) => match[1],
    );
    const fields = [...block.matchAll(/^    ([a-z_][a-z0-9_]*)\??\s*:/gm)]
      .map((match) => match[1])
      .filter((field) => field !== "type");
    for (const variant of variants) shapes.set(variant, fields);
  }
  return shapes;
}

function typescriptEventOptionalFields(source: string): Map<string, string[]> {
  const start = source.indexOf("export type AgentEvent =");
  const end = source.indexOf("export type TeamActor", start);
  const shapes = new Map<string, string[]>();
  for (const block of source.slice(start, end).split("\n  | {").slice(1)) {
    const typeDeclaration = block.slice(0, block.indexOf(";"));
    const variants = [...typeDeclaration.matchAll(/"([a-z0-9_]+)"/g)].map(
      (match) => match[1],
    );
    const fields = [...block.matchAll(/^    ([a-z_][a-z0-9_]*)\?\s*:/gm)]
      .map((match) => match[1]);
    for (const variant of variants) shapes.set(variant, fields);
  }
  return shapes;
}

function typescriptStringUnion(source: string, typeName: string): string[] {
  const start = source.indexOf(`export type ${typeName} =`);
  if (start < 0) throw new Error(`missing TypeScript type ${typeName}`);
  const end = source.indexOf(";", start);
  return [...source.slice(start, end).matchAll(/"([^"]+)"/g)].map((match) =>
    match[1]
  );
}

function runtimeContractSet(source: string, name: string): string[] {
  const start = source.indexOf(`const ${name} = new Set([`);
  if (start < 0) throw new Error(`missing runtime contract set ${name}`);
  const end = source.indexOf("]);", start);
  return [...source.slice(start, end).matchAll(/"([^"]+)"/g)].map((match) =>
    match[1]
  );
}

Deno.test("Rust and TypeScript expose the same event and profile variants", async () => {
  const [events, agents, workflow, types, runtimeContract] = await Promise.all([
    Deno.readTextFile("src/events.rs"),
    Deno.readTextFile("src/agent_core.rs"),
    Deno.readTextFile("src/workflow/mod.rs"),
    Deno.readTextFile("webui/src/types/index.ts"),
    Deno.readTextFile("webui/src/lib/eventContract.ts"),
  ]);

  deepEqual(
    typescriptEventVariants(types).sort(),
    rustEnumVariants(events, "AgentEvent").map(snakeCase).sort(),
  );
  const browserFields = typescriptEventFields(types);
  const serverFields = rustEventFields(events);
  const browserOptionalFields = typescriptEventOptionalFields(types);
  const serverOptionalFields = rustEventOptionalFields(events);
  for (const variant of typescriptEventVariants(types)) {
    const browser = browserFields.get(variant)?.toSorted();
    const server = serverFields.get(variant)?.toSorted();
    if (JSON.stringify(browser) !== JSON.stringify(server)) {
      throw new Error(
        `event fields drifted for ${variant}: browser=${
          JSON.stringify(browser)
        } server=${JSON.stringify(server)}`,
      );
    }
    deepEqual(
      browserOptionalFields.get(variant)?.toSorted(),
      serverOptionalFields.get(variant)?.toSorted(),
      `event optionality drifted for ${variant}`,
    );
    const runtime = eventFields[variant as keyof typeof eventFields];
    deepEqual(
      runtime?.[0].toSorted(),
      server?.filter((field) =>
        !serverOptionalFields.get(variant)?.includes(field)
      ).toSorted(),
      `runtime required fields drifted for ${variant}`,
    );
    deepEqual(
      runtime?.[1].toSorted(),
      serverOptionalFields.get(variant)?.toSorted(),
      `runtime optional fields drifted for ${variant}`,
    );
  }
  deepEqual(
    Object.keys(eventFields).toSorted(),
    typescriptEventVariants(types).toSorted(),
    "runtime event variants drifted from AgentEvent",
  );

  const typeProfiles = types
    .slice(
      types.indexOf("export type AgentProfile"),
      types.indexOf("export type AgentEvent"),
    )
    .match(/"([^"]+)"/g)
    ?.map((profile) => profile.slice(1, -1)) || [];
  deepEqual(
    typeProfiles.sort(),
    rustEnumVariants(agents, "AgentProfile").map(snakeCase).sort(),
  );
  deepEqual(
    typescriptStringUnion(types, "SessionStatus").sort(),
    rustEnumVariants(events, "SessionLifecycleStatus").map(snakeCase).sort(),
  );

  for (
    const [rustSource, typeName] of [
      [events, "TeamMessageTone"],
      [events, "TeamMessagePurpose"],
      [events, "CorrectionKind"],
      [events, "GoalChangeKind"],
      [events, "ChatterAudience"],
      [events, "TranscriptVisibility"],
      [events, "TranscriptKind"],
      [events, "HandoffOutcome"],
      [events, "TerminationReason"],
      [workflow, "WorkflowStage"],
      [workflow, "WorkflowOutcome"],
      [workflow, "WorkflowBlockCause"],
    ] as const
  ) {
    deepEqual(
      typescriptStringUnion(types, typeName).sort(),
      rustEnumVariants(rustSource, typeName).map(snakeCase).sort(),
    );
  }

  for (
    const [typeName, contractSet] of [
      ["HandoffOutcome", "handoffOutcomes"],
      ["SessionStatus", "sessionStatuses"],
      ["TurnIntent", "turnIntents"],
      ["WorkflowStage", "workflowStages"],
      ["WorkflowOutcome", "workflowOutcomes"],
      ["GoalStage", "goalStages"],
      ["GoalOutcome", "goalOutcomes"],
      ["GoalCompletionBasis", "goalCompletionBases"],
      ["MultiTaskStage", "multiTaskStages"],
      ["MultiTaskOutcome", "multiTaskOutcomes"],
    ] as const
  ) {
    deepEqual(
      runtimeContractSet(runtimeContract, contractSet).sort(),
      typescriptStringUnion(types, typeName).sort(),
      `${contractSet} drifted from ${typeName}`,
    );
  }

  for (
    const required of [
      "chatter: EventChatter[]",
      "evidence: EventEvidence[]",
      "transcript: TranscriptMetadata",
      "sequence: number",
      "entry_key: string",
      "supersedes: string[]",
      "summary_redundant: boolean",
      "cause: WorkflowBlockCause",
      "purpose: TeamMessagePurpose",
      "kind: CorrectionKind",
      "summary: string",
      "detail: string",
      "call_id: string",
      "batch_id: string",
      "evidence: EvidenceRef[]",
      "commits: HandoffCommitSummary[]",
      "started_at_ms: number",
      "usage_records: SessionMetricsSnapshot[]",
      "reset_history: boolean",
      "warnings: string[]",
      "id: string",
      "stream_id: string",
      "terminal_transition_floor: number",
      "terminal_transitions: ProjectSessionTerminalTransition[]",
      "usage_window_start_ms: number",
      "usage_window_end_ms: number",
      "projects: ProjectEntry[]",
      "sessions: SessionItem[]",
      "revision: number",
      "overall_usage: ProjectUsageSummary",
      "project_usage: Record<string, ProjectUsageSummary>",
      "warnings: string[]",
      "total: ProjectUsageStats",
      "today: ProjectUsageStats",
    ]
  ) {
    if (!types.includes(required)) {
      throw new Error(`missing required v6 event field: ${required}`);
    }
  }
});

Deno.test("v6 browser parsing rejects obsolete event envelopes", () => {
  throws(
    () => parseEventEnvelopeJson('{"version":"v4"}'),
    /unsupported event schema 'v4'; expected 'v6'/,
  );
});

Deno.test("session and goal start responses have exact runtime contracts", () => {
  deepEqual(parseSessionResponseJson('{"session_id":"session-1"}'), {
    session_id: "session-1",
  });
  deepEqual(
    parseGoalResponseJson(
      '{"session_id":"session-1","goal_id":"goal-1","goal_sha256":"abc"}',
    ),
    {
      session_id: "session-1",
      goal_id: "goal-1",
      goal_sha256: "abc",
    },
  );
  throws(
    () => parseSessionResponseJson('{"session_id":"session-1","running":true}'),
    /session start response contains unknown field running/,
  );
  throws(
    () => parseSessionResponseJson("{}"),
    /session start response is missing field session_id/,
  );
  throws(
    () =>
      parseGoalResponseJson(
        '{"session_id":"session-1","goal_id":"goal-1"}',
      ),
    /goal start response is missing field goal_sha256/,
  );
  throws(
    () =>
      parseGoalResponseJson(
        '{"session_id":"session-1","goal_id":"goal-1","goal_sha256":"abc","refresh":true}',
      ),
    /goal start response contains unknown field refresh/,
  );
});

Deno.test("v6 session stream parsing rejects wrapper drift", () => {
  throws(
    () =>
      parseSessionStreamSnapshotJson(
        JSON.stringify({ session: {}, reset_history: true, refresh: true }),
      ),
    /session stream snapshot contains unknown field refresh/,
  );
  throws(
    () => parseSessionStreamSnapshotJson(JSON.stringify({ session: {} })),
    /session stream snapshot is missing field reset_history/,
  );
  throws(
    () =>
      parseSessionStreamSnapshotJson(
        JSON.stringify({ session: {}, reset_history: false }),
      ),
    /session stream snapshot is missing field warnings/,
  );
});

Deno.test("v6 browser parsing rejects incomplete event and session projections", () => {
  const envelope = {
    version: "v6",
    event: { type: "session_metrics" },
    chatter: [],
    evidence: [],
    transcript: {
      sequence: 1,
      visibility: "activity",
      kind: "activity",
      entry_key: "session_metrics:1",
      supersedes: [],
      summary_redundant: false,
    },
  };
  throws(
    () => parseEventEnvelopeJson(JSON.stringify(envelope)),
    /session_metrics is missing field llm_invocations/,
  );
  envelope.event = {
    type: "session_title",
    title: "Boundary",
  } as typeof envelope.event;
  Object.assign(envelope.chatter, [{
    actor: { kind: "automation", id: "trinity" },
    tone: "info",
    message: "Boundary updated",
    detail: "",
  }]);
  throws(
    () => parseEventEnvelopeJson(JSON.stringify(envelope)),
    /event chatter 0 is missing field audience/,
  );
  envelope.event = {
    type: "correction",
    message: "Use the server projection.",
    kind: "lifecycle",
    summary: "Boundary correction",
    actor: { kind: "automation", id: "trinity" },
  } as typeof envelope.event;
  envelope.chatter = [];
  throws(
    () => parseEventEnvelopeJson(JSON.stringify(envelope)),
    /must include exactly one server-authored team chatter projection/,
  );
  envelope.event = {
    type: "workflow_blocked",
    workflow_id: "workflow-1",
    outcome: "executor_unavailable",
    cause: "executor_unavailable",
    reason: "No executor is available.",
  } as typeof envelope.event;
  Object.assign(envelope, {
    chatter: [{
      actor: { kind: "automation", id: "trinity" },
      tone: "warning",
      message: "The workflow is blocked.",
      detail: "No executor is available.",
      audience: "team",
    }],
  });
  throws(
    () => parseEventEnvelopeJson(JSON.stringify(envelope)),
    /must include team and current-user chatter projections/,
  );
  throws(
    () =>
      parseSessionDetailsJson('{"session_id":"s","revision":0,"events":[]}'),
    /session snapshot is missing field task/,
  );
  envelope.event = {
    type: "session_state_changed",
    status: "completed",
    running: true,
    paused: false,
  } as typeof envelope.event;
  envelope.chatter = [];
  throws(
    () => parseEventEnvelopeJson(JSON.stringify(envelope)),
    /event payload session_state_changed contains unknown field running/,
  );
});

Deno.test("project snapshot parsing validates terminal transition semantics", () => {
  const snapshot = {
    stream_id: "process-a",
    revision: 2,
    usage_window_start_ms: 1_767_225_600_000,
    usage_window_end_ms: 1_767_312_000_000,
    terminal_transition_floor: 1,
    terminal_transitions: [{
      entry_key: "event-2",
      revision: 2,
      session_id: "session-1",
      status: "completed",
      task: "finish the event boundary",
      title: "Boundary complete",
      handoff_outcome: "ready",
      project: {
        id: "project-1",
        name: "pb",
        path: "/workspace/pb",
        notify_on_finish: true,
      },
    }],
    projects: [{
      id: "project-1",
      name: "pb",
      path: "/workspace/pb",
      notify_on_finish: true,
    }],
    sessions: [{
      session_id: "session-1",
      task: "finish the event boundary",
      title: "Boundary complete",
      status: "completed",
      intent: "deliver",
      branch: "feature/boundary",
      workdir: "/workspace/pb",
      project: {
        id: "project-1",
        name: "pb",
        path: "/workspace/pb",
      },
      handoff_outcome: "ready",
      pending_question: null,
      started_at_ms: 1,
      updated_at_ms: 2,
      workflow_id: null,
      workflow_stage: null,
      workflow_outcome: null,
      strict_workflow: false,
      goal: null,
      active_goal: false,
      multi_task: null,
      active_multi_task: false,
    }],
    overall_usage: {
      total: { tokens: 3, runtime_ms: 4, tool_calls: 1 },
      today: { tokens: 2, runtime_ms: 3, tool_calls: 1 },
    },
    project_usage: {
      "project-1": {
        total: { tokens: 3, runtime_ms: 4, tool_calls: 1 },
        today: { tokens: 2, runtime_ms: 3, tool_calls: 1 },
      },
    },
    warnings: [],
  };
  deepEqual(
    parseProjectSessionSnapshotJson(JSON.stringify(snapshot)),
    snapshot,
  );
  const contradictoryLifecycle = structuredClone(snapshot);
  Object.assign(contradictoryLifecycle.sessions[0], { running: true });
  throws(
    () =>
      parseProjectSessionSnapshotJson(JSON.stringify(contradictoryLifecycle)),
    /session 0 contains unknown field running/,
  );
  const deletionResponse = {
    deletion: {
      session_id: "session-1",
      deleted: true,
      cleanup_warnings: ["temporary transcript cleanup was deferred"],
    },
    snapshot: {
      ...structuredClone(snapshot),
      revision: 3,
      terminal_transition_floor: 3,
      terminal_transitions: [],
      sessions: [],
      overall_usage: {
        total: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
        today: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
      },
      project_usage: {
        "project-1": {
          total: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
          today: { tokens: 0, runtime_ms: 0, tool_calls: 0 },
        },
      },
    },
  };
  deepEqual(
    parseDeleteSessionMutationResponseJson(JSON.stringify(deletionResponse)),
    deletionResponse,
  );
  const deletionWithUnknownField = structuredClone(deletionResponse);
  Object.assign(deletionWithUnknownField.deletion, { refresh: true });
  throws(
    () =>
      parseDeleteSessionMutationResponseJson(
        JSON.stringify(deletionWithUnknownField),
      ),
    /session deletion result contains unknown field refresh/,
  );
  const deletionWithoutCommittedSnapshot = structuredClone(deletionResponse);
  Reflect.deleteProperty(deletionWithoutCommittedSnapshot, "snapshot");
  throws(
    () =>
      parseDeleteSessionMutationResponseJson(
        JSON.stringify(deletionWithoutCommittedSnapshot),
      ),
    /session deletion response is missing field snapshot/,
  );
  const deletionWithRetainedSession = structuredClone(deletionResponse);
  Object.assign(deletionWithRetainedSession.snapshot, {
    sessions: structuredClone(snapshot.sessions),
  });
  throws(
    () =>
      parseDeleteSessionMutationResponseJson(
        JSON.stringify(deletionWithRetainedSession),
      ),
    /deleted session remains in the committed snapshot/,
  );
  const deletionWithTransitionDelta = structuredClone(deletionResponse);
  deletionWithTransitionDelta.snapshot.terminal_transition_floor = 0;
  Object.assign(deletionWithTransitionDelta.snapshot, {
    terminal_transitions: structuredClone(snapshot.terminal_transitions),
  });
  throws(
    () =>
      parseDeleteSessionMutationResponseJson(
        JSON.stringify(deletionWithTransitionDelta),
      ),
    /session deletion snapshot must suppress terminal transition deltas/,
  );
  const staleTransition = structuredClone(snapshot);
  staleTransition.terminal_transitions[0].revision = 1;
  throws(
    () => parseProjectSessionSnapshotJson(JSON.stringify(staleTransition)),
    /outside the snapshot delta/,
  );
  const unorderedTransitions = structuredClone(snapshot);
  unorderedTransitions.terminal_transitions.push({
    ...unorderedTransitions.terminal_transitions[0],
    entry_key: "event-2-duplicate-revision",
  });
  throws(
    () => parseProjectSessionSnapshotJson(JSON.stringify(unorderedTransitions)),
    /out of order/,
  );
  const missingProjectUsage = structuredClone(snapshot);
  Reflect.deleteProperty(missingProjectUsage.project_usage, "project-1");
  throws(
    () => parseProjectSessionSnapshotJson(JSON.stringify(missingProjectUsage)),
    /contain every project exactly once/,
  );
  const invalidWarnings = structuredClone(snapshot);
  (invalidWarnings as { warnings: unknown[] }).warnings = [42];
  throws(
    () => parseProjectSessionSnapshotJson(JSON.stringify(invalidWarnings)),
    /project session warnings must be strings/,
  );
  const invalidSession = structuredClone(snapshot);
  invalidSession.sessions[0].status = "finished";
  throws(
    () => parseProjectSessionSnapshotJson(JSON.stringify(invalidSession)),
    /session 0 status is invalid/,
  );
  const obsoleteSession = structuredClone(snapshot);
  Object.assign(obsoleteSession.sessions[0], { usage_records: [] });
  throws(
    () => parseProjectSessionSnapshotJson(JSON.stringify(obsoleteSession)),
    /session 0 contains unknown field usage_records/,
  );
  snapshot.terminal_transitions[0].status = "running";
  throws(
    () => parseProjectSessionSnapshotJson(JSON.stringify(snapshot)),
    /status must be completed or failed/,
  );
});

Deno.test("Rust and TypeScript collection structs keep exact fields", async () => {
  const [server, types] = await Promise.all([
    Deno.readTextFile("src/web.rs"),
    Deno.readTextFile("webui/src/types/index.ts"),
  ]);
  for (
    const name of [
      "ProjectSessionSnapshot",
      "ProjectSessionTerminalTransition",
      "ProjectUsageSummary",
      "SessionStreamSnapshot",
      "SessionResponse",
      "GoalResponse",
      "DeleteSessionResponse",
      "DeleteSessionMutationResponse",
    ]
  ) {
    deepEqual(
      typescriptInterfaceFields(types, name).sort(),
      rustStructFields(server, name).sort(),
      `${name} fields drifted across the Rust/TypeScript boundary`,
    );
  }
  deepEqual(
    typescriptInterfaceFields(types, "SessionItem").sort(),
    rustStructFields(server, "SessionListItem").sort(),
    "SessionItem fields drifted from the Rust list projection",
  );
  deepEqual(
    typescriptInterfaceFields(types, "SessionDetails").sort(),
    rustStructFields(server, "SessionDetails").sort(),
    "SessionDetails fields drifted across the Rust/TypeScript boundary",
  );
});

Deno.test("session and project stream boundaries are server-authored", async () => {
  const [server, types, sessionPage, hooks] = await Promise.all([
    Deno.readTextFile("src/web.rs"),
    Deno.readTextFile("webui/src/types/index.ts"),
    Deno.readTextFile("webui/src/pages/SessionPage.tsx"),
    Deno.readTextFile("webui/src/lib/hooks.ts"),
  ]);

  for (
    const required of [
      'event("session_snapshot")',
      "reset_history: bool",
      "warnings: Vec<String>",
      'route("/api/sessions/{id}/goal", post(start_session_goal))',
      "pub struct ProjectSessionSnapshot",
      "pub stream_id: String",
      "pub usage_window_start_ms: u64",
      "pub usage_window_end_ms: u64",
      "pub terminal_transition_floor: u64",
      "pub terminal_transitions: Vec<ProjectSessionTerminalTransition>",
      "pub warnings: Vec<String>",
      'route("/api/project-sessions", get(list_project_sessions))',
      '"/api/project-sessions/events"',
      '.event("project_session_snapshot")',
      "pub project_id: Option<String>",
    ]
  ) {
    if (!server.includes(required)) {
      throw new Error(`missing server boundary contract: ${required}`);
    }
  }
  for (
    const required of [
      "export interface SessionStreamSnapshot",
      "reset_history: boolean",
      "warnings: string[]",
      "cancel_requested: boolean",
      "export interface ProjectSessionSnapshot",
    ]
  ) {
    if (!types.includes(required)) {
      throw new Error(`missing browser boundary contract: ${required}`);
    }
  }
  if (!sessionPage.includes('addEventListener("session_snapshot"')) {
    throw new Error("session page does not consume server snapshots");
  }
  if (!server.includes("Result<Json<SessionStreamSnapshot>, ApiError>")) {
    throw new Error(
      "existing-session mutations do not return the stream snapshot projection",
    );
  }
  if (!hooks.includes('projectSessionUrl("/api/project-sessions"')) {
    throw new Error("project pages do not consume the atomic server snapshot");
  }
  const notificationMutation = server.slice(
    server.indexOf("async fn update_project_notifications("),
    server.indexOf("async fn get_settings("),
  );
  if (
    !notificationMutation.includes(".commit_project_registry_owned(") ||
    !notificationMutation.includes("usage_window") ||
    notificationMutation.includes("project_session_snapshot(&state")
  ) {
    throw new Error(
      "project notification mutation reconstructs its collection receipt after commit",
    );
  }
  if (
    !hooks.includes('"/api/project-sessions/events"') ||
    !hooks.includes("projectSessionUrl(") ||
    hooks.includes("setInterval")
  ) {
    throw new Error(
      "project pages do not consume the pushed collection stream",
    );
  }
  if (
    server.includes("pub refresh: bool") || types.includes("refresh: boolean")
  ) {
    throw new Error(
      "session snapshot scheduling leaked into the wire contract",
    );
  }
});

Deno.test("session writes and terminal frames have one type-enforced boundary", async () => {
  const [server, store, protocol, client, events, sessionPage] = await Promise
    .all([
      Deno.readTextFile("src/web.rs"),
      Deno.readTextFile("src/session_store.rs"),
      Deno.readTextFile("src/daemon_protocol.rs"),
      Deno.readTextFile("src/daemon_client.rs"),
      Deno.readTextFile("src/events.rs"),
      Deno.readTextFile("webui/src/pages/SessionPage.tsx"),
    ]);
  for (
    const required of [
      "struct SessionRecord",
      "struct SessionRuntime",
      "session.record = staged.record;",
      "commit_session_change_owned",
      "commit_session_change_owned_after",
      "create_session_owned",
      "claim_next_session_owned",
      "delete_session_owned",
      "pub struct ProjectMutationReceipt",
      "commit_project_registry_owned",
      "publish_project_session_reconciliation",
      "SessionPersistenceError",
      "pub struct SessionMutationReceipt",
      "pub struct GoalMutationReceipt",
      "dispatch_unary_rpc",
      "enum SessionStreamPublication",
      "SessionStreamPublication::Finished",
      "publish_committed",
    ]
  ) {
    if (!server.includes(required)) {
      throw new Error(`missing atomic session boundary: ${required}`);
    }
  }
  for (
    const obsolete of [
      "persist_session_snapshot",
      "persist_live_session",
      "persist_authoritative_session",
      "project_session_publisher",
      "CollectionPublication",
      "struct RpcResponse",
      "struct RpcNotification",
      "fn publish_collection(mut self)",
      "StagedSessionChange::terminal",
      ".dispatch()",
      "fn publish_session_state_changed",
    ]
  ) {
    if (server.includes(obsolete)) {
      throw new Error(`obsolete protocol workaround returned: ${obsolete}`);
    }
  }
  const coordinatorEnd = server.indexOf(
    "// `deno task build:web` refreshes these assets",
  );
  const testsStart = server.indexOf("#[cfg(test)]");
  if (coordinatorEnd < 0 || testsStart < 0) {
    throw new Error("could not locate the server coordinator boundary");
  }
  const productionOutsideCoordinator = server.slice(coordinatorEnd, testsStart);
  if (
    /sessions\s*\.\s*(?:get_mut|values_mut|insert|remove)\b/.test(
      productionOutsideCoordinator,
    )
  ) {
    throw new Error(
      "production session collection mutation bypasses AppState's coordinator",
    );
  }
  if (!store.includes("pub(crate) trait SessionRepository")) {
    throw new Error("session persistence is not injectable");
  }
  if (
    store.includes("pub fn save_session") ||
    store.includes("pub fn delete_session")
  ) {
    throw new Error(
      "direct production session persistence entry point returned",
    );
  }
  const sessionStream = server.slice(
    server.indexOf("async fn session_events("),
    server.indexOf("fn session_event_sse_event("),
  );
  if (
    sessionStream.includes("session_details_snapshot") ||
    !sessionStream.includes("SessionStreamPublication::Snapshot")
  ) {
    throw new Error(
      "session SSE reconstructs state after publication instead of carrying the committed receipt",
    );
  }
  if (
    events.includes("session_effect") ||
    sessionPage.includes("transcript.session_effect")
  ) {
    throw new Error(
      "the browser event contract is reconstructing lifecycle state outside the committed snapshot",
    );
  }
  if (/transcript\.sequence\s*[<>]=?\s*details\.revision/.test(sessionPage)) {
    throw new Error(
      "the browser is comparing the event and session revision clocks",
    );
  }
  const answerTransaction = server.slice(
    server.indexOf("async fn answer_question_inner("),
    server.indexOf("fn session_history_summary("),
  );
  if (
    !answerTransaction.includes("commit_session_change_owned_after") ||
    answerTransaction.includes("let (responder, answer) = commit.value")
  ) {
    throw new Error(
      "question delivery escaped the cancellation-safe commit owner",
    );
  }
  const registryReload = server.slice(
    server.indexOf("async fn reload_projects("),
    server.indexOf("async fn reconcile_project_registry("),
  );
  if (
    !registryReload.includes(".commit_project_registry_owned(") ||
    registryReload.includes("state.projects.lock().await")
  ) {
    throw new Error(
      "project registry reload bypasses the cancellation-safe coordinator",
    );
  }
  for (
    const required of [
      '#[serde(tag = "frame", rename_all = "snake_case", deny_unknown_fields)]',
      "SessionEvent",
      "SessionFinished",
      "ReplayReset",
      "StreamError",
    ]
  ) {
    if (!protocol.includes(required)) {
      throw new Error(`missing tagged daemon protocol contract: ${required}`);
    }
  }
  const sessionEventFrame = protocol.slice(
    protocol.indexOf("SessionEvent {"),
    protocol.indexOf("SessionFinished {"),
  );
  if (
    protocol.includes("Notification {") ||
    !sessionEventFrame.includes("session_id: String") ||
    !client.includes("RpcFrame::SessionEvent") ||
    !client.includes("RpcFrame::SessionFinished") ||
    !client.includes("RpcFrame::ReplayReset") ||
    !client.includes("RpcFrame::StreamError") ||
    !client.includes("daemon streamed session") ||
    !client.includes("before session_finished") ||
    !client.includes("without replay_reset")
  ) {
    throw new Error(
      "terminal client does not consume replay and stream phases",
    );
  }
  const terminalWatch = server.slice(
    server.indexOf("async fn stream_session_watch("),
    server.indexOf("async fn write_session_event("),
  );
  if (
    !terminalWatch.includes("SessionStreamPublication::Finished") ||
    terminalWatch.includes("tokio::time::sleep")
  ) {
    throw new Error("terminal watch is polling around a missing finish phase");
  }
  if (
    !server.includes("warnings: publication.warnings.clone()") ||
    !server.includes(
      "project_reconciliation_failure_is_revisioned_until_a_retry_clears_it",
    ) ||
    !client.includes("ProjectMutationReceipt<ProjectEntry>")
  ) {
    throw new Error(
      "project reconciliation warnings are not retained across adapters",
    );
  }
  const sessionRecord = server.slice(
    server.indexOf("struct SessionRecord"),
    server.indexOf("struct SessionRuntime"),
  );
  const persistedSession = store.slice(
    store.indexOf("pub struct PersistedSession"),
    store.indexOf("impl PersistedSession"),
  );
  const sessionListItem = server.slice(
    server.indexOf("pub struct SessionListItem"),
    server.indexOf("pub struct ProjectSessionSnapshot"),
  );
  const sessionDetails = server.slice(
    server.indexOf("pub struct SessionDetails"),
    server.indexOf("pub struct SessionStreamSnapshot"),
  );
  const lifecycleEvent = events.slice(
    events.indexOf("SessionStateChanged {"),
    events.indexOf("LlmInvocation {"),
  );
  if (
    sessionRecord.includes("running:") ||
    sessionRecord.includes("paused:") ||
    persistedSession.includes("running:") ||
    persistedSession.includes("paused:") ||
    sessionListItem.includes("running:") ||
    sessionListItem.includes("paused:") ||
    sessionDetails.includes("running:") ||
    sessionDetails.includes("paused:") ||
    lifecycleEvent.includes("running:") ||
    lifecycleEvent.includes("paused:") ||
    !server.includes("change.effects.dispatch |=") ||
    !server.includes("change.effects.publish_collection |= usage_changed") ||
    !server.includes(
      "staged.metrics = combined_metrics(&staged_usage_records)",
    ) ||
    !server.includes("let lifecycle_changed =") ||
    !server.includes(
      "lifecycle_changed.then(|| stage_session_state_changed(&staged))",
    ) ||
    !server.includes("let became_terminal = !was_terminal") ||
    !server.includes("let terminal_entry_key = became_terminal.then(||") ||
    !server.includes(
      "let watch_finished = watch_active_before && !session_watch_active(&staged)",
    ) ||
    !server.includes("publish_finished: commit.watch_finished")
  ) {
    throw new Error(
      "lifecycle event, terminal transition, or watch finish ordering still depends on handlers",
    );
  }
  for (
    const receipt of [
      "Result<SessionMutationReceipt>",
      "Result<GoalMutationReceipt>",
      "Result<DeleteSessionMutationResponse>",
      "Result<ProjectMutationReceipt<ProjectEntry>>",
    ]
  ) {
    if (!client.includes(receipt)) {
      throw new Error(
        `terminal mutation dropped its committed receipt: ${receipt}`,
      );
    }
  }
});

Deno.test("v6 consumers do not reconstruct omitted server state", async () => {
  const [helpers, hooks, session, sessionPage, projectsPage, energy] =
    await Promise
      .all([
        Deno.readTextFile("webui/src/lib/helpers.ts"),
        Deno.readTextFile("webui/src/lib/hooks.ts"),
        Deno.readTextFile("webui/src/components/Session.tsx"),
        Deno.readTextFile("webui/src/pages/SessionPage.tsx"),
        Deno.readTextFile("webui/src/pages/ProjectsPage.tsx"),
        Deno.readTextFile("webui/src/lib/energy.ts"),
      ]);

  for (
    const workaround of [
      "session.usage_records?.length",
      "call.event.call_id || result.event.call_id",
      'event.purpose || "unclassified"',
      "teammateFeedback.detail || event.reason",
      'String(event.message || "")',
      "rawDetail.startsWith",
      "latestPendingDeliveryProposal",
      "latestPendingGoalProposal",
      "latestGoalChangeRequest",
      "projectName(",
      "metrics.ended_at_ms ??",
      "metrics.started_at_ms ??",
      "usageStatsForToday(",
      "seenTerminalEntries",
      "terminal_transitions.filter",
    ]
  ) {
    if (
      helpers.includes(workaround) || hooks.includes(workaround) ||
      session.includes(workaround) ||
      sessionPage.includes(workaround) || projectsPage.includes(workaround)
    ) {
      throw new Error(
        `v6 UI still reconstructs server state with: ${workaround}`,
      );
    }
  }
  if (energy.includes("legacy") || energy.includes("llm_energy_kwh ??")) {
    throw new Error(
      "v6 energy totals still contain a legacy snapshot fallback",
    );
  }
});

Deno.test("web endpoints and consumers preserve structured API failures", async () => {
  const [server, sessionPage, settingsPage] = await Promise.all([
    Deno.readTextFile("src/web.rs"),
    Deno.readTextFile("webui/src/pages/SessionPage.tsx"),
    Deno.readTextFile("webui/src/pages/SettingsPage.tsx"),
  ]);

  for (
    const signature of [
      ") -> Result<Json<SessionDetails>, ApiError>",
      ") -> Result<Json<DeleteSessionMutationResponse>, ApiError>",
      ") -> Result<Json<crate::goal::GoalCheckpoint>, ApiError>",
      ") -> Result<Json<WebSettingsResponse>, ApiError>",
      ") -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, ApiError>",
    ]
  ) {
    if (!server.includes(signature)) {
      throw new Error(
        `endpoint lost its structured error contract: ${signature}`,
      );
    }
  }
  if (!sessionPage.includes('apiErrorMessage(res, "Session request failed")')) {
    throw new Error("session page does not render the server API error");
  }
  if (
    settingsPage.includes("responseError") ||
    settingsPage.includes("`HTTP ${response.status}`") ||
    !settingsPage.includes("apiErrorMessage")
  ) {
    throw new Error("settings page reconstructs API failures locally");
  }
});
