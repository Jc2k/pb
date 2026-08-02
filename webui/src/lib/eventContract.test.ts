/// <reference lib="deno.ns" />
import { deepEqual } from "node:assert/strict";

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

function typescriptStringUnion(source: string, typeName: string): string[] {
  const start = source.indexOf(`export type ${typeName} =`);
  if (start < 0) throw new Error(`missing TypeScript type ${typeName}`);
  const end = source.indexOf(";", start);
  return [...source.slice(start, end).matchAll(/"([^"]+)"/g)].map((match) =>
    match[1]
  );
}

Deno.test("Rust and TypeScript expose the same event and profile variants", async () => {
  const [events, agents, workflow, types] = await Promise.all([
    Deno.readTextFile("src/events.rs"),
    Deno.readTextFile("src/agent_core.rs"),
    Deno.readTextFile("src/workflow/mod.rs"),
    Deno.readTextFile("webui/src/types/index.ts"),
  ]);

  deepEqual(
    typescriptEventVariants(types).sort(),
    rustEnumVariants(events, "AgentEvent").map(snakeCase).sort(),
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
    const required of [
      "chatter: EventChatter[]",
      "evidence: EventEvidence[]",
      "transcript: TranscriptMetadata",
      "sequence: number",
      "entry_key: string",
      "supersedes: string[]",
      "summary_redundant: boolean",
      "session_effect: SessionEffect",
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
      "id: string",
      "projects: ProjectEntry[]",
      "sessions: SessionItem[]",
    ]
  ) {
    if (!types.includes(required)) {
      throw new Error(`missing required v5 event field: ${required}`);
    }
  }
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
      "pub struct ProjectSessionSnapshot",
      'route("/api/project-sessions", get(list_project_sessions))',
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
  if (!hooks.includes('fetch("/api/project-sessions"')) {
    throw new Error("project pages do not consume the atomic server snapshot");
  }
  if (
    server.includes("pub refresh: bool") || types.includes("refresh: boolean")
  ) {
    throw new Error(
      "session snapshot scheduling leaked into the wire contract",
    );
  }
});

Deno.test("v5 consumers do not reconstruct omitted server state", async () => {
  const [helpers, session, sessionPage, projectsPage, energy] = await Promise
    .all([
      Deno.readTextFile("webui/src/lib/helpers.ts"),
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
    ]
  ) {
    if (
      helpers.includes(workaround) || session.includes(workaround) ||
      sessionPage.includes(workaround) || projectsPage.includes(workaround)
    ) {
      throw new Error(
        `v5 UI still reconstructs server state with: ${workaround}`,
      );
    }
  }
  if (energy.includes("legacy") || energy.includes("llm_energy_kwh ??")) {
    throw new Error(
      "v5 energy totals still contain a legacy snapshot fallback",
    );
  }
});
