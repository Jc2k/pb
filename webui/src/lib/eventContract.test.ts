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

  for (
    const [rustSource, typeName] of [
      [events, "TeamMessageTone"],
      [events, "TeamMessagePurpose"],
      [events, "CorrectionKind"],
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
});
