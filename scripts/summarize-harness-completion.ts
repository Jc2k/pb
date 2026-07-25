export interface CompletionRunSummary {
  version: "v1";
  scratch_root: string;
  run_id: string;
  task: string;
  model?: string;
  status: string;
  contract_status: string;
  verified_completed: boolean;
  reported_completion: boolean;
  termination_reason?: string;
  handoff_outcome?: string;
  workflow_outcome?: string;
  workflow_stages: string[];
  workflow_stage_steps: Record<string, number>;
  rejected_workflow_actions: number;
  checks_planned: string[];
  checks_executed: number;
  checks_passed: number;
  checks_failed: number;
  repair_cycles: number;
  commit_disposition?: string;
  commit_oid?: string;
  commit_changed_paths: string[];
  wall_runtime_ms?: number;
  llm_invocations?: number;
  prompt_tokens?: number;
  rendered_prompt_tokens?: number;
  cached_prefix_tokens?: number;
  fresh_prefill_tokens?: number;
  prompt_cache_hit_invocations?: number;
  prompt_cache_miss_reasons: Record<string, number>;
  tool_schema_sha256s: string[];
  generated_tokens?: number;
  tool_calls?: number;
  total_energy_kwh?: number;
  energy_complete?: boolean;
}

interface StartedRecord {
  state: "started";
  run_id: string;
  task: string;
  run_events: string;
}

interface FinishedRecord {
  state: "finished";
  run_id: string;
  status: string;
  contract_status: string;
  verified_completed: boolean;
  termination_reason?: string;
  handoff_outcome?: string;
  audit?: Record<string, unknown>;
}

type RunRecord = StartedRecord | FinishedRecord;

interface EventEnvelope {
  event?: Record<string, unknown>;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function numberOrZero(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function numberRecord(value: unknown): Record<string, number> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1])
    ),
  );
}

export function parseJsonLines(text: string, label: string): unknown[] {
  const records: unknown[] = [];
  for (const [index, raw] of text.split(/\r?\n/).entries()) {
    const line = raw.trim();
    if (line.length === 0) continue;
    try {
      records.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`${label}:${index + 1}: invalid JSON: ${error}`);
    }
  }
  return records;
}

function safeRunEventsPath(scratchRoot: string, relative: string): string {
  if (relative.startsWith("/") || relative.split("/").includes("..")) {
    throw new Error(
      `run event path must stay beneath the scratch root: ${relative}`,
    );
  }
  return `${scratchRoot}/${relative}`;
}

function eventOfType(
  events: EventEnvelope[],
  type: string,
): Record<string, unknown> | undefined {
  return events.map((envelope) => envelope.event).find((event) =>
    event?.type === type
  );
}

function eventsOfType(
  events: EventEnvelope[],
  type: string,
): Record<string, unknown>[] {
  return events
    .map((envelope) => envelope.event)
    .filter((event): event is Record<string, unknown> => event?.type === type);
}

export async function summarizeScratch(
  scratchPath: string,
): Promise<CompletionRunSummary[]> {
  const scratchRoot = await Deno.realPath(scratchPath);
  const runIndexText = await Deno.readTextFile(
    `${scratchRoot}/run-index.jsonl`,
  );
  const records = parseJsonLines(
    runIndexText,
    `${scratchRoot}/run-index.jsonl`,
  ) as RunRecord[];
  const started = new Map(
    records
      .filter((record): record is StartedRecord => record?.state === "started")
      .map((
        record,
      ) => [requiredString(record.run_id, "started run_id"), record]),
  );

  const summaries: CompletionRunSummary[] = [];
  for (
    const finished of records.filter(
      (record): record is FinishedRecord => record?.state === "finished",
    )
  ) {
    const runId = requiredString(finished.run_id, "finished run_id");
    const start = started.get(runId);
    if (!start) throw new Error(`run ${runId} has no matching started record`);

    const runEvents = requiredString(
      start.run_events,
      `run ${runId} event path`,
    );
    const eventText = await Deno.readTextFile(
      safeRunEventsPath(scratchRoot, runEvents),
    );
    const events = parseJsonLines(eventText, runEvents) as EventEnvelope[];
    const startEvent = eventOfType(events, "started");
    const metrics = eventOfType(events, "session_metrics");
    const invocations = eventsOfType(events, "llm_invocation");
    const renderedPromptTokens = invocations.reduce(
      (total, event) => total + numberOrZero(event.prompt_tokens),
      0,
    );
    const cachedPrefixTokens = invocations.reduce((total, event) => {
      const cache = event.prompt_cache;
      return total +
        (cache !== null && typeof cache === "object"
          ? numberOrZero((cache as Record<string, unknown>).cached_tokens)
          : 0);
    }, 0);
    const freshPrefillTokens = invocations.reduce((total, event) => {
      const native = event.native;
      if (native !== null && typeof native === "object") {
        return total + numberOrZero(
          (native as Record<string, unknown>).fresh_prefill_tokens,
        );
      }
      const cache = event.prompt_cache;
      return total +
        (cache !== null && typeof cache === "object"
          ? numberOrZero((cache as Record<string, unknown>).prefilled_tokens)
          : numberOrZero(event.prompt_tokens));
    }, 0);
    const promptCacheHitInvocations = invocations.filter((event) => {
      const cache = event.prompt_cache;
      return cache !== null && typeof cache === "object" &&
        numberOrZero((cache as Record<string, unknown>).cached_tokens) > 0;
    }).length;
    const promptCacheMissReasons = invocations.reduce<Record<string, number>>(
      (counts, event) => {
        const cache = event.prompt_cache;
        if (cache === null || typeof cache !== "object") return counts;
        const reason = optionalString(
          (cache as Record<string, unknown>).miss_reason,
        );
        if (reason) counts[reason] = (counts[reason] ?? 0) + 1;
        return counts;
      },
      {},
    );
    const toolSchemaSha256s = Array.from(
      new Set(invocations.flatMap((event) => {
        const native = event.native;
        if (native === null || typeof native !== "object") return [];
        const digest = optionalString(
          (native as Record<string, unknown>).tool_schema_sha256,
        );
        return digest ? [digest] : [];
      })),
    );
    const audit = finished.audit ?? {};
    const contractStatus = requiredString(
      finished.contract_status,
      "contract_status",
    );

    summaries.push({
      version: "v1",
      scratch_root: scratchRoot,
      run_id: runId,
      task: requiredString(start.task, `run ${runId} task`),
      model: optionalString(startEvent?.model),
      status: requiredString(finished.status, "status"),
      contract_status: contractStatus,
      verified_completed: finished.verified_completed === true,
      reported_completion: contractStatus === "satisfied" &&
        finished.verified_completed === true,
      termination_reason: optionalString(finished.termination_reason),
      handoff_outcome: optionalString(finished.handoff_outcome),
      workflow_outcome: optionalString(audit.workflow_outcome),
      workflow_stages: stringArray(audit.workflow_stage_sequence),
      workflow_stage_steps: numberRecord(audit.workflow_stage_steps),
      rejected_workflow_actions: numberOrZero(audit.rejected_workflow_actions),
      checks_planned: stringArray(audit.checks_planned),
      checks_executed: numberOrZero(audit.checks_executed),
      checks_passed: numberOrZero(audit.checks_passed),
      checks_failed: numberOrZero(audit.checks_failed),
      repair_cycles: numberOrZero(audit.repair_cycles),
      commit_disposition: optionalString(audit.commit_disposition),
      commit_oid: optionalString(audit.commit_oid),
      commit_changed_paths: stringArray(audit.commit_changed_paths),
      wall_runtime_ms: optionalNumber(metrics?.wall_runtime_ms),
      llm_invocations: optionalNumber(metrics?.llm_invocations),
      prompt_tokens: optionalNumber(metrics?.prompt_tokens),
      rendered_prompt_tokens: invocations.length > 0
        ? renderedPromptTokens
        : optionalNumber(metrics?.prompt_tokens),
      cached_prefix_tokens: invocations.length > 0
        ? cachedPrefixTokens
        : undefined,
      fresh_prefill_tokens: invocations.length > 0
        ? freshPrefillTokens
        : undefined,
      prompt_cache_hit_invocations: invocations.length > 0
        ? promptCacheHitInvocations
        : undefined,
      prompt_cache_miss_reasons: promptCacheMissReasons,
      tool_schema_sha256s: toolSchemaSha256s,
      generated_tokens: optionalNumber(metrics?.generated_tokens),
      tool_calls: optionalNumber(metrics?.tool_calls),
      total_energy_kwh: optionalNumber(metrics?.total_energy_kwh),
      energy_complete: typeof metrics?.energy_complete === "boolean"
        ? metrics.energy_complete
        : undefined,
    });
  }

  return summaries;
}

if (import.meta.main) {
  if (Deno.args.length === 0) {
    console.error(
      "usage: deno run --allow-read scripts/summarize-harness-completion.ts <scratch-root>...",
    );
    Deno.exit(2);
  }
  for (const scratchRoot of Deno.args) {
    for (const summary of await summarizeScratch(scratchRoot)) {
      console.log(JSON.stringify(summary));
    }
  }
}
