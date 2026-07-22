import {
  parseJsonLines,
  summarizeScratch,
} from "./summarize-harness-completion.ts";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

Deno.test("parseJsonLines names the invalid source line", () => {
  let message = "";
  try {
    parseJsonLines('{"ok":true}\nnot-json\n', "fixture.jsonl");
  } catch (error) {
    message = String(error);
  }
  assert(message.includes("fixture.jsonl:2"), message);
});

Deno.test("summarizeScratch keeps completion and efficiency evidence separate", async () => {
  const root = await Deno.makeTempDir();
  try {
    await Deno.mkdir(`${root}/runs/run-1`, { recursive: true });
    await Deno.writeTextFile(
      `${root}/run-index.jsonl`,
      [
        JSON.stringify({
          state: "started",
          run_id: "run-1",
          task: "create two files",
          run_events: "runs/run-1/events.jsonl",
        }),
        JSON.stringify({
          state: "finished",
          run_id: "run-1",
          status: "verified_completed",
          contract_status: "satisfied",
          verified_completed: true,
          termination_reason: "final",
          handoff_outcome: "ready",
          audit: {
            workflow_outcome: "ready",
            workflow_stage_sequence: ["planning", "committing"],
            workflow_stage_steps: { planning: 1 },
            checks_planned: ["exact"],
            checks_executed: 1,
            checks_passed: 1,
            commit_disposition: "created",
            commit_oid: "abc123",
            commit_changed_paths: ["alpha.txt", "beta.txt"],
          },
        }),
      ].join("\n") + "\n",
    );
    await Deno.writeTextFile(
      `${root}/runs/run-1/events.jsonl`,
      [
        JSON.stringify({ event: { type: "started", model: "local-model" } }),
        JSON.stringify({
          event: {
            type: "llm_invocation",
            prompt_tokens: 120,
            prompt_cache: { cached_tokens: 80, prefilled_tokens: 40 },
            native: {
              fresh_prefill_tokens: 40,
              tool_schema_sha256: "schema-a",
            },
          },
        }),
        JSON.stringify({
          event: {
            type: "llm_invocation",
            prompt_tokens: 80,
            prompt_cache: { cached_tokens: 0, prefilled_tokens: 80 },
            native: {
              fresh_prefill_tokens: 80,
              tool_schema_sha256: "schema-b",
            },
          },
        }),
        JSON.stringify({
          event: {
            type: "session_metrics",
            wall_runtime_ms: 1000,
            llm_invocations: 6,
            prompt_tokens: 200,
            generated_tokens: 100,
            tool_calls: 4,
            total_energy_kwh: 0.002,
            energy_complete: true,
          },
        }),
      ].join("\n") + "\n",
    );

    const summaries = await summarizeScratch(root);
    assert(summaries.length === 1, JSON.stringify(summaries));
    const summary = summaries[0];
    assert(summary.reported_completion, JSON.stringify(summary));
    assert(summary.contract_status === "satisfied", JSON.stringify(summary));
    assert(summary.wall_runtime_ms === 1000, JSON.stringify(summary));
    assert(summary.total_energy_kwh === 0.002, JSON.stringify(summary));
    assert(summary.commit_oid === "abc123", JSON.stringify(summary));
    assert(summary.checks_passed === 1, JSON.stringify(summary));
    assert(summary.rendered_prompt_tokens === 200, JSON.stringify(summary));
    assert(summary.cached_prefix_tokens === 80, JSON.stringify(summary));
    assert(summary.fresh_prefill_tokens === 120, JSON.stringify(summary));
    assert(summary.prompt_cache_hit_invocations === 1, JSON.stringify(summary));
    assert(summary.tool_schema_sha256s.length === 2, JSON.stringify(summary));
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("summarizeScratch rejects run event traversal", async () => {
  const root = await Deno.makeTempDir();
  try {
    await Deno.writeTextFile(
      `${root}/run-index.jsonl`,
      [
        JSON.stringify({
          state: "started",
          run_id: "run-1",
          task: "task",
          run_events: "../events.jsonl",
        }),
        JSON.stringify({
          state: "finished",
          run_id: "run-1",
          status: "incomplete",
          contract_status: "unsatisfied",
          verified_completed: false,
        }),
      ].join("\n") + "\n",
    );
    let message = "";
    try {
      await summarizeScratch(root);
    } catch (error) {
      message = String(error);
    }
    assert(message.includes("must stay beneath"), message);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});
