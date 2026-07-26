import {
  aggregateAudits,
  classifyAudit,
  UsabilityAudit,
} from "./audit-harness-usability.ts";
import { validatedUsabilityCorpus } from "./check-harness-usability-corpus.ts";
import { prepareCorpusCase } from "./run-harness-task-corpus.ts";
import {
  buildPairedReport,
  median,
  pairedVariantOrder,
} from "./run-paired-harness-usability.ts";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

Deno.test("usability corpus is balanced, sourced, bounded, and unguided", () => {
  const corpus = validatedUsabilityCorpus();
  assert(corpus.cases.length === 24, "case count");
  assert(
    new Set(corpus.cases.map((item) => item.source?.family)).size === 4,
    "source family coverage",
  );
  for (const corpusCase of corpus.cases) {
    assert(corpusCase.limits.max_steps <= 7, `${corpusCase.id}: max steps`);
    assert(corpusCase.limits.max_tokens <= 1792, `${corpusCase.id}: tokens`);
    assert(
      corpusCase.resume_files.length === 0,
      `${corpusCase.id}: fresh repository`,
    );
  }
});

Deno.test("paired qualification uses raw odd-sample medians and all promotion gates", () => {
  assert(median([900, 100, 300]) === 300, "odd median");
  assert(
    pairedVariantOrder(0).join(",") === "baseline,candidate" &&
      pairedVariantOrder(1).join(",") === "candidate,baseline",
    "pair ordering alternates by round",
  );
  const trials = [];
  const caseIds = [
    "rust_registry_removal",
    "python_ttl_cache_boundary",
    "react_accessible_alert",
  ];
  for (let round = 0; round < 3; round++) {
    for (const [caseIndex, caseId] of caseIds.entries()) {
      for (const variant of ["baseline", "candidate"] as const) {
        const audit = fakeAudit(
          caseId.startsWith("rust")
            ? "rust"
            : caseId.startsWith("python")
            ? "python"
            : "react_typescript",
          true,
          true,
          false,
        );
        audit.case_id = caseId;
        audit.efficiency.wall_runtime_ms = variant === "baseline"
          ? 1000 + round
          : 800 + round;
        audit.efficiency.llm_invocations = 4;
        audit.efficiency.fresh_prefill_tokens = variant === "baseline"
          ? 1000
          : 700;
        audit.efficiency.total_energy_kwh = variant === "baseline" ? 1 : 0.8;
        audit.efficiency.eligible_root_tokens = 10;
        audit.efficiency.reused_root_tokens = variant === "candidate" ? 10 : 5;
        audit.efficiency.prompt_root_hit_invocations = variant === "candidate"
          ? 4
          : 0;
        trials.push({
          round,
          order: caseIndex,
          variant,
          binary_sha256: variant,
          audit,
        });
      }
    }
  }
  const report = buildPairedReport(
    {
      repeats: 3,
      caseIds,
      baselineRevision: "base",
      candidateRevision: "candidate",
      model: "local-model",
    },
    { baseline: "base-sha", candidate: "candidate-sha" },
    trials,
  );
  assert(report.complete, "complete report");
  assert(
    report.variants.baseline.summary.wall_runtime_ms.median === 3003,
    "median aggregate wall",
  );
  assert(
    report.comparison.wall_time_reduction_percent > 19,
    "wall reduction",
  );
  assert(
    report.gates.production_performance_promoted,
    "all gates promote",
  );
});

Deno.test("checked-in typed corpus materializes without reference leakage", async () => {
  const corpus = validatedUsabilityCorpus();
  assert(corpus.cases.length === 24, "loaded cases");
  const parent = await Deno.makeTempDir();
  try {
    const corpusCase = corpus.cases.find((item) =>
      item.id === "react_accessible_alert"
    );
    assert(corpusCase, "React fixture");
    await prepareCorpusCase(corpusCase, `${parent}/case`);
    const metadata = JSON.parse(
      await Deno.readTextFile(`${parent}/case/corpus-case.json`),
    );
    assert(metadata.language === "react_typescript", "language metadata");
    assert(metadata.source.family === "react-bench", "source metadata");
    const implementation = await Deno.readTextFile(
      `${parent}/case/workspace/src/Component.tsx`,
    );
    const seed = corpusCase.seed_files.find((item) =>
      item.path === "src/Component.tsx"
    );
    const reference = corpusCase.reference_files.find((item) =>
      item.path === "src/Component.tsx"
    );
    assert(
      implementation === seed?.content,
      "seed implementation materialized",
    );
    assert(
      implementation !== reference?.content,
      "reference solution stays outside model workspace",
    );
  } finally {
    await Deno.remove(parent, { recursive: true });
  }
});

Deno.test("audit classification treats false verification as a pb defect", () => {
  assert(
    classifyAudit(true, true) === "positive_evidence",
    "verified correct completion",
  );
  assert(
    classifyAudit(false, true) === "pb_defect_false_verification",
    "incorrect verified completion",
  );
  assert(
    classifyAudit(true, false) === "model_or_control_limit",
    "correct but unverified work",
  );
  assert(
    classifyAudit(false, false, false) === "experiment_error",
    "invalid experiment",
  );
  assert(
    classifyAudit(true, true, true, false) ===
      "pb_defect_false_verification",
    "dirty or unsafe verified completion",
  );
  assert(
    classifyAudit(true, true, true, true, false) ===
      "pb_defect_telemetry_invariant",
    "invalid cache telemetry",
  );
});

function fakeAudit(
  language: string,
  officialPassed: boolean,
  verifiedClean: boolean,
  falseVerified: boolean,
): UsabilityAudit {
  return {
    version: "usability-audit-v1",
    scratch_root: "/private/tmp/fake",
    case_id: `${language}_case`,
    language,
    source_family: "synthetic",
    official: {
      behavior_passed: officialPassed,
      immutable_fixture_passed: true,
      task_passed: officialPassed,
      check_exit_code: officialPassed ? 0 : 1,
      check_output: "",
    },
    pb: {
      status: verifiedClean || falseVerified ? "completed" : "failed",
      contract_status: verifiedClean || falseVerified
        ? "satisfied"
        : "unsatisfied",
      verified_completed: verifiedClean || falseVerified,
      head_oid: "abc",
      commit_oid_matches: verifiedClean,
      semantic_commit: verifiedClean,
    },
    safety: {
      workspace_clean: verifiedClean,
      changed_paths: ["app"],
      changed_paths_allowed: verifiedClean,
      false_verified_completion: falseVerified,
      verified_clean_completion: verifiedClean,
    },
    efficiency: {
      wall_runtime_ms: 10,
      llm_invocations: 1,
      workflow_stages: ["Planning"],
      workflow_stage_steps: { Planning: 1 },
      rejected_workflow_actions: 0,
      repair_cycles: 0,
      rendered_prompt_tokens: 30,
      cached_prefix_tokens: 5,
      fresh_prefill_tokens: 25,
      prompt_cache_miss_reasons: { cold_session: 1 },
      prompt_cache_lookup_details: { exact_root_checkpoint_missing: 1 },
      prompt_cache_miss_reasons_by_stage: {
        planning: { cold_session: 1 },
      },
      prompt_cache_miss_reasons_by_authority_class: {
        planning: { cold_session: 1 },
      },
      prompt_cache_reconciliation_failures: 0,
      eligible_root_tokens: 10,
      reused_root_tokens: 5,
      prompt_root_hit_invocations: 1,
      prompt_root_authority_classes: { planning: 1 },
      refill_cache_lookup_wall_ms: 1,
      refill_disk_read_decode_wall_ms: 5,
      refill_cpu_state_validation_allocation_wall_ms: 6,
      refill_state_hydration_wall_ms: 2,
      refill_fresh_suffix_prefill_wall_ms: 3,
      refill_snapshot_capture_wall_ms: 4,
      refill_persistence_queue_wall_ms: 7,
      prefill_command_kinds: { qwen_layer_major_matrix: 1 },
      prefill_command_reasons: { fresh_suffix_at_or_above_threshold: 1 },
      generated_tokens: 20,
      tool_calls: 2,
      cache_persistence_queued_checkpoints: 3,
      cache_persistence_completed_checkpoints: 3,
      cache_persistence_wall_ms: 8,
      cache_persistence_failures: 0,
      total_energy_kwh: 0.001,
      energy_complete: true,
    },
    classification: falseVerified
      ? "pb_defect_false_verification"
      : verifiedClean
      ? "positive_evidence"
      : "model_or_control_limit",
  };
}

Deno.test("aggregate keeps correctness, verified completion, and efficiency separate", () => {
  const aggregate = aggregateAudits([
    fakeAudit("rust", true, true, false),
    fakeAudit("python", true, false, false),
    fakeAudit("react_typescript", false, false, true),
  ]);
  assert(aggregate.runs === 3, "runs");
  assert(aggregate.official_passed === 2, "official passes");
  assert(aggregate.pb_verified_completed === 2, "pb verified");
  assert(aggregate.verified_clean_completion === 1, "verified clean");
  assert(aggregate.false_verified_completion === 1, "false verified");
  assert(aggregate.total_generated_tokens === 60, "tokens");
  assert(aggregate.total_llm_invocations === 3, "invocations");
  assert(aggregate.total_rendered_prompt_tokens === 90, "prompt tokens");
  assert(aggregate.total_fresh_prefill_tokens === 75, "fresh prefill");
  assert(aggregate.total_eligible_root_tokens === 30, "eligible roots");
  assert(aggregate.total_reused_root_tokens === 15, "reused roots");
  assert(aggregate.total_prompt_root_hit_invocations === 3, "root hits");
  assert(
    aggregate.prompt_root_authority_classes.planning === 3,
    "root authorities",
  );
  assert(aggregate.total_refill_cache_lookup_wall_ms === 3, "refill lookup");
  assert(aggregate.total_refill_disk_read_decode_wall_ms === 15, "refill disk");
  assert(
    aggregate.total_refill_cpu_state_validation_allocation_wall_ms === 18,
    "refill validation",
  );
  assert(
    aggregate.total_refill_state_hydration_wall_ms === 6,
    "refill hydration",
  );
  assert(
    aggregate.total_refill_fresh_suffix_prefill_wall_ms === 9,
    "refill suffix",
  );
  assert(
    aggregate.total_refill_snapshot_capture_wall_ms === 12,
    "refill snapshot",
  );
  assert(
    aggregate.total_refill_persistence_queue_wall_ms === 21,
    "refill queue",
  );
  assert(
    aggregate.prefill_command_kinds.qwen_layer_major_matrix === 3,
    "prefill commands",
  );
  assert(
    aggregate.prefill_command_reasons.fresh_suffix_at_or_above_threshold === 3,
    "prefill reasons",
  );
  assert(
    aggregate.total_cache_persistence_queued_checkpoints === 9,
    "persistence queued",
  );
  assert(
    aggregate.total_cache_persistence_completed_checkpoints === 9,
    "persistence completed",
  );
  assert(aggregate.total_cache_persistence_wall_ms === 24, "persistence wall");
  assert(
    aggregate.total_cache_persistence_failures === 0,
    "persistence failures",
  );
  assert(
    aggregate.prompt_cache_miss_reasons.cold_session === 3,
    "cache misses",
  );
  assert(
    aggregate.prompt_cache_lookup_details.exact_root_checkpoint_missing === 3,
    "cache lookup details",
  );
  assert(
    aggregate.prompt_cache_miss_reasons_by_stage.planning.cold_session === 3,
    "cache misses by stage",
  );
  assert(
    aggregate.prompt_cache_miss_reasons_by_authority_class.planning
      .cold_session === 3,
    "cache misses by authority",
  );
  assert(
    aggregate.total_prompt_cache_reconciliation_failures === 0,
    "cache reconciliation",
  );
  assert(aggregate.total_tool_calls === 6, "tools");
  assert(aggregate.energy_complete, "energy completeness");
});
