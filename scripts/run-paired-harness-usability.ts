import {
  aggregateAudits,
  auditScratch,
  UsabilityAggregate,
  UsabilityAudit,
} from "./audit-harness-usability.ts";
import { validatedUsabilityCorpus } from "./check-harness-usability-corpus.ts";
import { CorpusCase, prepareCorpusCase } from "./run-harness-task-corpus.ts";

const DEFAULT_CASE_IDS = [
  "rust_registry_removal",
  "python_ttl_cache_boundary",
  "react_accessible_alert",
];
const LOCKED_CONTEXT_SIZE = 131_072;
const LOCKED_GPU_LAYERS = 999;

type Variant = "baseline" | "candidate";

interface Options {
  scratchParent: string;
  baselineBinary: string;
  candidateBinary: string;
  baselineRevision: string;
  candidateRevision: string;
  model: string;
  repeats: number;
  caseIds: string[];
}

interface Trial {
  round: number;
  order: number;
  variant: Variant;
  binary_sha256: string;
  process_exit_code: number;
  audit: UsabilityAudit;
}

interface MetricSummary {
  raw: number[];
  median: number;
}

interface VariantSummary {
  wall_runtime_ms: MetricSummary;
  fresh_prefill_tokens: MetricSummary;
  energy_kwh: MetricSummary;
  llm_invocations: MetricSummary;
}

export interface PairedPromotionReport {
  version: "paired-usability-v1";
  complete: boolean;
  protocol: {
    repeats: number;
    cases: string[];
    machine_class: string;
    model: string;
    sampling: {
      temperature: 0;
      top_k: 1;
      seed: 0;
      ctx_size: number;
      gpu_layers: number;
    };
    ordering: string;
  };
  variants: Record<Variant, {
    revision: string;
    binary_sha256: string;
    summary: VariantSummary;
  }>;
  per_case_wall_runtime_ms: Record<string, {
    baseline: MetricSummary;
    candidate: MetricSummary;
    candidate_change_percent: number;
  }>;
  comparison: {
    wall_time_reduction_percent: number;
    fresh_prefill_reduction_percent: number;
    energy_reduction_percent: number;
  };
  gates: {
    all_official_correct: boolean;
    all_verified_clean: boolean;
    zero_false_verified_completion: boolean;
    candidate_rust_python_four_call_floor: boolean;
    candidate_exact_root_reuse: boolean;
    wall_time_reduction_at_least_15_percent: boolean;
    fresh_prefill_reduction_at_least_25_percent: boolean;
    complete_energy_coverage: boolean;
    energy_reduction_at_least_15_percent: boolean;
    no_case_wall_regression_above_10_percent: boolean;
    production_performance_promoted: boolean;
  };
  trials: Trial[];
}

function requireValue(args: string[], index: number, argument: string): string {
  const value = args[index + 1];
  if (!value) throw new Error(`${argument} requires a value`);
  return value;
}

function parseOptions(args: string[]): Options {
  let scratchParent: string | undefined;
  let baselineBinary: string | undefined;
  let candidateBinary: string | undefined;
  let baselineRevision: string | undefined;
  let candidateRevision: string | undefined;
  let model: string | undefined;
  let repeats = 3;
  const caseIds: string[] = [];

  for (let index = 0; index < args.length; index++) {
    const argument = args[index];
    if (argument === "--scratch-parent") {
      scratchParent = requireValue(args, index, argument);
      index++;
    } else if (argument === "--baseline-binary") {
      baselineBinary = requireValue(args, index, argument);
      index++;
    } else if (argument === "--candidate-binary") {
      candidateBinary = requireValue(args, index, argument);
      index++;
    } else if (argument === "--baseline-revision") {
      baselineRevision = requireValue(args, index, argument);
      index++;
    } else if (argument === "--candidate-revision") {
      candidateRevision = requireValue(args, index, argument);
      index++;
    } else if (argument === "--model") {
      model = requireValue(args, index, argument);
      index++;
    } else if (argument === "--repeats") {
      repeats = Number(requireValue(args, index, argument));
      index++;
    } else if (argument === "--case") {
      caseIds.push(requireValue(args, index, argument));
      index++;
    } else {
      throw new Error(`unknown argument: ${argument}`);
    }
  }

  if (
    !scratchParent || !baselineBinary || !candidateBinary ||
    !baselineRevision || !candidateRevision || !model
  ) {
    throw new Error(
      "--scratch-parent, --model, both --*-binary paths, and both --*-revision values are required",
    );
  }
  if (!Number.isInteger(repeats) || repeats < 3 || repeats % 2 === 0) {
    throw new Error("--repeats must be an odd integer of at least 3");
  }
  return {
    scratchParent,
    baselineBinary,
    candidateBinary,
    baselineRevision,
    candidateRevision,
    model,
    repeats,
    caseIds: caseIds.length > 0 ? caseIds : DEFAULT_CASE_IDS,
  };
}

export function median(values: number[]): number {
  if (values.length === 0) {
    throw new Error("median requires at least one value");
  }
  const sorted = values.toSorted((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

export function pairedVariantOrder(round: number): [Variant, Variant] {
  return round % 2 === 0
    ? ["baseline", "candidate"]
    : ["candidate", "baseline"];
}

function metric(values: number[]): MetricSummary {
  return { raw: values, median: median(values) };
}

function reductionPercent(baseline: number, candidate: number): number {
  if (baseline === 0) throw new Error("baseline metric must be nonzero");
  return ((baseline - candidate) / baseline) * 100;
}

function requiredNumber(value: number | undefined, label: string): number {
  if (value === undefined) throw new Error(`${label} is missing`);
  return value;
}

function aggregateForRound(
  trials: Trial[],
  round: number,
  variant: Variant,
): UsabilityAggregate {
  return aggregateAudits(
    trials.filter((trial) => trial.round === round && trial.variant === variant)
      .map((trial) => trial.audit),
  );
}

function summarizeVariant(aggregates: UsabilityAggregate[]): VariantSummary {
  return {
    wall_runtime_ms: metric(
      aggregates.map((item) => item.total_wall_runtime_ms),
    ),
    fresh_prefill_tokens: metric(
      aggregates.map((item) => item.total_fresh_prefill_tokens),
    ),
    energy_kwh: metric(aggregates.map((item) => item.total_energy_kwh)),
    llm_invocations: metric(
      aggregates.map((item) => item.total_llm_invocations),
    ),
  };
}

export function buildPairedReport(
  options: Pick<
    Options,
    | "repeats"
    | "caseIds"
    | "baselineRevision"
    | "candidateRevision"
    | "model"
  >,
  binaryHashes: Record<Variant, string>,
  trials: Trial[],
): PairedPromotionReport {
  const expectedTrials = options.repeats * options.caseIds.length * 2;
  if (trials.length !== expectedTrials) {
    throw new Error(
      `expected ${expectedTrials} trials, received ${trials.length}`,
    );
  }
  const rounds = Array.from({ length: options.repeats }, (_, index) => index);
  const baselineAggregates = rounds.map((round) =>
    aggregateForRound(trials, round, "baseline")
  );
  const candidateAggregates = rounds.map((round) =>
    aggregateForRound(trials, round, "candidate")
  );
  const baselineSummary = summarizeVariant(baselineAggregates);
  const candidateSummary = summarizeVariant(candidateAggregates);

  const perCaseWallRuntime: PairedPromotionReport["per_case_wall_runtime_ms"] =
    {};
  for (const caseId of options.caseIds) {
    const baseline = metric(
      trials.filter((trial) =>
        trial.variant === "baseline" && trial.audit.case_id === caseId
      ).map((trial) =>
        requiredNumber(
          trial.audit.efficiency.wall_runtime_ms,
          `${caseId} baseline wall runtime`,
        )
      ),
    );
    const candidate = metric(
      trials.filter((trial) =>
        trial.variant === "candidate" && trial.audit.case_id === caseId
      ).map((trial) =>
        requiredNumber(
          trial.audit.efficiency.wall_runtime_ms,
          `${caseId} candidate wall runtime`,
        )
      ),
    );
    perCaseWallRuntime[caseId] = {
      baseline,
      candidate,
      candidate_change_percent: -reductionPercent(
        baseline.median,
        candidate.median,
      ),
    };
  }

  const comparison = {
    wall_time_reduction_percent: reductionPercent(
      baselineSummary.wall_runtime_ms.median,
      candidateSummary.wall_runtime_ms.median,
    ),
    fresh_prefill_reduction_percent: reductionPercent(
      baselineSummary.fresh_prefill_tokens.median,
      candidateSummary.fresh_prefill_tokens.median,
    ),
    energy_reduction_percent: reductionPercent(
      baselineSummary.energy_kwh.median,
      candidateSummary.energy_kwh.median,
    ),
  };
  const candidateTrials = trials.filter((trial) =>
    trial.variant === "candidate"
  );
  const gates = {
    all_official_correct: trials.every((trial) =>
      trial.audit.official.task_passed
    ),
    all_verified_clean: trials.every((trial) =>
      trial.audit.safety.verified_clean_completion
    ),
    zero_false_verified_completion: trials.every((trial) =>
      !trial.audit.safety.false_verified_completion
    ),
    candidate_rust_python_four_call_floor: candidateTrials.filter((trial) =>
      trial.audit.language === "rust" || trial.audit.language === "python"
    ).every((trial) => trial.audit.efficiency.llm_invocations === 4),
    candidate_exact_root_reuse: candidateTrials.every((trial) =>
      (trial.audit.efficiency.eligible_root_tokens ?? 0) > 0 &&
      trial.audit.efficiency.eligible_root_tokens ===
        trial.audit.efficiency.reused_root_tokens &&
      trial.audit.efficiency.prompt_root_hit_invocations ===
        trial.audit.efficiency.llm_invocations &&
      trial.audit.efficiency.prompt_cache_reconciliation_failures === 0
    ),
    wall_time_reduction_at_least_15_percent:
      comparison.wall_time_reduction_percent >= 15,
    fresh_prefill_reduction_at_least_25_percent:
      comparison.fresh_prefill_reduction_percent >= 25,
    complete_energy_coverage: trials.every((trial) =>
      trial.audit.efficiency.energy_complete
    ),
    energy_reduction_at_least_15_percent:
      comparison.energy_reduction_percent >= 15,
    no_case_wall_regression_above_10_percent: Object.values(
      perCaseWallRuntime,
    ).every((item) => item.candidate_change_percent <= 10),
    production_performance_promoted: false,
  };
  gates.production_performance_promoted = Object.entries(gates).filter(
    ([name]) => name !== "production_performance_promoted",
  ).every(([, passed]) => passed);

  return {
    version: "paired-usability-v1",
    complete: true,
    protocol: {
      repeats: options.repeats,
      cases: options.caseIds,
      machine_class: `${Deno.build.os}-${Deno.build.arch}`,
      model: options.model,
      sampling: {
        temperature: 0,
        top_k: 1,
        seed: 0,
        ctx_size: LOCKED_CONTEXT_SIZE,
        gpu_layers: LOCKED_GPU_LAYERS,
      },
      ordering:
        "case order rotates by round; baseline/candidate order alternates by round and case",
    },
    variants: {
      baseline: {
        revision: options.baselineRevision,
        binary_sha256: binaryHashes.baseline,
        summary: baselineSummary,
      },
      candidate: {
        revision: options.candidateRevision,
        binary_sha256: binaryHashes.candidate,
        summary: candidateSummary,
      },
    },
    per_case_wall_runtime_ms: perCaseWallRuntime,
    comparison,
    gates,
    trials,
  };
}

async function sha256(path: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    await Deno.readFile(path),
  );
  return Array.from(new Uint8Array(digest)).map((byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

async function writeJsonAtomically(
  path: string,
  value: unknown,
): Promise<void> {
  const temporary = `${path}.tmp`;
  await Deno.writeTextFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    create: true,
  });
  await Deno.rename(temporary, path);
}

function selectedCases(caseIds: string[]): CorpusCase[] {
  const corpus = validatedUsabilityCorpus();
  const selected = caseIds.map((caseId) => {
    const item = corpus.cases.find((candidate) => candidate.id === caseId);
    if (!item) throw new Error(`unknown case: ${caseId}`);
    return item;
  });
  if (new Set(caseIds).size !== caseIds.length) {
    throw new Error("--case values must be unique");
  }
  return selected;
}

async function requireAbsent(path: string): Promise<void> {
  try {
    await Deno.stat(path);
    throw new Error(`scratch parent already exists: ${path}`);
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) return;
    throw error;
  }
}

async function runTrial(
  corpusCase: CorpusCase,
  scratchRoot: string,
  binary: string,
  model: string,
): Promise<{ audit: UsabilityAudit; exitCode: number }> {
  const prepared = await prepareCorpusCase(corpusCase, scratchRoot);
  const status = await new Deno.Command(binary, {
    args: [
      "harness",
      "agent",
      "--scratch-dir",
      prepared.scratch_root,
      "--contract",
      prepared.contract,
      "--model",
      model,
      "--max-steps",
      String(corpusCase.limits.max_steps),
      "--max-tokens",
      String(corpusCase.limits.max_tokens),
      "--ctx-size",
      String(LOCKED_CONTEXT_SIZE),
      "--gpu-layers",
      String(LOCKED_GPU_LAYERS),
      "--temperature",
      "0",
      "--top-k",
      "1",
      "--seed",
      "0",
      corpusCase.task,
    ],
    stdin: "null",
    stdout: "inherit",
    stderr: "inherit",
  }).spawn().status;
  const audit = await auditScratch(scratchRoot);
  return { audit, exitCode: status.code };
}

async function main(): Promise<void> {
  const options = parseOptions(Deno.args);
  await requireAbsent(options.scratchParent);
  const cases = selectedCases(options.caseIds);
  const binaries: Record<Variant, string> = {
    baseline: await Deno.realPath(options.baselineBinary),
    candidate: await Deno.realPath(options.candidateBinary),
  };
  const binaryHashes: Record<Variant, string> = {
    baseline: await sha256(binaries.baseline),
    candidate: await sha256(binaries.candidate),
  };
  await Deno.mkdir(options.scratchParent, { recursive: true });
  const reportPath = `${options.scratchParent}/paired-results.json`;
  const trials: Trial[] = [];

  for (let round = 0; round < options.repeats; round++) {
    for (let offset = 0; offset < cases.length; offset++) {
      const corpusCase = cases[(offset + round) % cases.length];
      const order = pairedVariantOrder(round);
      for (let position = 0; position < order.length; position++) {
        const variant = order[position];
        const scratchRoot = `${options.scratchParent}/round-${
          round + 1
        }/${variant}/${corpusCase.id}`;
        console.error(
          `paired trial round=${
            round + 1
          }/${options.repeats} case=${corpusCase.id} variant=${variant}`,
        );
        const result = await runTrial(
          corpusCase,
          scratchRoot,
          binaries[variant],
          options.model,
        );
        trials.push({
          round,
          order: position,
          variant,
          binary_sha256: binaryHashes[variant],
          process_exit_code: result.exitCode,
          audit: result.audit,
        });
        await writeJsonAtomically(reportPath, {
          version: "paired-usability-v1",
          complete: false,
          protocol: {
            repeats: options.repeats,
            cases: options.caseIds,
            machine_class: `${Deno.build.os}-${Deno.build.arch}`,
            model: options.model,
            sampling: {
              temperature: 0,
              top_k: 1,
              seed: 0,
              ctx_size: LOCKED_CONTEXT_SIZE,
              gpu_layers: LOCKED_GPU_LAYERS,
            },
          },
          variants: {
            baseline: {
              revision: options.baselineRevision,
              binary_sha256: binaryHashes.baseline,
            },
            candidate: {
              revision: options.candidateRevision,
              binary_sha256: binaryHashes.candidate,
            },
          },
          completed_trials: trials,
        });
      }
    }
  }

  const report = buildPairedReport(options, binaryHashes, trials);
  await writeJsonAtomically(reportPath, report);
  console.log(JSON.stringify(report));
  if (!report.gates.production_performance_promoted) Deno.exitCode = 2;
}

if (import.meta.main) {
  try {
    await main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}
