import {
  CompletionRunSummary,
  summarizeScratch,
} from "./summarize-harness-completion.ts";
import {
  runBehaviorCheck,
  validatedUsabilityCorpus,
} from "./check-harness-usability-corpus.ts";
import type { CorpusCase } from "./run-harness-task-corpus.ts";

export type AuditClassification =
  | "positive_evidence"
  | "pb_defect_false_verification"
  | "model_or_control_limit"
  | "experiment_error";

export interface UsabilityAudit {
  version: "usability-audit-v1";
  scratch_root: string;
  case_id: string;
  language: string;
  source_family: string;
  official: {
    behavior_passed: boolean;
    immutable_fixture_passed: boolean;
    task_passed: boolean;
    check_exit_code: number;
    check_output: string;
  };
  pb: {
    status: string;
    contract_status: string;
    verified_completed: boolean;
    recorded_commit_oid?: string;
    head_oid: string;
    commit_oid_matches: boolean;
    semantic_commit: boolean;
  };
  safety: {
    workspace_clean: boolean;
    changed_paths: string[];
    changed_paths_allowed: boolean;
    false_verified_completion: boolean;
    verified_clean_completion: boolean;
  };
  efficiency: {
    wall_runtime_ms?: number;
    llm_invocations?: number;
    workflow_stages: string[];
    workflow_stage_steps: Record<string, number>;
    rejected_workflow_actions: number;
    repair_cycles: number;
    rendered_prompt_tokens?: number;
    cached_prefix_tokens?: number;
    fresh_prefill_tokens?: number;
    prompt_cache_miss_reasons: Record<string, number>;
    eligible_root_tokens?: number;
    reused_root_tokens?: number;
    prompt_root_hit_invocations?: number;
    prompt_root_authority_classes: Record<string, number>;
    refill_cache_lookup_wall_ms?: number;
    refill_state_hydration_wall_ms?: number;
    refill_fresh_suffix_prefill_wall_ms?: number;
    refill_snapshot_capture_wall_ms?: number;
    generated_tokens?: number;
    tool_calls?: number;
    total_energy_kwh?: number;
    energy_complete?: boolean;
  };
  classification: AuditClassification;
}

export interface UsabilityAggregate {
  version: "usability-aggregate-v1";
  runs: number;
  official_passed: number;
  pb_verified_completed: number;
  verified_clean_completion: number;
  false_verified_completion: number;
  by_language: Record<
    string,
    {
      runs: number;
      official_passed: number;
      verified_clean_completion: number;
    }
  >;
  total_wall_runtime_ms: number;
  total_llm_invocations: number;
  total_rendered_prompt_tokens: number;
  total_cached_prefix_tokens: number;
  total_fresh_prefill_tokens: number;
  prompt_cache_miss_reasons: Record<string, number>;
  total_eligible_root_tokens: number;
  total_reused_root_tokens: number;
  total_prompt_root_hit_invocations: number;
  prompt_root_authority_classes: Record<string, number>;
  total_refill_cache_lookup_wall_ms: number;
  total_refill_state_hydration_wall_ms: number;
  total_refill_fresh_suffix_prefill_wall_ms: number;
  total_refill_snapshot_capture_wall_ms: number;
  total_generated_tokens: number;
  total_tool_calls: number;
  total_energy_kwh: number;
  energy_complete: boolean;
}

const decoder = new TextDecoder();

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function commandText(
  command: string,
  args: string[],
  cwd: string,
): Promise<string> {
  const output = await new Deno.Command(command, { args, cwd }).output();
  if (!output.success) {
    throw new Error(
      `${command} ${args.join(" ")} failed: ${decoder.decode(output.stderr)}`,
    );
  }
  return decoder.decode(output.stdout).trim();
}

async function commandRaw(
  command: string,
  args: string[],
  cwd: string,
): Promise<string> {
  const output = await new Deno.Command(command, { args, cwd }).output();
  if (!output.success) {
    throw new Error(
      `${command} ${args.join(" ")} failed: ${decoder.decode(output.stderr)}`,
    );
  }
  return decoder.decode(output.stdout);
}

function porcelainPaths(status: string): string[] {
  return status.split("\n").filter((line) => line.trim().length > 0).map(
    (line) => line.slice(3).split(" -> ").at(-1)!,
  );
}

async function immutableFixturePassed(
  corpusCase: CorpusCase,
  workspace: string,
): Promise<boolean> {
  const allowed = new Set(corpusCase.contract.allowed_paths as string[]);
  for (const seed of corpusCase.seed_files) {
    if (allowed.has(seed.path)) continue;
    try {
      if (
        await Deno.readTextFile(`${workspace}/${seed.path}`) !== seed.content
      ) {
        return false;
      }
    } catch (error) {
      if (error instanceof Deno.errors.NotFound) return false;
      throw error;
    }
  }
  return true;
}

export function classifyAudit(
  officialPassed: boolean,
  pbVerified: boolean,
  experimentValid = true,
  verifiedClean = officialPassed && pbVerified,
): AuditClassification {
  if (!experimentValid) return "experiment_error";
  if (pbVerified && !verifiedClean) return "pb_defect_false_verification";
  if (verifiedClean) return "positive_evidence";
  return "model_or_control_limit";
}

function latestSummary(
  summaries: CompletionRunSummary[],
): CompletionRunSummary {
  assert(summaries.length > 0, "scratch root has no finished harness run");
  return summaries.at(-1)!;
}

export async function auditScratch(
  scratchRoot: string,
): Promise<UsabilityAudit> {
  const metadata = JSON.parse(
    await Deno.readTextFile(`${scratchRoot}/corpus-case.json`),
  ) as Record<string, unknown>;
  assert(typeof metadata.case_id === "string", "missing corpus case id");
  const corpus = validatedUsabilityCorpus();
  const corpusCase = corpus.cases.find((item) => item.id === metadata.case_id);
  assert(corpusCase, `unknown usability case: ${metadata.case_id}`);
  const workspace = `${scratchRoot}/workspace`;
  const summary = latestSummary(await summarizeScratch(scratchRoot));
  const behavior = await runBehaviorCheck(corpusCase, workspace);
  const immutable = await immutableFixturePassed(corpusCase, workspace);
  const headOid = await commandText("git", ["rev-parse", "HEAD"], workspace);
  const status = await commandRaw(
    "git",
    ["status", "--porcelain=v1", "--untracked-files=all"],
    workspace,
  );
  const baselineOid = await commandText(
    "git",
    ["rev-list", "--max-parents=0", "HEAD"],
    workspace,
  );
  const trackedChanges = (await commandText(
    "git",
    ["diff", "--name-only", baselineOid],
    workspace,
  )).split("\n").filter(Boolean).sort();
  const changed = [...new Set([...trackedChanges, ...porcelainPaths(status)])]
    .sort();
  const allowed = new Set(corpusCase.contract.allowed_paths as string[]);
  const changedPathsAllowed = changed.length > 0 &&
    changed.every((path) => allowed.has(path));
  const subject = await commandText(
    "git",
    ["show", "-s", "--format=%s", "HEAD"],
    workspace,
  );
  const semanticCommit =
    /^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([^)]+\))?!?: .+/
      .test(
        subject,
      );
  const officialPassed = behavior.success && immutable;
  const commitOidMatches = summary.commit_oid !== undefined &&
    summary.commit_oid === headOid;
  const workspaceClean = status.trim().length === 0;
  const verifiedClean = officialPassed && summary.verified_completed &&
    summary.contract_status === "satisfied" && commitOidMatches &&
    semanticCommit && workspaceClean && changedPathsAllowed;
  const falseVerified = summary.verified_completed && !verifiedClean;

  return {
    version: "usability-audit-v1",
    scratch_root: await Deno.realPath(scratchRoot),
    case_id: corpusCase.id,
    language: corpusCase.language ?? "unknown",
    source_family: corpusCase.source?.family ?? "unknown",
    official: {
      behavior_passed: behavior.success,
      immutable_fixture_passed: immutable,
      task_passed: officialPassed,
      check_exit_code: behavior.code,
      check_output: behavior.output,
    },
    pb: {
      status: summary.status,
      contract_status: summary.contract_status,
      verified_completed: summary.verified_completed,
      recorded_commit_oid: summary.commit_oid,
      head_oid: headOid,
      commit_oid_matches: commitOidMatches,
      semantic_commit: semanticCommit,
    },
    safety: {
      workspace_clean: workspaceClean,
      changed_paths: changed,
      changed_paths_allowed: changedPathsAllowed,
      false_verified_completion: falseVerified,
      verified_clean_completion: verifiedClean,
    },
    efficiency: {
      wall_runtime_ms: summary.wall_runtime_ms,
      llm_invocations: summary.llm_invocations,
      workflow_stages: summary.workflow_stages,
      workflow_stage_steps: summary.workflow_stage_steps,
      rejected_workflow_actions: summary.rejected_workflow_actions,
      repair_cycles: summary.repair_cycles,
      rendered_prompt_tokens: summary.rendered_prompt_tokens,
      cached_prefix_tokens: summary.cached_prefix_tokens,
      fresh_prefill_tokens: summary.fresh_prefill_tokens,
      prompt_cache_miss_reasons: summary.prompt_cache_miss_reasons,
      eligible_root_tokens: summary.eligible_root_tokens,
      reused_root_tokens: summary.reused_root_tokens,
      prompt_root_hit_invocations: summary.prompt_root_hit_invocations,
      prompt_root_authority_classes: summary.prompt_root_authority_classes,
      refill_cache_lookup_wall_ms: summary.refill_cache_lookup_wall_ms,
      refill_state_hydration_wall_ms: summary.refill_state_hydration_wall_ms,
      refill_fresh_suffix_prefill_wall_ms:
        summary.refill_fresh_suffix_prefill_wall_ms,
      refill_snapshot_capture_wall_ms: summary.refill_snapshot_capture_wall_ms,
      generated_tokens: summary.generated_tokens,
      tool_calls: summary.tool_calls,
      total_energy_kwh: summary.total_energy_kwh,
      energy_complete: summary.energy_complete,
    },
    classification: classifyAudit(
      officialPassed,
      summary.verified_completed,
      true,
      verifiedClean,
    ),
  };
}

export function aggregateAudits(audits: UsabilityAudit[]): UsabilityAggregate {
  const byLanguage: UsabilityAggregate["by_language"] = {};
  for (const audit of audits) {
    const language = byLanguage[audit.language] ?? {
      runs: 0,
      official_passed: 0,
      verified_clean_completion: 0,
    };
    language.runs += 1;
    language.official_passed += Number(audit.official.task_passed);
    language.verified_clean_completion += Number(
      audit.safety.verified_clean_completion,
    );
    byLanguage[audit.language] = language;
  }
  const promptCacheMissReasons: Record<string, number> = {};
  const promptRootAuthorityClasses: Record<string, number> = {};
  for (const audit of audits) {
    for (
      const [reason, count] of Object.entries(
        audit.efficiency.prompt_cache_miss_reasons,
      )
    ) {
      promptCacheMissReasons[reason] = (promptCacheMissReasons[reason] ?? 0) +
        count;
    }
  }
  for (const audit of audits) {
    for (
      const [authority, count] of Object.entries(
        audit.efficiency.prompt_root_authority_classes,
      )
    ) {
      promptRootAuthorityClasses[authority] =
        (promptRootAuthorityClasses[authority] ?? 0) + count;
    }
  }
  return {
    version: "usability-aggregate-v1",
    runs: audits.length,
    official_passed: audits.filter((item) => item.official.task_passed).length,
    pb_verified_completed: audits.filter((item) => item.pb.verified_completed)
      .length,
    verified_clean_completion:
      audits.filter((item) => item.safety.verified_clean_completion).length,
    false_verified_completion:
      audits.filter((item) => item.safety.false_verified_completion).length,
    by_language: byLanguage,
    total_wall_runtime_ms: audits.reduce(
      (total, item) => total + (item.efficiency.wall_runtime_ms ?? 0),
      0,
    ),
    total_llm_invocations: audits.reduce(
      (total, item) => total + (item.efficiency.llm_invocations ?? 0),
      0,
    ),
    total_rendered_prompt_tokens: audits.reduce(
      (total, item) => total + (item.efficiency.rendered_prompt_tokens ?? 0),
      0,
    ),
    total_cached_prefix_tokens: audits.reduce(
      (total, item) => total + (item.efficiency.cached_prefix_tokens ?? 0),
      0,
    ),
    total_fresh_prefill_tokens: audits.reduce(
      (total, item) => total + (item.efficiency.fresh_prefill_tokens ?? 0),
      0,
    ),
    prompt_cache_miss_reasons: promptCacheMissReasons,
    total_eligible_root_tokens: audits.reduce(
      (total, item) => total + (item.efficiency.eligible_root_tokens ?? 0),
      0,
    ),
    total_reused_root_tokens: audits.reduce(
      (total, item) => total + (item.efficiency.reused_root_tokens ?? 0),
      0,
    ),
    total_prompt_root_hit_invocations: audits.reduce(
      (total, item) =>
        total + (item.efficiency.prompt_root_hit_invocations ?? 0),
      0,
    ),
    prompt_root_authority_classes: promptRootAuthorityClasses,
    total_refill_cache_lookup_wall_ms: audits.reduce(
      (total, item) =>
        total + (item.efficiency.refill_cache_lookup_wall_ms ?? 0),
      0,
    ),
    total_refill_state_hydration_wall_ms: audits.reduce(
      (total, item) =>
        total + (item.efficiency.refill_state_hydration_wall_ms ?? 0),
      0,
    ),
    total_refill_fresh_suffix_prefill_wall_ms: audits.reduce(
      (total, item) =>
        total + (item.efficiency.refill_fresh_suffix_prefill_wall_ms ?? 0),
      0,
    ),
    total_refill_snapshot_capture_wall_ms: audits.reduce(
      (total, item) =>
        total + (item.efficiency.refill_snapshot_capture_wall_ms ?? 0),
      0,
    ),
    total_generated_tokens: audits.reduce(
      (total, item) => total + (item.efficiency.generated_tokens ?? 0),
      0,
    ),
    total_tool_calls: audits.reduce(
      (total, item) => total + (item.efficiency.tool_calls ?? 0),
      0,
    ),
    total_energy_kwh: audits.reduce(
      (total, item) => total + (item.efficiency.total_energy_kwh ?? 0),
      0,
    ),
    energy_complete: audits.length > 0 &&
      audits.every((item) => item.efficiency.energy_complete === true),
  };
}

async function main(): Promise<void> {
  if (Deno.args.length === 0) {
    throw new Error(
      "usage: deno run --allow-read --allow-run scripts/audit-harness-usability.ts <scratch-root>...",
    );
  }
  const audits: UsabilityAudit[] = [];
  for (const scratchRoot of Deno.args) {
    const audit = await auditScratch(scratchRoot);
    audits.push(audit);
    console.log(JSON.stringify(audit));
  }
  console.log(JSON.stringify({ aggregate: aggregateAudits(audits) }));
}

if (import.meta.main) {
  try {
    await main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}
