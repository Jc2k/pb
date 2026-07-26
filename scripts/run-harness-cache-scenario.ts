import { validatedUsabilityCorpus } from "./check-harness-usability-corpus.ts";
import { CorpusCase, prepareCorpusCase } from "./run-harness-task-corpus.ts";

export const CACHE_EVAL_ARMS = [
  "cold_empty_storage",
  "warm_same_session_same_process",
  "warm_new_session_same_process",
  "changed_authority_same_process",
  "matching_authority_same_process",
  "persisted_root_new_process",
] as const;

export interface PreparedCacheEvalScenario {
  version: 1;
  case_id: string;
  task: string;
  scratch_parent: string;
  cache_dir: string;
  report: string;
  manifest: string;
  arms: Array<{
    id: typeof CACHE_EVAL_ARMS[number];
    scratch_dir: string;
    contract: string;
  }>;
}

interface Options {
  scratchParent: string;
  binary: string;
  caseId: string;
  prepareOnly: boolean;
}

function parseOptions(args: string[]): Options {
  let scratchParent: string | undefined;
  let binary = "target/aarch64-apple-darwin/release/pb";
  let caseId = "rust_registry_removal";
  let prepareOnly = false;
  for (let index = 0; index < args.length; index++) {
    const argument = args[index];
    if (argument === "--prepare-only") {
      prepareOnly = true;
      continue;
    }
    if (
      argument === "--scratch-parent" || argument === "--binary" ||
      argument === "--case"
    ) {
      const value = args[++index];
      if (!value) throw new Error(`${argument} requires a value`);
      if (argument === "--scratch-parent") scratchParent = value;
      else if (argument === "--binary") binary = value;
      else caseId = value;
      continue;
    }
    throw new Error(`unknown argument: ${argument}`);
  }
  if (!scratchParent) throw new Error("--scratch-parent is required");
  return { scratchParent, binary, caseId, prepareOnly };
}

export function cacheEvalArgv(
  scenario: PreparedCacheEvalScenario,
  corpusCase: CorpusCase,
): string[] {
  const args = ["harness", "cache-eval", scenario.task];
  for (const arm of scenario.arms) {
    args.push("--scratch-dir", arm.scratch_dir);
  }
  for (const arm of scenario.arms) {
    args.push("--contract", arm.contract);
  }
  args.push(
    "--cache-dir",
    scenario.cache_dir,
    "--output",
    scenario.report,
    "--changed-authority-tool",
    "read_file",
    "--max-steps",
    String(corpusCase.limits.max_steps),
    "--max-tokens",
    String(corpusCase.limits.max_tokens),
    "--temperature",
    "0",
    "--top-k",
    "1",
    "--seed",
    "0",
  );
  return args;
}

export async function prepareCacheEvalScenario(
  corpusCase: CorpusCase,
  requestedParent: string,
): Promise<PreparedCacheEvalScenario> {
  await Deno.mkdir(requestedParent, { recursive: true });
  const scratchParent = await Deno.realPath(requestedParent);
  const cacheDir = `${scratchParent}/inference-cache`;
  const report = `${scratchParent}/cache-eval-report.json`;
  const manifest = `${scratchParent}/cache-eval-scenario.json`;
  for (const reserved of [cacheDir, report, manifest]) {
    try {
      await Deno.lstat(reserved);
      throw new Error(
        `cache scenario reserved path already exists: ${reserved}`,
      );
    } catch (error) {
      if (!(error instanceof Deno.errors.NotFound)) throw error;
    }
  }

  const arms: PreparedCacheEvalScenario["arms"] = [];
  for (const id of CACHE_EVAL_ARMS) {
    const prepared = await prepareCorpusCase(
      corpusCase,
      `${scratchParent}/${id}`,
    );
    arms.push({
      id,
      scratch_dir: prepared.scratch_root,
      contract: prepared.contract,
    });
  }
  return {
    version: 1,
    case_id: corpusCase.id,
    task: corpusCase.task,
    scratch_parent: scratchParent,
    cache_dir: cacheDir,
    report,
    manifest,
    arms,
  };
}

async function commandText(command: string, args: string[]): Promise<string> {
  const output = await new Deno.Command(command, {
    args,
    stdout: "piped",
    stderr: "piped",
  }).output();
  if (!output.success) {
    throw new Error(
      `${command} ${args.join(" ")} failed: ${
        new TextDecoder().decode(output.stderr).trim()
      }`,
    );
  }
  return new TextDecoder().decode(output.stdout).trim();
}

async function sha256File(path: string): Promise<string> {
  const bytes = await Deno.readFile(path);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function main(): Promise<void> {
  const options = parseOptions(Deno.args);
  const corpusCase = validatedUsabilityCorpus().cases.find((item) =>
    item.id === options.caseId
  );
  if (!corpusCase) throw new Error(`unknown usability case: ${options.caseId}`);
  const scenario = await prepareCacheEvalScenario(
    corpusCase,
    options.scratchParent,
  );
  const argv = cacheEvalArgv(scenario, corpusCase);
  const binary = await Deno.realPath(options.binary);
  const revision = await commandText("git", ["rev-parse", "HEAD"]);
  const status = await commandText("git", [
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
  ]);
  const manifest = {
    ...scenario,
    prepared_at: new Date().toISOString(),
    revision,
    source_worktree_clean: status.length === 0,
    binary,
    binary_sha256: await sha256File(binary),
    machine_class: `${Deno.build.os}-${Deno.build.arch}`,
    argv,
  };
  await Deno.writeTextFile(
    scenario.manifest,
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  console.log(JSON.stringify(manifest));
  if (options.prepareOnly) return;

  const runStatus = await new Deno.Command(binary, {
    args: argv,
    stdin: "null",
    stdout: "inherit",
    stderr: "inherit",
  }).spawn().status;
  await Deno.writeTextFile(
    scenario.manifest,
    `${
      JSON.stringify(
        {
          ...manifest,
          finished_at: new Date().toISOString(),
          exit_code: runStatus.code,
        },
        null,
        2,
      )
    }\n`,
  );
  if (!runStatus.success) {
    throw new Error(`cache scenario exited with status ${runStatus.code}`);
  }
}

if (import.meta.main) {
  try {
    await main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}
