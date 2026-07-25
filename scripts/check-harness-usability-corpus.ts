import corpusDefinition from "../fixtures/harness-usability/corpus.ts";
import {
  CorpusCase,
  prepareCorpusCase,
  validateCorpus,
} from "./run-harness-task-corpus.ts";

export interface CommandResult {
  success: boolean;
  code: number;
  output: string;
}

export interface FixtureQualification {
  case_id: string;
  language: string;
  initial_failed: boolean;
  reference_passed: boolean;
  changed_paths: string[];
}

const decoder = new TextDecoder();

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function behaviorCommand(corpusCase: CorpusCase): string {
  const checks = corpusCase.contract.checks;
  assert(
    Array.isArray(checks) && checks.length === 1,
    `${corpusCase.id}: check`,
  );
  const command = (checks[0] as Record<string, unknown>).command;
  assert(typeof command === "string", `${corpusCase.id}: check command`);
  return command;
}

export async function runBehaviorCheck(
  corpusCase: CorpusCase,
  workspace: string,
): Promise<CommandResult> {
  const output = await new Deno.Command("sh", {
    args: ["-lc", behaviorCommand(corpusCase)],
    cwd: workspace,
    stdout: "piped",
    stderr: "piped",
  }).output();
  const combined = `${decoder.decode(output.stdout)}${
    decoder.decode(output.stderr)
  }`;
  return {
    success: output.success,
    code: output.code,
    output: combined.slice(-4_000),
  };
}

async function changedPaths(workspace: string): Promise<string[]> {
  const output = await new Deno.Command("git", {
    args: ["diff", "--name-only", "HEAD"],
    cwd: workspace,
  }).output();
  assert(output.success, "git diff failed");
  return decoder.decode(output.stdout).trim().split("\n").filter(Boolean)
    .sort();
}

async function dirtyPaths(workspace: string): Promise<string[]> {
  const output = await new Deno.Command("git", {
    args: ["status", "--porcelain=v1", "--untracked-files=all"],
    cwd: workspace,
  }).output();
  assert(output.success, "git status failed");
  return decoder.decode(output.stdout).split("\n").filter((line) =>
    line.trim().length > 0
  ).map((line) => line.slice(3).split(" -> ").at(-1)!).sort();
}

async function applyReference(
  corpusCase: CorpusCase,
  workspace: string,
): Promise<void> {
  for (const reference of corpusCase.reference_files) {
    await Deno.writeTextFile(
      `${workspace}/${reference.path}`,
      reference.content,
    );
  }
}

export function validatedUsabilityCorpus() {
  const corpus = validateCorpus(corpusDefinition);
  assert(corpus.cases.length === 24, "usability corpus must contain 24 cases");
  const expectedLanguages = ["python", "react_typescript", "rust"];
  const languages = [...new Set(corpus.cases.map((item) => item.language))]
    .sort();
  assert(
    JSON.stringify(languages) === JSON.stringify(expectedLanguages),
    `unexpected languages: ${languages.join(", ")}`,
  );
  for (const language of expectedLanguages) {
    assert(
      corpus.cases.filter((item) => item.language === language).length === 8,
      `${language}: expected eight cases`,
    );
  }
  for (const corpusCase of corpus.cases) {
    assert(corpusCase.source, `${corpusCase.id}: source provenance`);
    assert(
      corpusCase.source.adaptation.endsWith("_derived"),
      `${corpusCase.id}: derived source`,
    );
    assert(
      corpusCase.reference_files.length > 0,
      `${corpusCase.id}: reference files`,
    );
    const allowed = corpusCase.contract.allowed_paths;
    assert(Array.isArray(allowed), `${corpusCase.id}: allowed paths`);
    const references = corpusCase.reference_files.map((item) => item.path)
      .sort();
    assert(
      JSON.stringify([...allowed].sort()) === JSON.stringify(references),
      `${corpusCase.id}: reference paths must equal allowed paths`,
    );
    const guidance = corpusCase.contract.work_unit_guidance;
    assert(
      guidance !== null && typeof guidance === "object" &&
        Object.keys(guidance).length === 0,
      `${corpusCase.id}: task-specific guidance is forbidden`,
    );
  }
  return corpus;
}

export async function qualifyFixture(
  corpusCase: CorpusCase,
  scratchRoot: string,
): Promise<FixtureQualification> {
  const prepared = await prepareCorpusCase(corpusCase, scratchRoot);
  const initial = await runBehaviorCheck(corpusCase, prepared.workspace);
  if (initial.success) {
    throw new Error(`${corpusCase.id}: initial fixture unexpectedly passed`);
  }
  const initialDirty = await dirtyPaths(prepared.workspace);
  if (initialDirty.length > 0) {
    throw new Error(
      `${corpusCase.id}: official check dirtied baseline: ${
        initialDirty.join(", ")
      }`,
    );
  }
  await applyReference(corpusCase, prepared.workspace);
  const reference = await runBehaviorCheck(corpusCase, prepared.workspace);
  if (!reference.success) {
    throw new Error(
      `${corpusCase.id}: reference failed (${reference.code})\n${reference.output}`,
    );
  }
  const paths = await changedPaths(prepared.workspace);
  const expected = corpusCase.reference_files.map((item) => item.path).sort();
  assert(
    JSON.stringify(paths) === JSON.stringify(expected),
    `${corpusCase.id}: unexpected reference delta: ${paths.join(", ")}`,
  );
  const dirty = await dirtyPaths(prepared.workspace);
  assert(
    JSON.stringify(dirty) === JSON.stringify(expected),
    `${corpusCase.id}: reference check added workspace artifacts: ${
      dirty.join(", ")
    }`,
  );
  return {
    case_id: corpusCase.id,
    language: corpusCase.language ?? "unknown",
    initial_failed: true,
    reference_passed: true,
    changed_paths: paths,
  };
}

async function main(): Promise<void> {
  let caseId: string | undefined;
  let preserveParent: string | undefined;
  for (let index = 0; index < Deno.args.length; index++) {
    if (Deno.args[index] === "--case") caseId = Deno.args[++index];
    else if (Deno.args[index] === "--preserve") {
      preserveParent = Deno.args[++index];
    } else throw new Error(`unknown argument: ${Deno.args[index]}`);
  }
  const corpus = validatedUsabilityCorpus();
  const cases = caseId
    ? corpus.cases.filter((item) => item.id === caseId)
    : corpus.cases;
  assert(cases.length > 0, `unknown case: ${caseId}`);
  const parent = preserveParent ?? await Deno.makeTempDir();
  if (preserveParent) await Deno.mkdir(parent, { recursive: true });
  try {
    for (const corpusCase of cases) {
      const result = await qualifyFixture(
        corpusCase,
        `${parent}/${corpusCase.id}`,
      );
      console.log(JSON.stringify(result));
    }
  } finally {
    if (!preserveParent) await Deno.remove(parent, { recursive: true });
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
