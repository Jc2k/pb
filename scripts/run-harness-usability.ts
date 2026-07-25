import { validatedUsabilityCorpus } from "./check-harness-usability-corpus.ts";
import { prepareCorpusCase } from "./run-harness-task-corpus.ts";

interface Options {
  scratchParent?: string;
  binary: string;
  caseIds: string[];
  list: boolean;
}

function parseOptions(args: string[]): Options {
  let scratchParent: string | undefined;
  let binary = "target/aarch64-apple-darwin/release/pb";
  const caseIds: string[] = [];
  let list = false;
  for (let index = 0; index < args.length; index++) {
    const argument = args[index];
    const value = args[index + 1];
    if (argument === "--scratch-parent" && value) {
      scratchParent = value;
      index++;
    } else if (argument === "--binary" && value) {
      binary = value;
      index++;
    } else if (argument === "--case" && value) {
      caseIds.push(value);
      index++;
    } else if (argument === "--list") {
      list = true;
    } else {
      throw new Error(`${argument} requires a value or is unknown`);
    }
  }
  if (!list && !scratchParent) {
    throw new Error("--scratch-parent is required unless --list is used");
  }
  return { scratchParent, binary, caseIds, list };
}

async function main(): Promise<void> {
  const options = parseOptions(Deno.args);
  const corpus = validatedUsabilityCorpus();
  if (options.list) {
    for (const item of corpus.cases) {
      console.log(
        `${item.id}\t${item.language}\t${item.category}\t${item.task}`,
      );
    }
    return;
  }
  const selected = options.caseIds.length === 0
    ? corpus.cases
    : corpus.cases.filter((item) => options.caseIds.includes(item.id));
  if (selected.length !== (options.caseIds.length || corpus.cases.length)) {
    const found = new Set(selected.map((item) => item.id));
    const missing = options.caseIds.filter((item) => !found.has(item));
    throw new Error(`unknown cases: ${missing.join(", ")}`);
  }
  const scratchParent = options.scratchParent!;
  await Deno.mkdir(scratchParent, { recursive: true });
  const resultPath = `${scratchParent}/run-results.jsonl`;
  let failures = 0;
  for (const corpusCase of selected) {
    const scratchRoot = `${scratchParent}/${corpusCase.id}`;
    const startedAt = new Date().toISOString();
    console.error(`running ${corpusCase.id} (${corpusCase.language})`);
    const prepared = await prepareCorpusCase(corpusCase, scratchRoot);
    const status = await new Deno.Command(options.binary, {
      args: [
        "harness",
        "agent",
        "--scratch-dir",
        prepared.scratch_root,
        "--contract",
        prepared.contract,
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
        corpusCase.task,
      ],
      stdin: "null",
      stdout: "inherit",
      stderr: "inherit",
    }).spawn().status;
    const result = {
      version: 1,
      case_id: corpusCase.id,
      language: corpusCase.language,
      scratch_root: scratchRoot,
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      exit_code: status.code,
    };
    await Deno.writeTextFile(resultPath, `${JSON.stringify(result)}\n`, {
      append: true,
      create: true,
    });
    console.log(JSON.stringify(result));
    if (!status.success) failures++;
  }
  if (failures > 0) {
    throw new Error(
      `${failures} of ${selected.length} harness runs exited nonzero`,
    );
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
