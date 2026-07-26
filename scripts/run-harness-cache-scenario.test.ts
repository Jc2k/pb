import corpusDefinition from "../fixtures/harness-usability/corpus.ts";
import {
  CACHE_EVAL_ARMS,
  cacheEvalArgv,
  PreparedCacheEvalScenario,
} from "./run-harness-cache-scenario.ts";
import { validateCorpus } from "./run-harness-task-corpus.ts";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

Deno.test("cache scenario preserves the six-arm order and locked sampler", () => {
  const corpusCase = validateCorpus(corpusDefinition).cases.find((item) =>
    item.id === "rust_registry_removal"
  );
  assert(corpusCase, "rust registry case");
  const scenario: PreparedCacheEvalScenario = {
    version: 1,
    case_id: corpusCase.id,
    task: corpusCase.task,
    scratch_parent: "/tmp/cache-scenario",
    cache_dir: "/tmp/cache-scenario/inference-cache",
    report: "/tmp/cache-scenario/cache-eval-report.json",
    manifest: "/tmp/cache-scenario/cache-eval-scenario.json",
    arms: CACHE_EVAL_ARMS.map((id) => ({
      id,
      scratch_dir: `/tmp/cache-scenario/${id}`,
      contract: `/tmp/cache-scenario/${id}/contract.json`,
    })),
  };
  const argv = cacheEvalArgv(scenario, corpusCase);
  const scratchValues = argv.flatMap((value, index) =>
    value === "--scratch-dir" ? [argv[index + 1]] : []
  );
  const contractValues = argv.flatMap((value, index) =>
    value === "--contract" ? [argv[index + 1]] : []
  );
  assert(
    scratchValues.join("\n") ===
      scenario.arms.map((arm) => arm.scratch_dir).join("\n"),
    "scratch arm order",
  );
  assert(
    contractValues.join("\n") ===
      scenario.arms.map((arm) => arm.contract).join("\n"),
    "contract arm order",
  );
  assert(argv.includes("read_file"), "authority-changing tool");
  assert(argv.at(-6) === "--temperature", "locked sampler position");
  assert(argv.at(-5) === "0", "zero temperature");
  assert(argv.at(-4) === "--top-k", "top-k position");
  assert(argv.at(-3) === "1", "top-k one");
  assert(argv.at(-2) === "--seed", "seed position");
  assert(argv.at(-1) === "0", "zero seed");
});
