import {
  prepareCorpusCase,
  validateCorpus,
  validateRelativePath,
} from "./run-harness-task-corpus.ts";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

Deno.test("checked-in task corpus is valid and spans ten cases", async () => {
  const parsed = JSON.parse(
    await Deno.readTextFile("fixtures/harness-task-completion/corpus.json"),
  );
  const corpus = validateCorpus(parsed);
  assert(corpus.cases.length >= 10, `${corpus.cases.length}`);
  assert(
    new Set(corpus.cases.map((item) => item.category)).size >= 7,
    "category coverage",
  );
  assert(
    corpus.cases.some((item) => item.seed_files.length === 0),
    "fresh create case",
  );
  assert(
    corpus.cases.some((item) => item.seed_files.length > 0),
    "seeded case",
  );
  assert(
    corpus.cases.some((item) => item.contract.mutation === "forbidden"),
    "no-change case",
  );
  assert(
    corpus.cases.some((item) => item.resume_files.length > 0),
    "resumed-work case",
  );
});

Deno.test("slugify case independently covers the exact normalization rule and registered tests", async () => {
  const parsed = JSON.parse(
    await Deno.readTextFile("fixtures/harness-task-completion/corpus.json"),
  );
  const corpus = validateCorpus(parsed);
  const slugify = corpus.cases.find((item) =>
    item.id === "create_slugify_repair"
  );
  assert(slugify, "slugify case");

  assert(Array.isArray(slugify.contract.checks), "slugify checks");
  const checks = slugify.contract.checks as Array<{
    id?: string;
    command?: string;
  }>;
  const behavior = checks.find((check) =>
    check.id === "behavior"
  );
  assert(behavior, "behavior check");
  assert(
    behavior.command?.includes("hello_world"),
    "underscore coverage",
  );
  assert(behavior.command?.includes("naïve"), "non-ASCII coverage");

  const modelTests = checks.find((check) =>
    check.id === "model_tests"
  );
  assert(modelTests, "model-tests check");
  assert(
    modelTests.command?.includes("Deno\\.test"),
    "registered-test gate",
  );
  assert(modelTests.command?.includes("deno test"), "test execution gate");
});

Deno.test("corpus paths fail closed on traversal", () => {
  let message = "";
  try {
    validateRelativePath("../escape.txt", "seed");
  } catch (error) {
    message = String(error);
  }
  assert(message.includes("stay beneath"), message);
});

Deno.test("prepareCorpusCase creates a clean committed baseline outside the contract", async () => {
  const parent = await Deno.makeTempDir();
  const scratch = `${parent}/case`;
  try {
    const corpus = validateCorpus(JSON.parse(
      await Deno.readTextFile("fixtures/harness-task-completion/corpus.json"),
    ));
    const seeded = corpus.cases.find((item) => item.seed_files.length > 0);
    assert(seeded, "seeded case");
    const prepared = await prepareCorpusCase(seeded, scratch);
    const status = await new Deno.Command("git", {
      args: ["status", "--porcelain"],
      cwd: prepared.workspace,
    }).output();
    assert(status.success, "git status");
    assert(
      new TextDecoder().decode(status.stdout).trim() === "",
      "clean baseline",
    );
    assert((await Deno.stat(prepared.contract)).isFile, "contract file");
    assert((await Deno.stat(`${scratch}/task.txt`)).isFile, "task file");
  } finally {
    await Deno.remove(parent, { recursive: true });
  }
});

Deno.test("prepareCorpusCase preserves a baseline before resumed work", async () => {
  const parent = await Deno.makeTempDir();
  const scratch = `${parent}/case`;
  try {
    const corpus = validateCorpus(JSON.parse(
      await Deno.readTextFile("fixtures/harness-task-completion/corpus.json"),
    ));
    const resumed = corpus.cases.find((item) => item.resume_files.length > 0);
    assert(resumed, "resumed case");
    const prepared = await prepareCorpusCase(resumed, scratch);
    const baseline = JSON.parse(
      await Deno.readTextFile(`${scratch}/task-baseline.json`),
    );
    assert(baseline.head, "baseline head");
    assert(
      Object.keys(baseline.content.paths).length === resumed.seed_files.length,
      "baseline seed paths",
    );
    const status = await new Deno.Command("git", {
      args: ["status", "--porcelain"],
      cwd: prepared.workspace,
    }).output();
    const dirty = new TextDecoder().decode(status.stdout);
    for (const file of resumed.resume_files) {
      assert(dirty.includes(file.path), `adopted path ${file.path}`);
    }
  } finally {
    await Deno.remove(parent, { recursive: true });
  }
});
