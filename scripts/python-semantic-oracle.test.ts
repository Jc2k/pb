import {
  compareOracleResult,
  materializeOracleRequest,
  type PythonOracleRequest,
  type PythonOracleResult,
} from "./python-semantic-oracle.ts";

const CORPUS =
  new URL("../fixtures/control-collar/semantic-python-v1.json", import.meta.url)
    .pathname;
const ZERO_SHA256 = "0".repeat(64);

function matchingResult(request: PythonOracleRequest): PythonOracleResult {
  return {
    version: 1,
    corpus_sha256: request.corpus_sha256,
    provider: {
      name: "independent_checker",
      version: "1.2.3",
      configuration_sha256: ZERO_SHA256,
    },
    cases: request.cases.map((item) => ({
      id: item.id,
      outcome: item.expected_outcome,
      diagnostic_ids: [],
    })),
  };
}

Deno.test("materializes complete baseline and every mutation shape deterministically", async () => {
  const owner = await Deno.makeTempDir({ prefix: "pb-python-oracle-" });
  try {
    const firstRoot = `${owner}/first`;
    const secondRoot = `${owner}/second`;
    const first = await materializeOracleRequest(CORPUS, firstRoot);
    const second = await materializeOracleRequest(CORPUS, secondRoot);
    if (JSON.stringify(first) !== JSON.stringify(second)) {
      throw new Error("materialized requests are not deterministic");
    }
    if (first.cases.length !== 24 || first.corpus_sha256.length !== 64) {
      throw new Error("materialized request lost corpus identity or cases");
    }

    const created = await Deno.readTextFile(
      `${firstRoot}/cases/valid_new_module/project/created_valid.py`,
    );
    if (!created.includes("def created")) {
      throw new Error("write_file case was not materialized");
    }
    const edited = await Deno.readTextFile(
      `${firstRoot}/cases/annotated_invalid_assignment/project/annotated.py`,
    );
    if (!edited.includes('value: int = "bad"')) {
      throw new Error("edit_file case was not materialized");
    }
    const patched = await Deno.readTextFile(
      `${firstRoot}/cases/coherent_signature_transaction/project/consumer.py`,
    );
    if (!patched.includes("result: int = render(1)")) {
      throw new Error("apply_patch case was not materialized");
    }
    const baseline = await Deno.readTextFile(
      `${firstRoot}/baseline/project/consumer.py`,
    );
    if (!baseline.includes('result: str = render("ok")')) {
      throw new Error("baseline was mutated");
    }
    const dependency = await Deno.readTextFile(
      `${firstRoot}/dependencies/httpx/__init__.pyi`,
    );
    if (!dependency.includes("def get")) {
      throw new Error("dependency root was not materialized");
    }
  } finally {
    await Deno.remove(owner, { recursive: true });
  }
});

Deno.test("comparison separates agreement disagreement and unknown without choosing an authority", async () => {
  const owner = await Deno.makeTempDir({ prefix: "pb-python-oracle-" });
  try {
    const request = await materializeOracleRequest(CORPUS, `${owner}/request`);
    const result = matchingResult(request);
    result.cases[0].outcome = result.cases[0].outcome === "allow"
      ? "reject"
      : "allow";
    result.cases[1].outcome = "unknown";
    const report = compareOracleResult(request, result);
    if (
      report.case_count !== 24 ||
      report.agreement_count !== 22 ||
      report.disagreement_count !== 1 ||
      report.unknown_count !== 1 ||
      report.passed
    ) {
      throw new Error(
        `unexpected oracle comparison report: ${JSON.stringify(report)}`,
      );
    }
    if (
      report.disagreement_ids[0] !== request.cases[0].id ||
      report.unknown_ids[0] !== request.cases[1].id
    ) {
      throw new Error("oracle comparison lost deterministic request order");
    }
  } finally {
    await Deno.remove(owner, { recursive: true });
  }
});

Deno.test("comparison rejects stale incomplete or content-bearing-shaped artifacts", async () => {
  const owner = await Deno.makeTempDir({ prefix: "pb-python-oracle-" });
  try {
    const request = await materializeOracleRequest(CORPUS, `${owner}/request`);
    const stale = matchingResult(request);
    stale.corpus_sha256 = "f".repeat(64);
    let staleRejected = false;
    try {
      compareOracleResult(request, stale);
    } catch {
      staleRejected = true;
    }
    if (!staleRejected) throw new Error("stale oracle result was accepted");

    const incomplete = matchingResult(request);
    incomplete.cases.pop();
    let incompleteRejected = false;
    try {
      compareOracleResult(request, incomplete);
    } catch {
      incompleteRejected = true;
    }
    if (!incompleteRejected) {
      throw new Error("incomplete oracle result was accepted");
    }

    const reordered = matchingResult(request);
    [reordered.cases[0], reordered.cases[1]] = [
      reordered.cases[1],
      reordered.cases[0],
    ];
    let reorderedRejected = false;
    try {
      compareOracleResult(request, reordered);
    } catch {
      reorderedRejected = true;
    }
    if (!reorderedRejected) {
      throw new Error("reordered oracle result was accepted");
    }

    const messageShaped = matchingResult(request);
    messageShaped.cases[0].diagnostic_ids = [
      "This looks like a diagnostic message",
    ];
    let messageRejected = false;
    try {
      compareOracleResult(request, messageShaped);
    } catch {
      messageRejected = true;
    }
    if (!messageRejected) {
      throw new Error("message-shaped diagnostic identifier was accepted");
    }

    const contentBearing = matchingResult(request) as unknown as Record<
      string,
      unknown
    >;
    contentBearing.source_excerpt = "private source";
    let contentRejected = false;
    try {
      compareOracleResult(request, contentBearing);
    } catch {
      contentRejected = true;
    }
    if (!contentRejected) {
      throw new Error("unknown content-bearing result fields were accepted");
    }
  } finally {
    await Deno.remove(owner, { recursive: true });
  }
});

Deno.test("materialization rejects traversal and symlink-producing corpus mutations", async () => {
  const owner = await Deno.makeTempDir({ prefix: "pb-python-oracle-" });
  try {
    const traversalCorpus = JSON.parse(await Deno.readTextFile(CORPUS));
    traversalCorpus.files[0].path = "../escape.py";
    const traversalPath = `${owner}/traversal.json`;
    await Deno.writeTextFile(traversalPath, JSON.stringify(traversalCorpus));
    let traversalRejected = false;
    try {
      await materializeOracleRequest(
        traversalPath,
        `${owner}/traversal-output`,
      );
    } catch {
      traversalRejected = true;
    }
    if (!traversalRejected) {
      throw new Error("traversing corpus path was materialized");
    }

    const symlinkCorpus = JSON.parse(await Deno.readTextFile(CORPUS));
    symlinkCorpus.cases = [{
      id: "symlink_candidate",
      category: "annotated",
      tool: "apply_patch",
      arguments: {
        patch:
          "diff --git a/link.py b/link.py\nnew file mode 120000\n--- /dev/null\n+++ b/link.py\n@@ -0,0 +1 @@\n+../../outside\n\\ No newline at end of file\n",
      },
      expected: { outcome: "reject", diagnostic_codes: ["invalid-assignment"] },
    }];
    const symlinkPath = `${owner}/symlink.json`;
    await Deno.writeTextFile(symlinkPath, JSON.stringify(symlinkCorpus));
    let symlinkRejected = false;
    try {
      await materializeOracleRequest(symlinkPath, `${owner}/symlink-output`);
    } catch {
      symlinkRejected = true;
    }
    if (!symlinkRejected) {
      throw new Error("symlink-producing corpus patch was materialized");
    }
  } finally {
    await Deno.remove(owner, { recursive: true });
  }
});
