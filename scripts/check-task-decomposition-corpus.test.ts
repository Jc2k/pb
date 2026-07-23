import {
  parseTaskDecompositionCorpusText,
  REQUIRED_CASE_IDS,
} from "./check-task-decomposition-corpus.ts";

function assertEquals(actual: unknown, expected: unknown): void {
  if (actual !== expected) {
    throw new Error(`expected ${String(expected)}, got ${String(actual)}`);
  }
}

function assertThrows(action: () => void, expectedMessage: string): void {
  let message = "";
  try {
    action();
  } catch (error) {
    message = String(error);
  }
  if (!message.includes(expectedMessage)) {
    throw new Error(
      `expected error containing '${expectedMessage}', got '${message}'`,
    );
  }
}

const corpusPath = new URL(
  "../fixtures/task-decomposition/corpus.json",
  import.meta.url,
);

Deno.test("checked-in Task decomposition corpus is complete", async () => {
  const corpus = parseTaskDecompositionCorpusText(
    await Deno.readTextFile(corpusPath),
  );
  assertEquals(corpus.version, 1);
  assertEquals(corpus.cases.length >= REQUIRED_CASE_IDS.length, true);
});

Deno.test("Task decomposition corpus rejects duplicate ids", () => {
  const caseValue = {
    id: "valid_single_build",
    hypothesis: "fixture",
    category: "deterministic",
    context: {},
    attempts: [{}],
    expected: { outcome: "accepted", reason: "fixture", dispatch: "build" },
  };
  assertThrows(
    () =>
      parseTaskDecompositionCorpusText(
        JSON.stringify({ version: 1, cases: [caseValue, caseValue] }),
      ),
    "duplicate Task decomposition case",
  );
});

Deno.test("Task decomposition corpus requires dispatch only for accepted cases", () => {
  const cases = REQUIRED_CASE_IDS.map((id) => ({
    id,
    hypothesis: "fixture",
    category: "deterministic",
    context: {},
    attempts: [{}],
    expected: {
      outcome: "validation_rejected",
      reason: "fixture",
      dispatch: "build",
    },
  }));
  assertThrows(
    () =>
      parseTaskDecompositionCorpusText(JSON.stringify({ version: 1, cases })),
    "dispatch is forbidden",
  );
});
