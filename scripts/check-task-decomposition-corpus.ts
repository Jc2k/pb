export const REQUIRED_CASE_IDS = [
  "valid_single_build",
  "valid_single_goal_explicit",
  "valid_multi_task",
  "dependency_cycle",
  "unknown_dependency",
  "missing_requirement_coverage",
  "unknown_acceptance_reference",
  "model_authored_numeric_budget",
  "unqualified_goal_selection",
  "migration_order_requires_review",
  "catch_all_requires_review",
  "two_invalid_attempts",
] as const;

const OUTCOMES = new Set([
  "accepted",
  "validation_rejected",
  "review_rejected",
  "task_plan_rejected",
]);
const DISPATCHES = new Set(["build", "goal", "multi_task"]);

type JsonObject = Record<string, unknown>;

export interface TaskDecompositionCase {
  id: string;
  hypothesis: string;
  category: "deterministic" | "semantic" | "attempt_limit";
  context: JsonObject;
  attempts: JsonObject[];
  expected: {
    outcome: string;
    reason: string;
    dispatch?: string;
  };
}

export interface TaskDecompositionCorpus {
  version: number;
  cases: TaskDecompositionCase[];
}

function object(value: unknown, label: string): JsonObject {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JsonObject;
}

function nonEmptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

export function parseTaskDecompositionCorpusText(
  text: string,
): TaskDecompositionCorpus {
  const root = object(JSON.parse(text), "corpus");
  if (root.version !== 1) {
    throw new Error(
      `unsupported Task decomposition corpus version ${String(root.version)}`,
    );
  }
  if (!Array.isArray(root.cases) || root.cases.length === 0) {
    throw new Error("Task decomposition corpus must contain cases");
  }

  const ids = new Set<string>();
  const cases = root.cases.map((raw, index): TaskDecompositionCase => {
    const value = object(raw, `cases[${index}]`);
    const id = nonEmptyString(value.id, `cases[${index}].id`);
    if (ids.has(id)) {
      throw new Error(`duplicate Task decomposition case '${id}'`);
    }
    ids.add(id);
    const category = nonEmptyString(value.category, `${id}.category`);
    if (!["deterministic", "semantic", "attempt_limit"].includes(category)) {
      throw new Error(`${id}.category is unsupported`);
    }
    if (
      !Array.isArray(value.attempts) || value.attempts.length === 0 ||
      value.attempts.length > 2
    ) {
      throw new Error(`${id}.attempts must contain one or two proposals`);
    }
    const attempts = value.attempts.map((attempt, attemptIndex) =>
      object(attempt, `${id}.attempts[${attemptIndex}]`)
    );
    const expected = object(value.expected, `${id}.expected`);
    const outcome = nonEmptyString(expected.outcome, `${id}.expected.outcome`);
    if (!OUTCOMES.has(outcome)) {
      throw new Error(`${id}.expected.outcome is unsupported`);
    }
    const dispatch = expected.dispatch;
    if (outcome === "accepted") {
      if (typeof dispatch !== "string" || !DISPATCHES.has(dispatch)) {
        throw new Error(
          `${id}.expected.dispatch is required for accepted cases`,
        );
      }
    } else if (dispatch !== undefined) {
      throw new Error(
        `${id}.expected.dispatch is forbidden for rejected cases`,
      );
    }
    return {
      id,
      hypothesis: nonEmptyString(value.hypothesis, `${id}.hypothesis`),
      category: category as TaskDecompositionCase["category"],
      context: object(value.context, `${id}.context`),
      attempts,
      expected: {
        outcome,
        reason: nonEmptyString(expected.reason, `${id}.expected.reason`),
        ...(typeof dispatch === "string" ? { dispatch } : {}),
      },
    };
  });

  for (const id of REQUIRED_CASE_IDS) {
    if (!ids.has(id)) {
      throw new Error(
        `Task decomposition corpus is missing required case '${id}'`,
      );
    }
  }
  return { version: 1, cases };
}

if (import.meta.main) {
  const path = Deno.args[0] ?? "fixtures/task-decomposition/corpus.json";
  const corpus = parseTaskDecompositionCorpusText(
    await Deno.readTextFile(path),
  );
  console.log(`checked ${corpus.cases.length} Task decomposition cases`);
}
