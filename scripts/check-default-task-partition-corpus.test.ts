import { deepEqual, equal, ok } from "node:assert/strict";
import { describe, it } from "node:test";

interface PartitionCase {
  id: string;
  category: "single_build" | "multi_task" | "ordered_multi_task";
  prompt: string;
  paired_build_prompt: string;
  expected: {
    route: "one_build" | "multi_task";
    min_tasks: number;
    max_tasks: number;
    order_terms?: [string, string][];
  };
}

const fixture = JSON.parse(
  await Deno.readTextFile(
    new URL("../fixtures/task-decomposition/default-partition-corpus.json", import.meta.url),
  ),
) as { version: number; cases: PartitionCase[] };

describe("default Task-partition corpus", () => {
  it("locks the paired default-routing shapes", () => {
    equal(fixture.version, 1);
    equal(fixture.cases.length, 6);
    deepEqual(
      new Set(fixture.cases.map((entry) => entry.category)),
      new Set(["single_build", "multi_task", "ordered_multi_task"]),
    );
    equal(
      fixture.cases.filter((entry) => entry.expected.route === "one_build").length,
      2,
    );
  });

  it("keeps every direct/decomposed comparison bounded and executable", () => {
    for (const entry of fixture.cases) {
      ok(entry.id.length > 0);
      ok(entry.prompt.length > 20);
      ok(entry.paired_build_prompt.length > 20);
      ok(entry.expected.min_tasks >= 1);
      ok(entry.expected.max_tasks >= entry.expected.min_tasks);
      ok(entry.expected.max_tasks <= 6);
      if (entry.expected.route === "one_build") {
        equal(entry.expected.min_tasks, 1);
        equal(entry.expected.max_tasks, 1);
      }
      for (const pair of entry.expected.order_terms ?? []) {
        equal(pair.length, 2);
        ok(pair.every((term) => entry.prompt.toLowerCase().includes(term)));
      }
    }
  });
});
