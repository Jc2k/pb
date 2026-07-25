import { prepareCorpusCase } from "./run-harness-task-corpus.ts";
import { validatedUsabilityCorpus } from "./check-harness-usability-corpus.ts";

async function main(): Promise<void> {
  const corpus = validatedUsabilityCorpus();
  const react = corpus.cases.find((item) =>
    item.language === "react_typescript"
  );
  if (!react) throw new Error("React usability fixture is missing");
  const parent = await Deno.makeTempDir();
  try {
    const prepared = await prepareCorpusCase(react, `${parent}/react-cache`);
    const status = await new Deno.Command(Deno.execPath(), {
      args: ["cache", "tests/component_test.tsx"],
      cwd: prepared.workspace,
      stdin: "null",
      stdout: "inherit",
      stderr: "inherit",
    }).spawn().status;
    if (!status.success) throw new Error("failed to cache React dependencies");
  } finally {
    await Deno.remove(parent, { recursive: true });
  }
  console.log("React usability dependencies are cached for offline checks.");
}

if (import.meta.main) {
  try {
    await main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}
