---
name: pb-harness-supervisor
description: Supervise repeatable, goal-driven pb harness experiments that ask pb to build something in an isolated scratch repository, then review its events, journal, commits, and output; independently verify the result; fix and commit clear bugs in pb; rerun when warranted; and produce ranked observations plus a prioritized improvement plan. Use when dogfooding pb agent tasks, evaluating agent quality, validating `pb harness agent`, or asking Codex to oversee a daemon-free pb build task from start to finish.
---

# PB Harness Supervisor

Run `pb harness agent` as an experiment, not as an opaque success/failure command. Preserve every
scratch run, review its evidence, and distinguish defects in pb from defects in the generated project.

## Establish the goal

1. Require a concrete artifact-building task. If it is missing, ask for it before running the harness.
2. Confirm the request explicitly authorizes goal creation. The skill's default prompt does; if a
   different invocation does not, ask before creating a goal.
3. Inspect the current goal. Resume it when it already describes this experiment. Do not replace an
   unrelated active goal without user direction.
4. When no goal exists, create one with this completion contract:
   - pb completes the requested task in a scratch workspace.
   - The result is independently reviewed and tested.
   - Clear defects in pb are fixed and committed semantically in the pb repository.
   - All runs and findings are recorded in a ranked journal.
   - Remaining improvements have a prioritized, evidence-backed plan.
5. Keep the goal active across experiments. Mark it complete only when every contract item is met.

## Prepare pb

1. Work from the pb repository and read its `AGENTS.md`.
2. Inspect `git status` before editing. Preserve unrelated and user-owned changes.
3. Ensure web assets and the release binary reflect the current source:

   ```bash
   deno task build:web
   cargo build --release --target aarch64-apple-darwin
   ```

4. Never start `pb serve` or use `pb queue` for this workflow.

## Run one experiment

Run the release harness as a blocking foreground command:

```bash
target/aarch64-apple-darwin/release/pb harness agent "<task>"
```

Pass model or inference overrides only when the user requests them or the configured default cannot
run. If Metal is unavailable inside the sandbox, request host execution for the same bounded command.
Do not hide the process behind a daemon, background job, or socket API.

Record the printed scratch, workspace, event, and journal paths immediately. Let the harness allocate
a fresh scratch directory for each iteration unless reproducing a run specifically requires an
explicit new `--scratch-dir` path.

## Review the evidence

After every completed, failed, or interrupted run:

1. Read `journal.md` and inspect relevant entries in `events.jsonl`.
2. Inspect the scratch repository branch, `git status`, commits beyond `main`, and full diff.
3. Run the generated project's acceptance checks independently. Do not accept an agent's claim that
   tests passed without checking the available evidence or rerunning them.
4. Compare the result with the literal task. Treat a final answer as evidence, not proof of completion.
5. Read [references/review-rubric.md](references/review-rubric.md) and add deduplicated supervisor
   observations to the run's `journal.md`.

## Fix and rerun

- Fix a problem in the pb repository only when evidence identifies a clear harness, runtime, prompting,
  tool, scheduling, or reporting defect in pb.
- Do not manually repair the generated scratch project to make pb appear successful. Record generated
  output defects as observations and let a subsequent pb run demonstrate the improvement.
- Add focused regression coverage for pb fixes, run checks proportional to risk, and create semantic
  commits such as `fix: ...` or `perf: ...`.
- Rebuild the release binary after a pb fix, then start a fresh scratch experiment with the same task.
- Keep failed and superseded scratch roots. Link each iteration from the latest journal or final report.

## Close the goal

Before completion, ensure the journal contains:

- the outcome and artifact path for every iteration;
- ranked P0-P3 observations with concrete evidence and disposition;
- pb fix commit hashes and verification results;
- a prioritized plan for valid improvements that were not required to complete the task.

Report the successful scratch path, generated-project commits, pb fix commits, tests, unresolved risks,
and plan. Mark the goal complete only after the requested project works and no required pb fix remains.
Use blocked status only under the goal tool's repeated-blocker rules.
