# Task-completion qualification fixtures

These fixtures measure whether a real local model can finish pb's complete strict delivery
workflow. They are artifact-quality experiments, not substitutes for the deterministic harness
control corpus.

Both fixtures are self-contained and require no network access. Their contract checks are trusted
shell commands embedded in the contract, so the harness workspace can remain a fresh empty Git
repository and the agent cannot edit the verifier.

## TC1 · ordered creation

TC1 qualifies the harness-owned next-missing-path controller with two small exact files. It passes
only when both files exist with the exact requested content, the named check passes, a fresh review
reads both paths and cites the check, pb creates a semantic commit, the worktree is clean, and the
terminal result is `contract_status=satisfied` with `verified_completed=true`.

Run from the pb repository with a new empty scratch path:

```bash
target/aarch64-apple-darwin/release/pb harness agent \
  --scratch-dir /private/tmp/pb-tc1-ordered-create-1 \
  --contract fixtures/harness-task-completion/tc1-contract.json \
  --max-steps 4 --max-tokens 1024 --temperature 0 --top-k 1 --seed 0 \
  "$(cat fixtures/harness-task-completion/tc1-task.txt)"
```

Use a different scratch directory for each of three locked trials. Preserve every scratch root,
including failures. The first native run qualifies the controller; three successful unchanged runs
establish the TC1 repeatability gate.

## TC2 · useful local coding

TC2 asks the model to implement, test, document, review, check, and commit a small dependency-free
JavaScript module. `behavior` uses independent examples embedded in the trusted contract;
`model_tests` separately executes the tests written by the model.

```bash
target/aarch64-apple-darwin/release/pb harness agent \
  --scratch-dir /private/tmp/pb-tc2-slugify-1 \
  --contract fixtures/harness-task-completion/tc2-contract.json \
  --max-steps 6 --max-tokens 1536 --temperature 0 --top-k 1 --seed 0 \
  "$(cat fixtures/harness-task-completion/tc2-task.txt)"
```

TC2 is attempted only after TC1's first native qualification. A successful run must satisfy both
checks and every strict workflow gate; attractive source or model-authored tests alone do not count.

## Independent review

After every run:

1. Read the run-local `journal.md` and `events.jsonl` named by `run-index.jsonl`.
2. Inspect the scratch workspace branch, log, status, and diff against the baseline commit.
3. Re-run each contract command from the scratch workspace.
4. Confirm the recorded commit OID is `HEAD` and `git status --porcelain` is empty.
5. Record wall time, energy, invocations, prompt/generated tokens, checks, stage sequence, rejected
   actions, commit OID, contract status, and verified-completion state.

The checked-in summarizer emits those recorded fields as JSONL without promoting them to independent
verification:

```bash
deno run --allow-read scripts/summarize-harness-completion.ts \
  /private/tmp/pb-tc1-ordered-create-1
```

`reported_completion=true` means the harness recorded both a satisfied contract and verified
completion. The supervisor must still perform the independent check and Git audit above.

Protocol containment and task completion remain separate outcomes. A safely rejected or incomplete
artifact is useful harness evidence, but it is not a TC1 or TC2 completion.
