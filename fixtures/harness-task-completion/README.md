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
JavaScript module. `behavior` uses independent examples embedded in the trusted contract,
`dependency_free` rejects remote or package imports, and `model_tests` separately executes the tests
written by the model.

```bash
target/aarch64-apple-darwin/release/pb harness agent \
  --scratch-dir /private/tmp/pb-tc2-slugify-1 \
  --contract fixtures/harness-task-completion/tc2-contract.json \
  --max-steps 6 --max-tokens 1536 --temperature 0 --top-k 1 --seed 0 \
  "$(cat fixtures/harness-task-completion/tc2-task.txt)"
```

TC2 is attempted only after TC1's first native qualification. A successful run must satisfy all
three checks and every strict workflow gate; attractive source or model-authored tests alone do not
count.

## TC3 · offline repository corpus

`corpus.json` defines 11 dependency-free cases across ordered creation, repair after a failed check,
one-file fixes, regression tests, related multi-file work, failed-check diagnosis, delete/modify,
scope resistance, mixed create/modify, resumed partial work, and truthful no-change completion. The
manifest embeds seeded files, trusted contracts, and locked per-case model bounds.

List the cases without executing a model:

```bash
deno run --allow-read scripts/run-harness-task-corpus.ts --list
```

Materialize and inspect one case without inference:

```bash
deno run --allow-read --allow-write --allow-run \
  scripts/run-harness-task-corpus.ts \
  --case resume_partial_case_helpers \
  --scratch-dir /private/tmp/pb-corpus-resume-prepare-1 \
  --prepare-only
```

Omit `--prepare-only` to run the current release binary. `--binary` selects a different pb binary.
Every invocation requires a new scratch path; preserve it after the run. A case with `resume_files`
writes an immutable clean task baseline before applying the earlier uncommitted work, so pb records
the adopted paths through its normal resumed-scratch path.

The runner prepares and executes one case at a time. It deliberately does not turn prepared inputs
into a pass claim: TC3 promotion still requires independent check/Git audits and an aggregate report
covering completion, safety, latency, energy, tokens, invocations, and repair cycles.

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
