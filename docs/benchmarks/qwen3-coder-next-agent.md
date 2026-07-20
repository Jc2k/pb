# Qwen3-Coder-Next Native Agent Evaluation

**Date:** 2026-07-19

**Host:** 64 GiB Apple Silicon Mac, macOS 26.5.1

**Model:** `hf://mlx-community/Qwen3-Coder-Next-4bit`
**Runtime:** built-in FlashMoe, 48 layers, 512 routed experts per layer, K=10

This record evaluates the largest Qwen3-Coder-Next checkpoint that pb can keep resident on this
host. It also records the lower-memory behavior and a bounded browser-game agent task. Explicit
GGUF remains a llama.cpp control path; no native failure in these runs was allowed to fall back to
it.

## Native inference qualification

- The nine-shard MLX affine-Q4 source prepares a `flashmoe-v2-qwen3-next-mlxq4` cache with a
  40.5 GiB packed expert corpus.
- The sampled resident decision used a 50,096,509,748-byte working limit. Dense weights accounted
  for 1,405,308,928 bytes, recurrent state for 80,815,104 bytes, and complete reusable expert
  wrappers for 43,486,543,872 bytes. Resident inference issued no expert reads.
- With `--metal-working-set-limit-mib 32768`, graph preparation selected the existing streamed
  scheduler. Startup took 277 ms and only ten active expert slots (17,694,720 reusable wrapper
  bytes) were retained; it did not create a partial application cache or alternate scheduler.
- Deterministic no-thinking inference for `What is 2+2? Answer with only the number.` returned
  `4`. Raw prompt `a` returned `=`, matching the official MLX-LM reference for the same source.
- A disposable explicit Qwen3-Coder-Next GGUF control returned `4` through llama.cpp, confirming
  that the explicit compatibility backend still functions independently.

The decisive correctness defect was norm ownership. The supported MLX conversion has already
applied Qwen3-Next's upstream `1 + weight` sanitization to decoder and Q/K norm tensors. Applying
the offset again in FlashMoe caused the first layer to diverge. Consuming those prepared tensors as
ordinary multiplicative weights restored layer parity and correct output.

## Agent-control controls

The deterministic small-model control suite passed all four scripted cases: false final after
inspection, repeated blocked action, final at step limit, and review missing a required check. The
effective Qwen3-Coder-Next policy reports thinking disabled. Independent native or compatibility
tool calls may still be returned as one batch; only order-dependent actions and workflow
transitions require separate turns.

## Browser-game field experiment

The contract allows only `index.html`, `styles.css`, `game.js`, and `game-logic.test.mjs`; requires
`deno test game-logic.test.mjs`; requires fresh review of all four files; and requires a semantic
commit with a clean worktree.

The first native run (`1784489079852-26237-0`) proved the resident graph but exposed a tool-schema
compatibility failure. Its cold planning completion took 690,173 ms for 4,354 prompt tokens and
515 generated tokens (4.70 Wh at 24.5 W). Qwen produced a coherent plan but encoded the typed
artifact as a JSON string. The next turn reused 4,869 prompt tokens, prefilling only 134, but
repeated the same rejected call. A third turn called unavailable `ask_user`. This is classified as
a pb interoperability defect followed by a model correction failure, not a scheduler failure.

pb now exposes workflow artifacts directly as JSON objects instead of hiding their type behind
`allOf`. For compatibility, it also decodes one complete JSON-stringified typed artifact exactly
once, preserves the original call, validates the decoded artifact normally, and reports the
normalization. It does not repair malformed JSON or invent missing structure.

The resumed probe (`1784490666930-34937-0`) reached that stricter boundary. Its first completion
took 689,587 ms for 4,354 prompt tokens and 515 generated tokens (4.58 Wh at 23.9 W). The decoded
plan was itself malformed because opening braces were missing for later steps. The cached
correction took 151,877 ms for 4,976 prompt tokens and 516 generated tokens (0.84 Wh at 19.8 W),
but inserted a quote before the missing brace instead of fixing the structure. pb correctly
rejected both without guessing.

The fresh typed-schema run (`1784491831209-39233-0`) accepted both its plan and plan review on the
first submission. That isolated the schema fix from the model's earlier malformed compatibility
output. Implementation also proved that one call per prompt is not a constraint: the model emitted
four independent TODO additions in one native batch, then paired a TODO update with each successful
`write_file` for `index.html` and `styles.css`.

Its next 2,048-token completion contained two complete TODO calls followed by a truncated
`write_file` for `game.js`. The old executor ran the complete prefix even though the intended file
mutation was incomplete. pb now recognizes Qwen's unfinished `<function=...>` tail, rejects the
entire max-token batch before executing any member, and enters the existing bounded truncation
recovery. No new scheduling path was introduced.

Process-level resume then exposed a second handoff defect: volatile TODO state was gone, so the
model retried `write_file` for the already-created `index.html`. Implementation and repair prompts
now list every planned path as missing, unchanged, created in this task, modified in this task, or
deleted in this task. The resumed workflow stopped truthfully at its step limit
(`1784500308030-56072-0`): ten model invocations and 5,460 generated tokens had produced only
`index.html` and `styles.css`; no check, review, or commit evidence existed.

A smaller continuation task over that adopted scaffold found that the non-interactive harness was
still advertising `ask_user` even though its event sink could not return an answer. Two repeated
question turns in `1784501585718-56961-0` were pb capability defects. Tool exposure now follows the
sink: the CLI harness omits `ask_user`, while an interactive web session retains it.

After that fix, continuation run `1784503415217-61220-0` completed planning with a valid two-file
plan, but only after three turns. Its planning invocations took 576,658 ms, 617,487 ms, and
1,140,884 ms. Plan review spent 695,893 ms rediscovering the same scaffold, then spent 714,606 ms
calling unexposed `write_file`. A cache-reused correction took 35,746 ms but repeated the exact
invalid call; the no-progress guard detected it and the experiment was stopped. This is classified
as a model stage-control failure amplified by expensive serial prefill. It did not mutate the
workspace.

The final scratch therefore contains only the incomplete HTML/CSS scaffold. `game.js` and
`game-logic.test.mjs` do not exist, the trusted logic check cannot run, and there is no
browser-loadable artifact to verify. Browser inspection was deliberately not used as a substitute
for the failed harness acceptance. The preserved typed outcome is `incomplete` / `step_limit`, not
a successful game evaluation.

## Ranked follow-up

1. Batch native prompt prefill so a 4–5k-token cold planning prompt does not require roughly eleven
   minutes and a 6–7k-token turn does not require roughly twelve to twenty minutes. Prefix reuse
   made the 80-token invalid-call correction cheap, but it does not address new processes or
   stage-specific prefixes.
2. Persist a compact, authoritative evidence digest across stages and process resumes. Planning
   and plan review repeatedly reread the same two files; accepted evidence should let review begin
   at its decision boundary without replaying large file contents.
3. Keep workflow terminal schemas direct and compact. Prefer one plan step covering several
   independent file creations when that preserves requirement and path coverage; fewer repeated
   nested objects reduce local-model syntax failure.
4. Add constrained/grammar-guided native function names and arguments or incremental schema
   validation during generation. This should prevent a plan-review turn from selecting an
   unexposed mutation tool while the executor continues to fail closed.
5. Spend model turns on high-value batches of independent reads and checks. One tool call per
   prompt is not a requirement and would make the current inference rate unnecessarily expensive;
   batching TODO bookkeeping without the intended mutation has little value.
6. Keep deterministic plan, check, review, and commit gates as the substitute for unsupported
   hidden thinking. For models trained with a real reasoning channel, disabling it can hurt
   ambiguous diagnosis; for Qwen3-Coder-Next the mode is unsupported, so enabling it would only
   create a misleading contract.
7. Decompose artifact work into compact, complete edits sized below the output cap, and add faster
   native decode kernels after batched prefill. Residency removes expert I/O but does not by itself
   make a 7k-token agent turn interactive.
