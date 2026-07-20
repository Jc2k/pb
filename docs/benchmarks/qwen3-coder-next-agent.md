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

## Layer-major qualification rerun — 2026-07-20

After the true layer-major graph passed its separate parity, memory, and performance gates, run
`1784569541795-68688-0` repeated the locked four-file contract in a fresh scratch using the resident
built-in runner. Planning completed on its first 435-token submission in 68,017 ms; plan review
completed on its first 371-token submission in 113,230 ms. Their native prefill/decode splits were
17,045/50,712 ms and 66,493/46,709 ms respectively. This confirms that semantic terminal stopping
and layer-major prefill removed the former 4,096-token terminal overrun and roughly eleven-minute
cold planning pass.

Implementation created `index.html` in 498 tokens and `styles.css` in 2,306 tokens. The next
2,425-token `game.js` call reached the dynamic 7,808-character content bound and was structurally
closed as a valid tool call, but the file ended at `document.addEventListener('keydown',` and failed
`deno check`. A later 4,096-token turn took 1,054,657 ms yet decoded to only `<tool_call>`; another
4,096-token repair took 1,255,109 ms and decoded to 129 visible characters. The model then made a
byte-identical `replace_file`, which the old progress accounting incorrectly marked useful. On the
last outer step, a 4,096-token named write activated the compact same-step retry; that retry used
2,283 more tokens and a smaller prompt but still selected `write_file` for the existing `game.js`,
so the executor correctly rejected it.

The run ended after 6,683,787 ms with 11 model invocations, 113,695 prompt tokens, 22,939 generated
tokens, and an estimated 57.3 Wh. Its typed outcome was `step_limit`; the named Deno check, code
review, managed commit, and browser inspection were never reached. The scratch preserves only
three untracked files, `game-logic.test.mjs` is absent, and the required check fails at module
resolution. Browser inspection was again deliberately not used as a substitute for acceptance.

This rerun exposed four clear pb defects, now covered by deterministic regressions:

- constrained non-EOS output must grow monotonically in decoded length, not merely differ from the
  previous decode;
- an open `write_file` or `replace_file` content string at `maxLength` stops as a truncated named
  call instead of being force-closed and executed;
- compact mutation recovery applies its half-size allowance to the retry schema as well as the
  correction text; and
- byte-identical edit results receive no diff, evidence invalidation, or useful-progress credit.

The harness journal also now classifies dirty state preserved by an incomplete delivery as model-
limitation evidence rather than an experiment error. It remains an experiment error after a
claimed ready or verified result and still fails a clean-workspace contract. An explicit cached
Qwen3-Coder-Next GGUF control independently returned `4` through llama.cpp after these native-runner
changes; it was not used as fallback.

## Ranked follow-up

1. Validate the new monotonic-progress, payload-limit, and compact-schema behavior with a targeted
   native large-mutation probe before paying for another full workflow. The expected result is an
   early named truncation and a materially shorter retry, with no cut-off file or no-op progress.
2. Profile native decode across the shared Qwen data flow. The rerun sustained only about 3–8
   tokens/s and spent 1,018–1,351 seconds decoding each capped action; promote a common kernel only
   with exact greedy parity and the existing 1.5x gate.
3. Shape implementation actions around small complete scaffolds and exact later edits. The
   controller should expose the enforced payload allowance prominently and prefer a bounded tail
   patch after an authoritative read rather than a whole-file replacement.
4. Persist a compact authoritative evidence bundle and preserve stable prompt/tool prefixes across
   recovery. The compact retry omitted the rejected payload but still spent 302,949 ms prefilling
   its changed 13,518-token prompt.
5. Spend model turns on high-value batches of independent reads and checks. One tool call per
   prompt is not a requirement and would make the current inference rate unnecessarily expensive;
   batching TODO bookkeeping without the intended mutation has little value.
6. Keep deterministic plan, check, review, and commit gates as the substitute for unsupported
   hidden thinking. For models trained with a real reasoning channel, disabling it can hurt
   ambiguous diagnosis; for Qwen3-Coder-Next the mode is unsupported, so enabling it would only
   create a misleading contract.
7. Rerun the locked workflow only after the targeted mutation probe and decode gate. Browser
   inspection remains last: it cannot substitute for all four files, the named Deno check, fresh
   code review, semantic commit, and clean worktree.
