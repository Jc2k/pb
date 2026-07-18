# DeepSeek V4 Flash agent-harness field run

Captured: 2026-07-18

Backend candidate: `79555aae` (`perf: close DeepSeek Flash prefill gap`)

This field run exercised the shipped strict workflow with the local DeepSeek V4 Flash graph after
the backend performance work. It is an agent-control stimulus, not an artifact-quality benchmark.
The preserved journal and events were reviewed independently, and the contract check and Git state
were verified outside the agent run.

## Locked stimulus

| Field | Value |
| --- | --- |
| Model | `antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` |
| Loaded graph | DeepSeek V4 Flash, 43 layers, 256 experts/layer, K=6, hidden size 4,096 |
| Expert store | scheduler-owned packed experts, approximately 72.6 GiB |
| Sampling | temperature 0, top-k 1, 2,048 maximum generated tokens per turn |
| Task | build a dependency-free Typing of the Dead-inspired browser game |
| Contract | `index.html`, `game.js`, and `game-logic.test.mjs`; required `deno test`; fresh review; semantic commit; clean worktree |
| Scratch workspace | `/private/tmp/pb-dsv4-totd-79555aae/workspace` |
| Run ID | `1784407861150-73241-0` |

## Outcome

The run exited nonzero and truthfully reported `verified_completed=false`. Planning and independent
plan review produced accepted, fingerprint-bound artifacts. During implementation the model created
only `index.html`, then repeatedly attempted to emit a verbose complete `game.js` in a single native
`write_file` call. Each attempt reached the 2,048-token cap before the DeepSeek DSML call closed.
After the first rejection the model incorrectly read the intended path as though a partial file
existed, received a structured not-found failure, and returned to the same oversized write.

pb rejected every malformed action, did not execute a partial write, stopped at the parse threshold,
and did not advance to checks, review, commit, or delivery. Independent verification found exactly
one uncommitted allowed file, only the harness baseline commit, and no `game.js` or test file.
`deno test game-logic.test.mjs` failed because the test module was absent.

| Metric | Recorded value |
| --- | ---: |
| Wall time | 2,100,333 ms |
| Model invocations | 12 |
| Prompt tokens | 87,767 |
| Generated tokens | 14,686 |
| Tool calls | 9 |
| Tool runtime | 22 ms |
| Estimated task energy | 26.63 Wh |
| Final contract status | unspecified / not verified |

## Ranked observations

1. **P2 — pb defect — equivalent capped native actions lost continuity across an intervening
   parsed failure.** A parsed `read_file` reset the parse signature history before its failed outcome
   was known. The unchanged run therefore consumed one additional 2,048-token generation before the
   consecutive parse limit stopped it.
2. **P2 — model limitation — the model did not shorten or split its implementation.** It repeated
   the same verbose file body after explicit action-only and parse corrections, and incorrectly
   inferred that a truncated tool call had created a partial file.
3. **Positive evidence — strict workflow and mutation containment held.** Accepted structured plan
   artifacts advanced stages; prose and malformed DSML did not. No partial tool payload executed,
   no check/review/commit evidence was fabricated, and the incomplete run could not report verified
   completion.
4. **Positive evidence — the DeepSeek Flash backend sustained a real multi-turn native-tool
   session.** The loaded graph completed 12 model invocations with session-prefix reuse and preserved
   a complete auditable event stream; the terminal failure was agent control/model behavior, not a
   backend crash or semantic corruption.

## Shipped follow-up

Repeated max-token native actions are now normalized by attempted tool name plus current workspace
and evidence fingerprints. A parsed action no longer erases that history merely by parsing; a real
workspace or evidence transition changes the signature. Capped file-write feedback also states that
no partial file exists and requires materially shorter complete content rather than another
oversized payload or a read of the nonexistent path.

The deterministic reproduction stops after five invocations, preserving one unused completion;
the recorded field run used six implementation invocations around the same sequence. The expensive
real-model task was not repeated because the regression is fully reproduced by the scripted
control fixture and the original run consumed 26.63 Wh. No model-family override, runtime fallback,
tool expansion, or acceptance-gate relaxation was introduced.
