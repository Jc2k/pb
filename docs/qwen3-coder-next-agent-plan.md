# Qwen3-Coder-Next Native Agent Plan

This record turns the Qwen3-Coder-Next browser-game field failure into a bounded implementation and
evaluation plan. The built-in FlashMoe runner is the primary target. llama.cpp remains supported
for explicit GGUF and other llama selections, but it is not the correctness oracle or fallback for
a requested native graph.

The baseline failure mixed two independent problems. The short model name was misclassified as
Qwen3.5, so native cache setup failed and the run silently used llama.cpp. The model then spent
large capped turns on prose or fabricated transcript continuations, made whole-file repairs, and
exhausted its budget before producing a browser-loadable artifact. Harness acceptance correctly
rejected the result; the provenance and efficiency failures are pb defects, while the generated
artifact mistakes are model evidence to address through clearer boundaries and cheaper valid
actions.

## Decisions

- Qwen3-Coder-Next is a typed native family with its own 48-layer, 512-expert, top-10 hybrid graph.
  It is not a Qwen3.5 alias.
- The default/plain alias selects the indexed MLX affine-Q4 source. Explicit GGUF paths select
  llama.cpp. Once selection is made, load failure is terminal rather than a backend change.
- Resident experts are a binary graph decision. If the complete corpus plus reserve fits the
  sampled Metal budget, every fixed whole-expert slot remains mapped for graph lifetime. Otherwise
  the existing scheduler streams selected experts with parallel `pread` and trusts the OS page
  cache. There is no partial cache or new scheduler.
- The model does not implement a thinking mode. FlashMoe clamps that mode off for prompt
  measurement and generation, and telemetry reports the effective policy.
- One tool call per prompt is not required. The native protocol and JSON compatibility protocol can
  return a batch of independent calls. pb validates every member and runs parallel-safe work
  concurrently; dependent actions and authority-changing transitions remain separate turns.

Disabling a real thinking mode can reduce performance on tasks where a model was trained to use a
private reasoning channel, especially ambiguous design and diagnosis. It can also make premature
actions more likely if the prompt asks for mutation before evidence. Those disadvantages do not
apply as a reason to enable an unsupported mode on Qwen3-Coder-Next. The compensating controls are
deterministic planning/review stages, authoritative tool results, small exact edits, batchable
discovery, action-first turns only when the plan makes the next mutation unambiguous, and bounded
repair rather than unmetered hidden reasoning.

## Milestones and gates

1. **Native identity and graph.** Resolve the alias to the native Q4 source, add a distinct family,
   preserve checkpoint K=10 and `full_attention_interval=4`, validate the shared expert and tensor
   geometry, and cover capability resolution with exact-shape fixtures.
2. **Published-checkpoint proof.** Build the cache from all nine indexed shards, load the Metal
   graph, record the memory decision, and complete deterministic raw and structured no-thinking
   inference. Import, tensor, or kernel gaps are fixed at their typed owner; no alternate backend is
   accepted as evidence.
3. **Agent-control alignment.** Enforce model-level non-thinking, keep native multi-call batches,
   make native load failure terminal, and ensure prompt/invocation events report the effective
   backend and thinking policy.
4. **Browser-task evaluation.** Start with scripted fixtures for batched reads, non-thinking
   truncation, invalid oversized edits, and browser acceptance. Then run one bounded native
   Qwen3-Coder-Next task to build a browser-based Typing-of-the-Dead clone. Preserve scratch state
   and events, serve the artifact, and verify load, interaction, visible failure state, and console
   cleanliness in a real browser.
5. **Qualification and delivery.** Classify each failure as pb, model, experiment, or external;
   implement only clear in-scope pb defects; update curated architecture/user docs and this record;
   run focused tests, the scripted corpus, web/docs/all-target checks, release build, and the narrow
   FlashMoe smoke; then make one scoped semantic commit without absorbing unrelated worktree edits.

## Implemented checkpoint

Milestones 1–3 are implemented. Published-checkpoint qualification covers both the resident graph
and a forced 32 GiB streamed graph using the existing positioned-read scheduler. The field
experiment implemented four clear pb fixes: direct terminal artifact schemas with bounded
one-level compatibility decoding, atomic rejection of a truncated native call batch, durable
planned-path state in implementation prompts, and removal of `ask_user` from non-interactive
harness runs.

Milestone 4 ended with the truthful incomplete outcome documented in the benchmark record. The
model created two of four files, then a decomposed continuation reached an accepted plan but failed
plan-review stage control by repeating an unexposed mutation call. Because no complete game or
logic test existed, browser verification was not applicable. The remaining work is performance and
model-control improvement, not another expert scheduler: batched prompt prefill, compact persisted
stage evidence, constrained function selection/arguments, and output-cap-aware task decomposition
are the highest-priority path to a successful rerun.

## Acceptance

Completion requires evidence that the selected runtime is FlashMoe, the loaded family is
Qwen3-Next with K=10, the effective thinking policy is off, and no llama.cpp setup occurs. The
native arithmetic smoke must produce a sensible answer. The agent experiment must either deliver a
browser-loadable, interactively verified game under its contract or stop with a truthful typed
outcome whose remaining failure is classified and preserved. A visually plausible file that fails
module load, omits required interaction, or bypasses harness acceptance is not success.
