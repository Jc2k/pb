# Qwen3-Coder-Next Agent Performance Follow-on

Status: **Integrated qualification incomplete; recovery hardening active.** Durable stage evidence, compact terminal schemas, Qwen
native tool constraints, output-aware mutation bounds, strict-stage TODO removal,
dependency-aware batch rejection, native telemetry, and true layer-major Qwen3-Coder-Next prefill
are implemented. The prefill graph passed exact state and resource/performance promotion gates; the
native runner now also stops on a semantically complete terminal submission and gives an oversized
file mutation one schema-enforced compact same-step retry. The preserved layer-major harness rerun
reached implementation but remained incomplete; browser qualification therefore remains open.
The first focused mutation probe confirmed complete bounded scaffolding but exposed false external
verification when a strict workflow reached `Ready`. Explicit contract projection and gating now
pass an executable native qualification: the undersized artifact failed its trusted check, never
reached review or commit, and ended unsatisfied and unverified. The locked integrated rerun then
accepted plan/review but again stopped with only HTML/CSS; exact-target, right-sized compact
recovery is now the focused gate before another full browser run.
Decode has been profiled; no change was promoted because the measured sampling path cannot meet the
1.5x gate.

Evaluation: [Qwen3-Coder-Next native agent](benchmarks/qwen3-coder-next-agent.md)

This plan covers the work needed for the largest resident Qwen3-Coder-Next checkpoint on a 64 GiB
Apple Silicon Mac to complete the bounded Typing Defense agent task. The built-in FlashMoe runner is
the implementation target. Explicit GGUF requests must continue to work through llama.cpp, but
llama.cpp is only a compatibility smoke and regression control for this work; it is not a fallback,
performance target, or correctness oracle for a native request.

The preserved browser run did not expose an unresolved safety or completion-gate failure. pb rejected
malformed and unavailable actions, did not execute a truncated native batch, did not claim missing
checks or review, and stopped with `incomplete / step_limit`. The remaining findings are therefore
primarily P2 efficiency and diagnostic-quality work plus model-control improvements. A new expert
scheduler, an application expert cache, fake reasoning, or weaker workflow gates would not address
the measured bottlenecks. The later focused mutation probe did expose one P1 completion-reporting
defect: strict `Ready` was projected as verified even when explicit contract status remained
unspecified. That defect is now in the active gate below.

## Outcome

The follow-on is complete when the same resident checkpoint can finish the original four-file task
through planning, plan review, implementation, checks, code review, managed commit, and browser
inspection within the existing bounded workflow. The implementation must also retain the forced
32 GiB streamed graph, exact native provenance, atomic tool batches, truthful terminal outcomes,
and explicit llama.cpp GGUF control.

The intended order is:

1. make cold native prompt prefill practical;
2. stop stages and resumed processes from paying again for unchanged repository evidence;
3. constrain native tool selection and JSON arguments before execution;
4. reduce terminal-schema and edit payload size while keeping internal validation exact;
5. improve high-value batching and then optimize decode using the shared Qwen data flow; and
6. rerun the locked agent and browser contract only after deterministic gates pass.

## Baseline and classification

| Observation | Classification | Priority | Disposition |
| --- | --- | --- | --- |
| A fresh 4,354-token planning prompt took about 690 seconds | pb native-runner efficiency | P2 | Implement Qwen layer-major batched prefill |
| New 6–7k stage prefixes took roughly 12–20 minutes | pb native-runner efficiency | P2 | Batch fresh suffixes; retain exact prefix reuse |
| Planning and plan review reread the same unchanged scaffold | pb context/persistence efficiency | P2 | Persist a bounded authoritative evidence bundle |
| Plan review selected unexposed `write_file` | model stage-control limitation amplified by pb | P2 | Constrain native function names and schemas |
| Large `game.js` output hit the generation cap | model/output-shaping limitation | P2 | Emit bounded complete edits and expose payload budgets |
| A 4,096-token constrained call decoded to only `<tool_call>` | pb native-constraint progress defect | P1 | Require monotonic decoded growth and stop bounded mutation strings as truncated named calls |
| An identical `replace_file` received useful mutation credit | pb tool/progress defect | P1 | Reject byte-identical replacements before diff or evidence mutation |
| Compact recovery requested half size only in prose | pb recovery defect | P2 | Apply the half-size limit to the retry schema |
| TODO-only batches consumed output without advancing the artifact | prompt/tool-surface inefficiency | P3 | Remove redundant bookkeeping from strict stages and measure batch value |
| Qwen3-Coder-Next has no supported thinking mode | model contract | Accepted | Keep thinking off and retain deterministic workflow gates |
| The final artifact was incomplete | model limitation, truthfully contained | Accepted fixture | Rerun only after the preceding deterministic improvements |
| Strict `Ready` projected as verified with `contract_status=unspecified` | pb completion-reporting defect | P1 | Fixed and natively qualified with an explicit failing content contract |
| Compact recovery drifted from capped `game.js` to existing `styles.css` and retained the 4,096-token ceiling | pb recovery control/efficiency defect | P2 | Bind the original target and derive a smaller retry token ceiling |

The direct workflow schema, one-level string compatibility decoder, atomic rejection of truncated
native batches, durable planned-path status, and sink-aware `ask_user` exposure are already shipped
and remain regression requirements.

The preserved evidence that drives implementation is deliberately small:

| Run | Relevant evidence | Deterministic proof before another model run |
| --- | --- | --- |
| `1784491831209-39233-0` | accepted plan/review, useful multi-call writes, then a capped `game.js` call | atomic truncated-batch and bounded-payload fixtures |
| `1784500308030-56072-0` | process resume recovered path state but ended after ten invocations with only two files | checkpoint/evidence round trip and output work-unit fixtures |
| `1784503415217-61220-0` | three slow planning turns, a slow plan-review reread, then repeated unexposed `write_file` | cold-prefill benchmark, carried-evidence fixture, and constrained-logit fixture |
| `1784555046072-44244-0` | terminal semantic stop reduced planning to 435 generated tokens; plan review accepted on its first 367-token submission | terminal-body completion and parser-envelope fixtures |
| `1784559198505-50026-0` | plan/review accepted, then capped CSS and JavaScript mutations consumed long retries and left only HTML/CSS | compact same-step mutation retry and bounded-string structural-close fixtures |
| `1784569541795-68688-0` | first-turn plan/review and three files, then cut-off JavaScript, collapsed constrained output, a no-op replacement, and step-limit containment | monotonic decode, mutation-limit stop, enforced compact schema, no-op mutation, and journal-classification fixtures |
| `1784586458472-40087-0` | 256-token plan review could not fit and stopped after three bounded parse failures | classify as experiment error; use the smallest cap that can express the terminal schema |
| `1784586718611-40960-0` | bounded 290-byte scaffold followed by false implementation/review claims and `unspecified` yet verified completion | executable acceptance check plus strict contract projection/gating fixtures |
| `1784588468585-81408-0` | trusted length check rejected a 263-character scaffold three times; bounded repair ended with no review or task commit | `unsatisfied`, unverified, non-zero terminal result plus independent byte audit |
| `1784589577078-95753-0` | required-check planning repeated because the compact example omitted `acceptance[].check_ids` | classify as experiment error; expose the exact field in examples and corrections |
| `1784590137641-98220-0` | first-turn plan/review and two files, then 53.7 Wh of capped/repeated CSS-directed actions; logic files remained absent | exact-target compact schema, proportional retry cap, and preserved unsatisfied terminal state |

The focused contract-backed acceptance gate passed, and the integrated run completed truthfully but
did not produce a runnable game. The next justified model spend is a focused native compact-recovery
probe. Another Phase 5 run waits for that smaller gate.

## Architectural invariants

- The native resident/streamed decision remains binary and load-resolved. If the complete expert
  corpus plus reserve fits, all experts remain resident. Otherwise the current scheduler owns
  parallel positioned reads and the OS page cache. Batched prefill uses either resolved graph; it
  does not add a partial cache or alternate scheduler.
- Prompt geometry may select a scalar or batched command, but an execution error may not switch the
  graph, expert policy, quantization, backend, or model family.
- Qwen3-Coder-Next remains non-thinking. `supports_thinking=false` is applied before prompt
  measurement and generation and is recorded in agent telemetry.
- Tool constraints narrow model output; they never grant a tool, repair an invalid call, or weaken
  executor-side capability and JSON-schema validation.
- Evidence reuse carries repository bytes and provenance, not another model's conclusion. A
  reviewer remains a fresh model stage and can cite only current, actually exposed evidence.
- A max-token native batch remains atomic. No complete prefix call executes when any following call
  is truncated.
- Internal `PlanArtifact`, review, implementation, evidence, and commit validation remain the
  authority boundary even if the model-facing wire form becomes smaller.
- One tool call per prompt is not a requirement. Independent calls may be batched; dependent
  actions, correction turns, and workflow transitions remain ordered.
- Native performance work must preserve the common supported Qwen MoE data flow. Do not land an
  isolated checkpoint-only or Q4-only fast path without a typed plan for every affected Qwen graph.

## Phase 0 — Lock measurements and deterministic fixtures

Before changing kernels or prompts, make the evaluation repeatable without another hour-long agent
run.

### Implementation

- Add a checked-in native prefill benchmark description for three rendered prompt geometries:
  approximately 512 tokens, the 4,354-token planning frontier, and a 6–7k plan-review frontier.
  Store hashes and generation settings rather than large duplicated prompt bodies where the
  harness can render the prompt deterministically.
- Extend generation events and the harness journal with fresh-prefill tokens, cached tokens,
  prefill wall time and tokens/second, decode wall time and tokens/second, graph expert strategy,
  prefill command kind, tool-constraint mode, serialized tool-schema tokens, and maximum serialized
  action size.
- Add scripted agent fixtures for:
  - a plan-review model preferring an unexposed mutation tool;
  - a valid batch of independent reads;
  - a TODO-only batch;
  - a write that cannot fit its declared per-turn payload allowance; and
  - process resume with current versus stale evidence.
- Preserve the original run IDs and contract in the benchmark record as the before measurement.

Likely owners are `src/inference/flashmoe/runtime.rs`, `src/events.rs`, `src/harness_eval.rs`,
`fixtures/harness-control-fixtures.json`, and `docs/harness.md`.

### Gate

The scripted corpus must deterministically reproduce each control decision without loading the
model. A single native benchmark command must report prefill and decode separately and identify
`Qwen3NextMoe`, K=10, resident or streamed experts, and effective thinking off.

## Phase 1 — Qwen layer-major batched prompt prefill

**Implemented and promoted on 2026-07-20.** The exact affine-Q4 graph passed resident and forced
streamed zero/restored-prefix parity, forced chunk-boundary parity, the 5% memory gate, and the
5x/120-second frontier gates. The locked 4,354-token geometry completed in 59.685 seconds at 72.95
token/s, 11.55x faster than the preserved scalar baseline with 3.00% additional allocation. See
[the qualification record](benchmarks/qwen3-coder-next-prefill-qualification.md). The design below
is retained as the implementation contract.

This was the highest-value work. The prior Qwen path sent every fresh prompt token through the
single-token graph. Exact in-memory prefix reuse made short corrections cheap, but a new process
or stage-local prefix still paid the scalar cost.

### Graph design

Add a load-resolved Qwen batch-prefill capability alongside the existing scalar token command. A
prompt suffix at or above one useful matrix tile is divided into bounded chunks chosen from model
geometry and the sampled Metal working-set budget at graph preparation. Smaller suffixes retain
the scalar correctness path.

Each chunk executes layer-major:

- gather embeddings and run resident dense projections over all rows;
- on full-attention layers, calculate causal attention over the restored prefix plus earlier rows
  in the chunk and append the same typed KV records used by decode;
- on linear-attention layers, batch the input projections but apply convolution and recurrent-state
  updates in token order so the final state is identical to scalar recurrence;
- calculate every row's exact softmax top-10 route and preserve row-local expert order and weights;
- for resident experts, execute the shared and routed affine-Q4 projections over all routed rows;
- for streamed experts, form the sorted union of selected experts for the chunk and ask the
  existing scheduler to issue its parallel positioned reads into request scratch; and
- return the final row hidden state plus complete KV/recurrent state for ordinary sampling and
  session capture.

The streamed implementation may add a batch-shaped scheduler command and request scratch, but the
existing scheduler remains the sole owner of expert I/O. The resident implementation must issue no
expert reads. Chunk buffers are released at request end and never become an application cache.

Likely owners are `src/inference/flashmoe/capabilities.rs`,
`src/inference/flashmoe/runtime.rs`, `src/inference/flashmoe/metal.rs`,
`src/inference/flashmoe/scheduler.rs`, `src/inference/flashmoe/state.rs`, and the existing Metal
kernel source embedded by the FlashMoe build. Keep
`docs/flashmoe-architecture-parity-plan.md` aligned as commands land.

### Correctness gates

- Scalar and batched paths must produce matching layer probes, final hidden state within the
  declared numeric tolerance, identical greedy next tokens, identical KV records, and identical
  linear-attention checkpoint state for zero-prefix and restored-prefix suffixes.
- Test chunk boundaries immediately below, at, and above the batch threshold, plus a suffix that
  requires several chunks.
- Run the same fixtures against complete resident experts and the forced 32 GiB streamed graph.
- The official MLX-LM raw `a` parity result, structured `2+2` smoke, session A/B restore, and native
  truncated-batch regression must remain unchanged.
- Failure to allocate or encode the declared batch command is terminal; it does not retry scalar
  after partial execution.

### Performance gate

On the qualification Mac, promote the command only when the 4–5k cold planning prompt is at least
5x faster than the preserved scalar baseline and completes within 120 seconds, with no more than a
5% increase over the declared request/session reserve. Record a 10x or 70-second result as the
optimization target, not as a reason to weaken correctness. Also record the 6–7k frontier and both
resident and streamed memory behavior.

## Phase 2 — Durable authoritative stage evidence

Prompt-prefix reuse is process-local and stage-local by design. It cannot stop a fresh plan
reviewer or resumed process from rereading unchanged files. Add a typed evidence bundle to the
workflow checkpoint rather than treating volatile chat/TODO state as durable context.

### Data model

Introduce a versioned `StageEvidenceBundle` owned by the workflow run. Each repository entry
contains:

- normalized repository-relative path and evidence kind;
- file SHA-256, byte/line count, workspace content fingerprint, and source stage;
- exact observed line ranges or a complete-file marker;
- bounded exact content for those ranges plus an omitted-byte/line count; and
- the originating tool name, normalized argument digest, success state, and capture time/order.

The harness, not the model, builds and hashes the bundle from successful read results. Small files
that fit the evidence budget may be carried completely. Large files retain only deterministic
observed ranges and bounded excerpts; unobserved ranges never become review evidence. The existing
prompt receipt representation can be reused, but its current rule remains important: a receipt by
itself grants no new authority.

### Stage and resume behavior

- Persist the bundle in `WorkflowCheckpoint` with serde defaults for old checkpoints.
- At a read-only stage transition, verify the workspace fingerprint and each referenced path hash.
  Inject current entries into the fresh stage anchor and seed that stage's observed-read ledger only
  for the exact complete files/ranges present in the bundle.
- If a relevant path changed, discard only its stale entry and require a fresh read. If Git control
  or the workspace graph changed unexpectedly, retain the existing fail-closed reconciliation.
- Keep planner prose, TODO state, and model conclusions out of the bundle.
- Bound total bytes, entries, and ranges. Evict deterministic oldest/non-plan-path evidence first
  and report omissions in the prompt and journal.

Likely owners are a new `src/workflow/evidence.rs`, plus `src/workflow/engine.rs`,
`src/workflow/persistence.rs`, `src/agent_context.rs`, and the stage/gate assembly in
`src/agent_core.rs`.

### Gate

A deterministic planning-to-review fixture must prove that an unchanged complete small file is not
reread, the reviewer can cite it, a changed file is invalidated, a partially carried large file
does not satisfy a full-file review requirement, and the same results survive checkpoint
serialization and process resume. Prompt telemetry must show the bytes/tokens avoided.

## Phase 3 — Constrained native tools and compact terminal wire schemas

Executor validation correctly rejects an unavailable tool, but doing so after a long decode wastes
the turn. Constrain the built-in runner at generation time while retaining executor validation as
the final authority.

### Constraint engine

**Shipped.** The built-in Qwen runner compiles exposed schemas, validates a terminal submission
name at preflight, stops when that terminal JSON body becomes complete, and closes a missing Qwen
tool envelope only for parsing. Ordinary independent calls remain batchable. Visible-progress,
bounded-string closing, and 32-token repetition guards prevent constrained decoding from spending
the remainder of a cap on a structurally finished or repeated action. After the locked rerun found
that unequal decoded prefixes could still collapse, visible progress was tightened to monotonic
decoded length. File-content strings now stop as truncated named mutations at their limit instead
of being force-closed into cut-off repository files.

- Add a request-level native constraint mode: `auto`, `tools_allowed`, or `tool_required`. Strict
  workflow stages pass the exact exposed tool set; terminal-only turns require the one terminal
  tool already selected by the deterministic controller.
- Compile Qwen's native tool-call envelope plus the current JSON Schemas into a byte-level DFA.
  The DFA permits ordinary final text only when the stage contract permits it. Once a native tool
  envelope begins, function names, JSON syntax, property names, enums, required properties, array
  structure, and scalar types are constrained.
- Match tokenizer token byte sequences against DFA transitions. At free-string positions, filter
  the normal top candidates. At small literal/structural frontiers, project the exact allowed
  LM-head rows so a valid token cannot be hidden outside top-k. If the schema cannot be compiled,
  fail the structured request before generation rather than silently running unconstrained.
- Allow several independently valid native calls in one response. Continue to reject the entire
  response when the final call is incomplete at the output cap.
- Emit the constraint mode, schema digest, rejected-candidate count, and terminal state in native
  diagnostics without logging private argument content.

Initial support should cover the JSON-schema subset used by built-in tools. Unsupported schema
features must be rejected at tool-registration/request preparation, not guessed during sampling.
llama.cpp retains its existing grammar/tool behavior and does not need to share this decoder.

Likely owners are `src/inference/flashmoe/types.rs`, `src/inference/flashmoe/text.rs`,
`src/inference/flashmoe/runtime.rs`, the resident LM-head projection helpers in
`src/inference/flashmoe/weights.rs`, and request construction in `src/agent_core.rs`.

### Compact wire form

In the same phase, reduce output structure without weakening the durable artifact:

- flatten the redundant `{id, plan: {...}}` / `{id, review: {...}}` envelope on the native wire and
  deterministically reconstruct the existing `ArtifactEnvelope` internally;
- stop requiring fields that already have safe serde defaults, such as empty risks, assumptions,
  open questions, resolved challenge IDs, component IDs, or check IDs;
- keep stable requirement/step IDs where cross-reference validation needs them;
- continue allowing one step to cover several independent paths and make that the preferred form
  when requirement and acceptance coverage stays unambiguous; and
- retain bounded parsing compatibility for the currently shipped nested form during checkpoint and
  client migration.

Measure schema tokens and serialized successful-call tokens before and after. Internal artifact
validation, digesting, stage transitions, and journal records remain unchanged.

### Gate

Scripted logits that prefer `write_file` during plan review must produce the allowed
`submit_plan_review` call or a typed constraint failure without mutation. Fixtures must cover wrong
property names, enums, missing required fields, escaped and Unicode string content, multi-call
batches, max-token truncation, and parser/executor revalidation. The compact plan fixture must
retain exactly the same normalized artifact digest and coverage decisions as its legacy form.

## Phase 4 — Output-cap-aware edits, useful batches, and decode

### Bounded complete edits

**Shipped, with field qualification active.** Dynamic mutation bounds are enforced during
generation and again at execution. A capped `write_file` or `replace_file` gets one compact retry
inside the same stage step, narrowed to that tool, with half the original schema allowance, and
without the rejected payload in context; all invocation and generated-token budgets remain charged.
Byte-identical replacements, edits, and patches receive no mutation or progress credit.

- Derive a conservative serialized-argument allowance from the remaining generated-token budget
  after reserving native envelope and JSON-closing overhead. Add `maxLength` to dynamic mutation
  schemas and put the exact allowance in the implementation anchor and generation event.
- For a missing file that is unlikely to fit, instruct the model to create the smallest complete,
  loadable scaffold and then add one accepted-plan feature per bounded `apply_patch`/`edit_file`
  turn. Never ask it to stream a partial JSON string or assume a capped call was executed.
- Persist per-path state and completed plan-step facts after every mutation so resume begins at the
  next bounded work unit.
- Keep each tool call atomic. Do not introduce partial repository files. If bounded edits still
  cannot complete the locked fixture, design a separate task-owned scratch/finalize protocol with
  hashes and atomic publication; do not add it speculatively in the first pass.

### High-value batches

- Remove `todo` from strict workflow stage capabilities; the accepted plan and checkpoint already
  own progress. Retain it for legacy conversational builds if still useful.
- Keep batches for independent repository reads, searches, checks, and separate file creations.
  Reject batches with data dependencies and preserve the current atomic truncated-tail rule.
- Record batch call count, parallel-safe count, useful evidence/mutation count, and bookkeeping-only
  count. A prompt policy change is successful only if it reduces model turns without increasing
  rejected, repeated, or stale actions.

### Decode optimization

After batched prefill and smaller actions are measured, profile Qwen decode by dense projection,
attention/linear-attention, routing, shared expert, routed expert, synchronization, and LM head.
Optimize the largest common Qwen bottleneck with load-resolved kernels used by every compatible
Qwen MoE graph. Do not add a checkpoint-specific Q4 shortcut. Preserve resident zero-read behavior
and streamed scheduler ownership.

The first decode promotion gate is identical greedy output with at least a 1.5x sustained decode
improvement on the locked resident checkpoint and no material regression on an existing supported
Qwen MoE control. A failed speed target is recorded; it is not grounds to change sampling or
precision.

## Phase 5 — Qualification rerun

Phases 1–4 have passed their deterministic and native smoke gates. Run the expensive model with the
same 64 GiB host, resident Qwen3-Coder-Next source, non-thinking policy, four allowed files,
required `deno test game-logic.test.mjs`, fresh four-file review, semantic commit, clean-worktree
requirement, and browser interaction contract from the evaluation record.

Record:

- model/source/cache digest, family, K, resident decision, working-set accounting, and backend;
- per-invocation prompt/cached/prefilled/generated tokens, TTFT, decode rate, constraint mode,
  schema tokens, action size, tool batch value, wall time, and energy;
- every process resume and evidence-bundle reuse/invalidation;
- all accepted/rejected calls and the reason without truncating authoritative event data;
- independent test output, review evidence, commit OID, final Git status; and
- browser load, typing interaction, score/health progression, failure state, restart behavior,
  responsive layout, and console errors.

Success requires all four files, the trusted logic check, fresh review, a managed semantic commit,
a clean task worktree, and browser verification. Otherwise preserve and classify the typed outcome
without manually repairing the generated artifact.

## Commit sequence

Keep the work reviewable and measurable:

1. `test: lock qwen agent performance baselines`
2. `perf: batch qwen prompt prefill`
3. `feat: persist workflow stage evidence`
4. `feat: constrain native tool generation`
5. `refactor: compact workflow terminal schemas`
6. `fix: guide output-cap-aware workflow edits`
7. `perf: improve qwen native decode` only when profiling justifies it
8. `docs: qualify qwen native agent completion`

Each FlashMoe data-flow commit updates the architecture parity record in the same change. Each
user-visible workflow/tool behavior commit updates the curated architecture and user chapters and
runs their required web/docs tests.

## Required regression matrix

Every implementation phase runs focused tests first and then the applicable repository gates:

- scalar/batch and resident/streamed Qwen native parity;
- native structured `2+2` and raw MLX-LM parity;
- session-prefix reuse, process-resume evidence, and atomic truncated batches;
- scripted harness control corpus and the new constrained-tool/output fixtures;
- explicit Qwen3-Coder-Next GGUF through llama.cpp as a compatibility smoke;
- `deno task build:web`, `deno task test:web`, `deno task test:docs`,
  `cargo test --all-targets`, release arm64 build, and the narrow FlashMoe smoke.

llama.cpp regressions block delivery, but native FlashMoe measurements and failures determine the
optimization decisions in this plan.
