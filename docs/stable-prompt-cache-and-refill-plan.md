# Stable prompt roots and FlashMoe refill

Status: **Design record; cache/refill implementation and release-candidate cache qualification are
complete, while managed-workflow performance promotion and the production-ready designation remain
open.**

This plan closes the efficiency gap left after the bounded-workflow call-reduction work. That work
reduced the locked Rust/Python/React sample from 35 model calls and 100,249 fresh-prefill tokens to
14 calls and 47,271 fresh-prefill tokens. It did not make stage prefixes reliably reusable, change
cache-root policy, or optimize FlashMoe work after a prefix restore.

The target is a local-first, privacy-first agent whose useful model calls do not repeatedly prefill
unchanged stage authority. A warm or restartable local cache should skip an exact stable root, and
FlashMoe should process only the genuinely new suffix using its qualified production graph. This
must not weaken stage-specific tool authority, make approximate cache matches, disclose prompt
content, introduce network dependencies, or hide a fallback behind successful completion.

The locked before-and-after evidence is retained in the
[private-workload usability record](benchmarks/private-workload-usability.md). The shipped
FlashMoe graph and parity evidence remain governed by the
[FlashMoe architecture parity plan](flashmoe-architecture-parity-plan.md) and
[Qwen prefill qualification](benchmarks/qwen3-coder-next-prefill-qualification.md).

The 26 July release-candidate result is retained in the
[stable prompt-root qualification](benchmarks/stable-prompt-root-production.md). It closes the
known stable-root, sessionless-root, retention, refill-parity, and 24-case cache-safety gaps. It does
not close the plan: the locked three-language sample missed the wall-time and call-shape gates.

## Outcome

This plan is complete only when all of the following are true:

1. Every managed model invocation reports a privacy-safe identity for its stable prompt root and
   cache namespace, along with exact root, cached, and fresh-suffix token counts.
2. Identical stage authority produces identical rendered root tokens across tasks, repositories,
   sessions, and process restarts. Changed authority produces a different root and cannot reuse the
   old state.
3. A warm memory cache and a valid restartable disk cache both reuse the complete eligible root.
   An unchanged root cannot be reported as `prompt_diverged`; a changed root cannot be reported as
   a hit.
4. FlashMoe restores exact KV and recurrent state once, prefills only the remaining suffix, selects
   the qualified layer-major graph whenever the suffix meets its prepared threshold, and exposes
   any scalar selection or unsupported graph explicitly.
5. The locked three-language sample improves materially beyond the 25 July call-reduction baseline
   without a correctness, review, commit, privacy, memory, or energy regression.
6. The complete 24-case corpus and the cache failure matrix pass before the work is described as
   production-ready.

The four-call Rust and Python path is an efficiency floor for successful bounded work, not the
definition of completion. Focused repair calls remain valid when the model produces a bad change.

## Terminology

The word "root" currently risks mixing unrelated concepts. This work uses four explicit terms:

| Term | Meaning | Authority |
| --- | --- | --- |
| Storage root | The local filesystem directory selected by `storage.cache_dir` or the platform pb cache default | Typed user configuration |
| Model namespace | The versioned backend/model/runtime fingerprint below the storage root | Inference backend |
| Prompt root | The exact rendered token prefix whose complete inference state can be restored | Tokenizer and backend |
| Stage root | The controller-owned stable system instructions plus exact rendered tool authority that form a prompt root | Workflow controller |

A stage root is not a new filesystem tree. The exact rendered token digest remains the final cache
key, and the model namespace remains the compatibility boundary. A typed stage-root descriptor
exists to make stability testable and misses diagnosable; it must never authorize approximate state
reuse.

"Refill" means the inference work after an exact prompt root has been restored. In current code this
is the fresh suffix beginning at `prefill_start`. It includes any required state hydration and suffix
prefill, but excludes decode of newly generated tokens.

## Current shipped baseline

The following behavior predates or accompanied the call-reduction work and must remain intact:

- Dynamic branch, run, contract, and recent-evidence material is rendered after the stable first
  system message. The native tool schema participates in the rendered stable prefix.
- FlashMoe and llama.cpp compare exact rendered tokens before reusing inference state.
- FlashMoe keeps logical session checkpoints and a separate content-addressed prefix checkpoint.
  Disk records are versioned, model-fingerprinted, byte-budgeted, owner-only, and atomically
  published.
- `storage.cache_dir` selects the shared configured inference-cache root. FlashMoe session caching
  has typed enablement and byte-budget settings; no standalone environment control is required.
- FlashMoe restores full-attention or MLA KV state plus typed linear-attention recurrence. It rejects
  stale or structurally incompatible checkpoints.
- Qwen3-Coder-Next has a qualified device-resident layer-major graph. Automatic selection promotes
  prepared fresh suffixes of at least 32 tokens; scalar execution remains the reference and the
  short-suffix path.
- Invocation events now distinguish cached and actually-prefilled tokens and classify disabled,
  cold, divergent, unavailable, unreadable, reset, and unsupported cache outcomes.

That foundation proves safe exact reuse is possible. It does not prove that controller-owned stage
roots remain identical, that the right checkpoint survives, or that restart refill is the fastest
qualified path.

Known gaps at the start of this plan are:

- the production-candidate sample still prefills 47,271 tokens and reports two cold sessions plus
  four prompt divergences across 14 calls;
- root identity is inferred from outcomes rather than recorded directly, so controller drift,
  tokenizer drift, tool-schema drift, retention, and missing disk state are difficult to separate;
- readiness-driven tool narrowing can create several legitimate authority variants, but those
  variants are not represented by a stable bounded type or measured independently;
- the in-memory cross-session FlashMoe prefix collection retains four entries by a hard-coded count;
  no evidence establishes whether that is sufficient or wasteful for the real authority variants;
- cache lookup, disk deserialization, Metal state hydration, fresh suffix prefill, checkpoint capture,
  and persistence are not timed as separate refill phases; and
- the recent qualification ran a three-case live slice, not the complete 24-case live corpus or a
  cold/warm/restart matrix.

## Invariants

These are release blockers, not optimization preferences.

### Correctness and authority

- Cache reuse remains exact-token reuse. A stage-root identifier, message hash, stage name, or tool
  class alone is never enough to restore model state.
- The root includes the exact native tool schema, tokenizer/chat-template output, and all system
  instructions that affect prefill. Output-constraint mode is recorded and freshly enforced for the
  invocation; when it does not change rendered input tokens, it need not duplicate KV state.
- Tool narrowing remains fail-closed. pb must not expose a broader tool schema merely to obtain a
  cache hit.
- Review stays a fresh workflow stage. Reusing its stable instructions and tool authority does not
  reuse another reviewer's conclusion or task-specific evidence.
- A cache failure may fall back to a truthful fresh prefill only before request state is mutated.
  A graph, state-hydration, or refill failure after mutation is terminal and cannot silently retry a
  different runtime path.
- Existing contract, check, review, allowed-path, semantic-commit, and clean-worktree gates remain
  unchanged.

### Privacy and security

- Prompts, repository paths, source text, tool arguments, and task identifiers are not added to
  telemetry. Root and namespace diagnostics use versioned digests, enum values, and token counts.
- Cache state remains local. This plan adds no network lookup, cloud cache, remote telemetry, or
  cross-machine synchronization.
- Files remain owner-only and atomically published. Reads reject symlink escapes, oversized records,
  version changes, fingerprint mismatches, malformed manifests, and incompatible state.
- Cache inspection reports paths, budgets, counts, ages, and digests only through an explicit CLI or
  existing diagnostic surface. It never prints cached tensors or recovered prompt text.
- User-visible controls use typed configuration or explicit CLI arguments. No `PB_*` feature flag is
  introduced.

### FlashMoe architecture

- The load-resolved resident or streamed expert policy does not change during refill.
- The shared scheduler continues to own streamed expert I/O and trusts the OS page cache. This plan
  does not add an application expert cache or a checkpoint-specific expert fast path.
- Qwen variants use the same prepared layer-major data flow. Any change to graph selection, state
  hydration, expert I/O, command scheduling, or CPU/GPU handoff updates the curated FlashMoe
  architecture in the same commit.
- The scalar path remains the exact qualification reference, not an unreported production fallback.

## Measurement contract

Optimization begins with a machine-readable baseline. Extend native invocation telemetry with a
privacy-safe `prompt_root` record containing at least:

- descriptor version;
- backend and model-namespace digest;
- invocation purpose, workflow stage, and bounded authority class;
- system-instruction version, tool-schema digest, output-constraint mode, and rendered root-token
  digest;
- rendered root tokens, root tokens reused, total cached tokens, and fresh suffix tokens;
- lookup outcome and miss reason; and
- storage tier used: none, live session, memory root, disk session, or disk root.

The root digest is calculated from the exact rendered tokens under the model namespace. Hashes of
pre-rendered strings may be included for diagnosis but cannot replace it.

Extend FlashMoe native statistics with non-overlapping refill phases:

1. session and root lookup;
2. disk open/read and checkpoint decode;
3. CPU state validation and allocation;
4. Metal KV/recurrent hydration;
5. fresh suffix prefill;
6. prompt-boundary snapshot capture; and
7. durable persistence queued and completed.

For suffix prefill, retain model family, resident/streamed policy, command kind, suffix length,
chunk geometry, Metal command count, host upload/readback bytes, peak request allocation, and
prefill throughput. Totals must reconcile with overall invocation wall time without double-counting
asynchronous work.

Add a deterministic harness matrix with these dimensions:

| Dimension | Required cases |
| --- | --- |
| Cache state | disabled, empty, warm memory, warm disk after restart, corrupted disk, incompatible model namespace, byte-budget eviction |
| Stage authority | planning, plan review, implementation read, implementation mutation, code review, focused repair |
| Root change | identical, dynamic task evidence only, changed tool authority, changed system version, changed tokenizer/template |
| Prefix length | zero, short, approximately 512, approximately 4,354, and a 6–7k frontier |
| Fresh suffix | 0, 1, 31, 32, 33, 256, 1,024, and forced multi-chunk |
| Expert policy | resident and forced streamed where supported |

The full cross-product is not required on every commit. Checked-in fixture tiers define a fast
deterministic set, a focused local-Metal set, and a release qualification set. Every reported run
records the exact revision, binary digest, model, tokenizer/template fingerprint, sampling settings,
machine class, cache directory disposition, and retained result path.

## Phase 0 — Make root behavior observable

### Implementation

- Introduce a backend-independent `StageRootDescriptor` owned by the managed invocation request. Use
  enums for invocation purpose, workflow stage, and authority class; do not infer them later from
  prompt text.
- Have each backend add its model namespace and exact rendered-token digest. Keep backend-specific
  cache keys private to the backend while exposing the common diagnostic fields.
- Split `prompt_diverged` into enough internal evidence to distinguish session divergence from a
  missing exact root checkpoint. Preserve a bounded public reason taxonomy if more detail would be
  unstable or disclose implementation internals.
- Add the refill phase timings and reconciliation tests described above.
- Extend the harness summarizer and usability auditor so aggregate reports include eligible root
  tokens, reused root tokens, root hit rate, refill tokens, refill wall time, and miss counts by
  stage and authority class.
- Record the existing three-language baseline through the new fields before changing root layout or
  refill execution.

Likely owners are `src/agent_core.rs`, `src/inference/mod.rs`,
`src/inference/flashmoe/runtime/generation.rs`, `src/inference/flashmoe/session_cache.rs`,
`src/inference/llamacpp/mod.rs`, `src/events.rs`, `src/harness_eval.rs`, the harness audit scripts,
and their web transcript types.

### Gate

- Every managed invocation reconciles `root_reused <= cached <= prompt` and
  `cached + actually_prefilled = prompt` under the backend's documented generation-prompt rules.
- Two identical rendered roots have the same digest without recording their content. A one-token
  difference has a different digest and cannot hit.
- A scripted session mismatch followed by a successful disk-root lookup reports a root hit, not a
  prompt divergence.
- The locked pre-change live sample is retained as the Phase 0 benchmark record.

## Phase 1 — Canonicalize stage roots

### Implementation

- Move stable-root construction behind one controller-owned builder. Its input is the versioned
  base agent instruction plus a bounded `StageAuthorityClass`; dynamic task, repository, branch,
  run, evidence, plan, and contract material remains in later messages.
- Replace incidental readiness-driven schema combinations with an explicit finite authority-class
  mapping. A class describes exactly the tools and constraints available for that controller state;
  it does not grant a superset for cache convenience.
- Canonicalize tool ordering, JSON Schema property ordering, descriptions, and constraint metadata
  before chat-template rendering. Semantically identical authority must serialize identically.
- Give intentional instruction or schema changes an explicit root-version change. Tests should fail
  on accidental root drift and require fixture review for intentional drift.
- Preserve stage isolation for logical session heads. Cross-session root reuse may share only the
  immutable root checkpoint; task-specific continuation state remains session-owned.
- Exercise both FlashMoe and llama.cpp prompt renderers so the controller contract is common even
  though their checkpoint formats are separate.

### Gate

- The same model/template, prompt-root version, stage, and authority class produce byte-identical
  root tokens across two repositories, two tasks, two sessions, and a process restart.
- Changing dynamic task evidence leaves the root unchanged while changing any rendered authorized
  tool, stable instruction, tokenizer/template, backend, or model namespace invalidates it. A
  decode-only constraint change may reuse identical KV state but must bind a fresh constraint engine
  and remain distinguishable in the authority descriptor.
- The finite authority-class fixture covers every reachable managed invocation state. An unclassified
  state fails closed before inference.
- Warm and restart matrix runs report 100% reuse of eligible root tokens for unchanged roots and
  zero false hits for changed roots.

## Phase 2 — Qualify cache namespaces and retention

### Implementation

- Route all inference cache paths through one typed resolver rooted at `storage.cache_dir`. Enumerate
  model artifacts, llama.cpp sessions, FlashMoe sessions, and FlashMoe prompt roots as separate
  versioned namespaces. Do not migrate or delete an existing cache until the old and new paths have
  been resolved and tested explicitly.
- Record a namespace digest and human-readable backend/format version in diagnostics. Model,
  tokenizer/template, quantization, runtime state schema, and checkpoint format changes must be part
  of the compatibility boundary either directly or through the validated model fingerprint.
- Audit the four-entry in-memory prompt-root limit against observed authority variants and snapshot
  sizes. Retain it if memory and disk-restore evidence justify it; otherwise replace it with a
  resource-derived byte budget and LRU policy. Do not simply increase the count.
- Ensure disk pruning accounts for checkpoints and manifests together, never removes files needed by
  the request currently being committed, and leaves no manifest pointing to a partially published
  checkpoint.
- Add explicit inspection and cleanup through an existing typed CLI/configuration surface. Cleanup
  must resolve an exact versioned namespace and use recoverable or narrowly bounded deletion rules.
- Test concurrent sessions, two pb processes, interrupted persistence, corrupt records, full disks,
  read-only roots, symlink attacks, model upgrades, and a storage-root change.

### Gate

- One configured storage root is used consistently by production model loading and session-cache
  persistence. A diagnostic test detects any fallback to an unrelated platform-default path.
- A process restart restores every valid stage root in the matrix without re-prefilling it.
- Corrupt, stale, missing, and evicted cache state produces the exact documented miss reason and a
  safe pre-request fallback. No case claims a hit or partially restores incompatible state.
- Peak resident memory and disk usage remain within their declared budgets. The retention policy is
  justified by measured stage-root reuse, not a hard-coded stage count.

## Phase 3 — Profile and optimize FlashMoe refill

Phase 3 starts only after Phase 0 can distinguish cache lookup, state hydration, and suffix prefill.
Do not optimize total TTFT from an aggregate profile that cannot identify which phase changed.

### Baseline

Run the restored-prefix harness matrix with automatic, forced scalar, and forced layer-major modes.
For every case, establish:

- whether the prefix was restored from memory or disk;
- exact cached and fresh suffix positions;
- CPU decode/validation and Metal hydration time;
- selected prefill command and chunk geometry;
- Metal command, upload, and readback deltas;
- final hidden, KV, router, and recurrent-state fingerprints; and
- greedy continuation equivalence.

### Optimization decision tree

Apply only the branch supported by the profile:

1. **Lookup or disk decode dominates:** reduce redundant manifest/checkpoint reads, validate once,
   and avoid copying serialized buffers more than required. Preserve the current owner-only atomic
   format unless a versioned replacement passes corruption and migration tests.
2. **Metal hydration dominates:** batch validated KV/recurrent uploads and eliminate duplicate
   CPU-to-GPU transfers. Keep complete state validation before mutating the live graph.
3. **Suffix prefill dominates:** ensure `auto` evaluates the actual remaining suffix length, sends
   suffixes at or above the prepared threshold directly to the qualified device-resident
   layer-major graph, and retains correct restored positions across forced chunks.
4. **Snapshot capture or persistence dominates:** avoid recapturing unchanged root state, keep the
   speculative generated head memory-only, and move durable writes off the inference critical path
   only when completion, error reporting, shutdown flush, and byte-budget ownership remain explicit.
5. **Short suffixes dominate:** measure scalar command overhead before adding a new command shape.
   Promote a short-suffix path only with exact parity, a material end-to-end win, and a typed plan for
   all supported Qwen MoE variants.

Likely owners are the FlashMoe generation lifecycle, session cache, typed KV/recurrent state,
Qwen runtime graph, scheduler-owned streamed expert command, and existing prefill harness. Any
data-flow change updates `docs/flashmoe-architecture-parity-plan.md` before promotion.

### Correctness gate

- Zero-prefix and restored-prefix scalar/layer-major runs match the existing declared state parity
  for 0/1/31/32/33-token and multi-chunk suffixes, under resident and forced-streamed policy.
- `prefill_start` is never less than the exact restored-token count and never exceeds prompt length.
  Restored tokens are not evaluated again.
- Automatic selection uses layer-major for every supported prepared suffix at or above the
  production threshold and records why any other suffix uses scalar.
- Allocation, encoding, hydration, or prefill failure never leaves a reusable partial checkpoint or
  silently changes graph, expert policy, quantization, model, or backend.
- The raw-token parity fixtures, structured smoke, session A/B restore, and explicit llama.cpp GGUF
  control continue to pass.

### Performance gate

- Compare against a checked-in Phase 3 baseline on the same binary settings and machine class.
- No suffix geometry may regress fresh-suffix prefill throughput by more than 5% without a documented
  correctness or resource justification.
- The promoted change must improve the phase it targets by at least 20% and improve restartable
  stage TTFT by at least 10%; otherwise retain the measurement and do not add production complexity.
- Root-only restoration with a zero-token suffix performs no prefill command. Memory restoration and
  restart restoration report their latency separately.
- Peak request/session allocation remains within the existing 5% FlashMoe promotion allowance and
  ends with zero leaked transient buffers or in-flight commands.

## Phase 4 — Integrate and qualify the managed workflow

### Implementation

- Add a harness cache scenario that runs the same bounded task in four modes: empty storage root,
  warm same process, new session in the same process, and new process using the persisted root.
- Add a second scenario that changes only the authority class and proves the old root is rejected
  while an already-populated matching root remains reusable.
- Keep useful call count separate from cache efficiency. Planning, review, mutation, and repair
  quality are reported independently from root reuse and refill speed.
- Run the locked Rust registry, Python TTL, and React alert cases with the same model, backend,
  sampling, contracts, and machine used by the 25 July baseline. Preserve raw events, scratch state,
  independent audits, binary digest, and a machine-readable aggregate.
- If controller changes alter a task result, classify the failure before changing inference. A model
  edit error, workflow-control defect, cache defect, runtime defect, and experiment error remain
  separate outcomes.

### Promotion gate

- Official correctness and pb verified-clean completion remain 3/3 with zero false verified
  completion and no forbidden mutation.
- The successful Rust and Python cases remain at four useful calls. Repairs remain bounded to the
  failed path and do not re-enter planning.
- After the required cold population, unchanged stage roots have 100% eligible-root-token reuse in
  warm, cross-session, and restart modes. Changed authority has zero false hits.
- Against the 47,271 fresh-prefill-token baseline, the identical locked sample must reduce total
  fresh prefill by at least 25% to 35,453 tokens or fewer. The 50% stretch target is 23,635 tokens.
- Total wall time and measured energy each improve by at least 15%, with no individual successful
  case regressing by more than 10% without a recorded machine-level explanation.
- Cache corruption and cache disablement retain correct task behavior and truthful diagnostics;
  they are excluded from performance scoring but included in correctness qualification.

If stable roots are smaller than the target makes achievable, retain the exact root-hit result and
revise the aggregate-token gate only with a token-accounted explanation in the benchmark record.
Do not inflate a root, expose more tools, or move dynamic evidence into the root to satisfy the
metric.

## Phase 5 — Production qualification and rollout

### Required qualification

- Run all 24 private-workload cases from clean workspaces with the release binary. Retain the
  independent official result, pb outcome, commit/worktree audit, invocation purposes, stage-root
  statistics, refill statistics, wall time, generated tokens, and energy estimate.
- Run focused failure cases for malformed model output, failed mutation, failed check, review
  rejection, context reset, disk corruption, full cache budget, interrupted persistence, and model
  fingerprint change.
- Run the full Rust, web, documentation, and macOS arm64 release gates. FlashMoe changes additionally
  run scalar/layer-major restored-prefix parity, resident/streamed resource qualification, the
  required native one-token smoke, and an explicit llama.cpp compatibility smoke.
- Update the curated workflow, security, local-privacy, configuration, and working-with-pb chapters
  for behavior that actually shipped. Keep this file and the benchmark record labelled as design
  evidence until every corresponding gate passes.

### Rollout and rollback

- Land measurement separately from behavior, root canonicalization separately from cache lifecycle,
  and each FlashMoe data-flow optimization separately from promotion. Every commit is semantically
  scoped and independently testable.
- Cache-format or namespace changes use an explicit version. Incompatible old state is ignored
  safely; it is not silently rewritten during inference.
- A production regression is rolled back by reverting the relevant behavior commit or selecting an
  already-supported typed configuration/CLI mode. Do not add a hidden environment switch.
- Preserve pre-change benchmark records and fixtures so rollback restores a known comparison rather
  than erasing unfavorable evidence.

### Final production gate

The work may be called production-ready only when:

- all Phase 0–4 gates pass on the release candidate;
- the 24-case corpus has no pb false verification, cache-related correctness failure, forbidden
  mutation, or unbounded workflow regression;
- all exact-root, corruption, restart, and restored-suffix parity cases pass;
- performance and resource gates pass on the locked qualification machine;
- current user and architecture documentation describes only shipped behavior; and
- the repository is clean at the recorded release revision.

## Planned delivery sequence

The sequence is deliberately evidence-first:

1. `feat: identify stable prompt roots`
2. `test: baseline prompt root reuse`
3. `refactor: canonicalize stage prompt roots`
4. `fix: unify inference cache namespaces`
5. `test: qualify restartable prompt roots`
6. `perf: measure flashmoe refill phases`
7. one narrowly named FlashMoe optimization commit selected by the profile
8. `test: qualify flashmoe restored suffixes`
9. `test: qualify cached private workflows`
10. `docs: record prompt cache production qualification`

Commit names are proposed boundaries, not permission to combine unrelated changes. If Phase 0
shows that a named optimization is unnecessary, record that result and omit the production change.

## Evidence ledger

Each phase appends a dated row rather than rewriting the original baseline:

| Phase | Revision | Artifact | Result | Status |
| --- | --- | --- | --- | --- |
| Call-reduction baseline | `a69239ce` | `benchmarks/private-workload-usability.md` and locked JSON baseline | 3/3 verified clean; 14 calls; 47,271 fresh-prefill tokens; 15m 52s; 9.97 Wh | Shipped baseline |
| Phase 0 root baseline | `f6fd5a71` | [Single-case diagnostic](benchmarks/prompt-root-phase0.md) and `fixtures/harness-usability/baselines/2026-07-25-prompt-root-phase0-rust.json` | Rust 1/1 verified clean; 4/4 complete disk-root hits; suffix prefill dominated | Historical partial baseline; superseded below |
| Phase 0 observability | `f6fd5a71` through `bb3e89f9` | Privacy-safe descriptors, miss detail, refill phases, auditor aggregate, and full-corpus JSON | 127/127 reconciled invocations; bounded root/namespace identity and all refill phases retained | Qualified |
| Phase 1 stable roots | `6aae3c8d` | Controller descriptor, canonical-tool, cross-repository, backend-renderer, dynamic-evidence, and fail-closed fixtures | Exact authority variants stable; changed rendered authority invalidates; 268,147/268,147 eligible tokens reused in final corpus | Qualified |
| Phase 2 cache lifecycle | `3ef1c449`, `bb3e89f9` | Byte-budget LRU, corruption/fingerprint/symlink/manifest/concurrency/storage-root fixtures; cold/restart Task-root pair; late corpus retention cases | Hot disk recency retained; sessionless root restores 30/30 after restart; 24/24 persistence completed | Qualified for shipped paths |
| Phase 3 refill | `6aae3c8d` | Retained resident/streamed, scalar/layer-major, restored-suffix, 0/1/31/32/33, and resource matrix | Exact parity and balanced resources; suffix prefill dominated; existing graph already selects actual suffix, so no extra fast path was promoted | Profile complete; no behavior change justified |
| Phase 4 workflow | `bb3e89f9` | Locked Rust/Python/React rerun in [qualification record](benchmarks/stable-prompt-root-production.md) | 3/3 correct; 100% roots; fresh prefill -25.73%; energy -24.22%; wall only -3.83%; Python 5 calls | **Promotion blocked** |
| Phase 5 production | `bb3e89f9` | Complete 24-case audit, failure matrix, llama.cpp control, macOS release and native smoke | 21/24 verified clean, 3 truthful model/control limits, 0 false verification/cache defects/experiment errors; release gates pass | Cache candidate qualified; production-ready designation open |

Passing a unit test, a one-token smoke, or the four-call happy path does not close an open row. Only
the named retained evidence and its acceptance gates can change this plan's status.
