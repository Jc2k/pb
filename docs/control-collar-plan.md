# Streaming tool-output control collar

Status: **Phases 0–5 are implemented and production-qualified. Phase 6 and the Phase 7/8
foundations are implemented but still completing their production qualification matrix.** This is
the durable design record, migration history, and evidence ledger for generation-time mutation
validation. Current curated behavior is described in the architecture documentation. Phase 7
native compiler profiles, portable dependency fact packs/name biasing, and Phase 9 speculation are
still design-record work and are not implied by the syntax-valid production milestone.

Current implementation ledger:

- **Phase 6 implemented:** conservative UTF-8/lexical/delimiter/Python-indentation prefix rules now
  advance only over newly committed logical bytes, use constant-time persistent-stack checkpoints,
  and retain bounded candidate-branch checkpoints. Incremental Tree-sitter edits/checkpoints and
  chunk-independent canonical patch validation share the same oracle; partial hunk additions and
  context are probed before their newline. Qwen and DeepSeek real-tokenizer split/rollback/chunk
  qualification passes the 1 ms p95 budget. The claim remains conservative prefix safety, not exact
  grammar extendability; longer fuzzing and refreshed end-to-end throughput evidence remain.
- **Phase 7 foundation configurable:** immutable semantic-world/document identities, exact
  full-content LSP overlays with monotonic versions, fresh immutable shadow workspaces and isolated
  LSP sessions, full pull-diagnostic baselines, diagnostic-debt deltas, required/advisory policy,
  repairable payload-close steering, final executor revalidation, and separate content-free
  generation/final receipts are implemented for classified rust-analyzer, TypeScript, and Pyright
  errors. Required Rust evidence additionally needs a loaded non-empty crate graph and per-document
  crate membership. Enforcement defaults to `disabled`. The first digest-pinned Rust provider
  qualification failed safely because its packaged analyzer produced no crate graph; no semantic
  profile has been promoted. Native compiler parity, strict Python/JavaScript profiles, boundary
  latency, multi-project overlays, and portable dependency fact packs remain promotion gates.
- **Phase 8 adapter implemented:** llama.cpp now masks the full vocabulary before sampling
  truncation, accepts sampler state once, reuses both tool dialects and the shared executor, and
  reports the same guarantee/rejection fields. Pinned live model/tokenizer/template differential
  qualification remains before any broad backend-parity claim.
- **Phase 9 capability only:** both backends explicitly report `candidate_probe_only`. The replay
  and snapshot variants exist as typed future capability names but are not implemented or inferred
  from session caches.

## Decision

The deterministic core of output constraints is implemented in a new unpublished Rust crate named
`pb-control-collar`. The repository is a non-virtual Cargo workspace containing the existing root
`pb` package and that crate.

The workspace contains exactly those two members. The crate remains an internal correctness boundary,
not a general decomposition of pb into micro-crates.

The dependency direction is strict:

```text
pb binary/library
    |
    +-- controller authority and workspace snapshots
    +-- FlashMoe / llama.cpp sampling adapters
    +-- model-state checkpoint or replay implementation
    +-- filesystem, Git, process, and LSP ownership
    |
    `--> pb-control-collar
             +-- tokenizer vocabulary and mask contracts
             +-- Qwen JSON and DeepSeek DSML tool-envelope state
             +-- decoded logical argument events
             +-- virtual mutation transactions
             +-- canonical streaming patch engine
             +-- syntax and semantic analyzer interfaces
             +-- deterministic receipts and diagnostics
```

`pb-control-collar` must never depend on `pb`, read the live workspace, execute a tool, run Git,
own a model graph, or decide which capability the agent has. It narrows possible output; it does not
grant authority.

## Outcome

For supported constrained mutation tools and language profiles, pb provides this guarantee:

> A completed model-generated mutation cannot execute unless the exact virtual result authorized by
> the controller is structurally valid, the mutation still applies to the same base snapshot, and
> the executor independently accepts the prepared result.

The default production guarantee remains deliberately syntax-scoped. Opt-in semantic providers can
reject definite name, type, ownership, and API errors at repairable payload closure and immediately
before publication, but state that stronger rung relative to a pinned language analyzer,
configuration, workspace, dependency identity, and complete baseline. Python runtime behavior
cannot generally be proven by a static streaming analyzer.

The priority language profiles are Rust, Python, TypeScript, JavaScript, HTML, and CSS. The initial
tool order is `write_file`, then the shared replacement/edit path, then a canonical subset of
`apply_patch`. Qwen/GLM FlashMoe is the first sampling integration. DeepSeek DSML is the second wire
dialect and is an architectural gate, not an afterthought. llama.cpp is now an adapter to the same
collar contracts, with profile qualification tracked separately from implementation.

## Why a crate boundary is warranted

This feature combines several state machines whose cross-product needs unusually thorough unit,
property, fuzz, differential, and tokenizer-split testing:

- model-specific tool envelopes;
- JSON and DSML escaping;
- tokenizer byte and special-token behavior;
- tool schema and argument ordering;
- virtual filesystem mutations;
- unified-diff hunk accounting and context matching;
- incremental language parsing;
- semantic obligations and construct boundaries; and
- sampling masks, biases, commit, checkpoint, and rollback.

Keeping those state machines in FlashMoe generation code would couple correctness to one output head
and make executor parity difficult to prove. A pure library can replay arbitrary token streams,
compare streaming and batch outcomes, fuzz without loading a model, and be reused by every backend.

The split also creates risks. A premature public API could freeze the wrong abstraction, workspace
builds can accidentally omit a member, and moving filesystem authority into the crate would blur the
security boundary. The mitigations are:

- keep the crate unpublished and workspace-internal initially;
- expose model-neutral source and tool events rather than pb controller types;
- keep live I/O and external-process ownership in `pb`;
- add one crate, not a speculative family of micro-crates;
- require differential parity before deleting an existing implementation; and
- version every wire, mutation, language, and receipt contract that can affect acceptance.

## Workspace shape

The root manifest can remain both the `pb` package and the workspace root:

```toml
[workspace]
members = [".", "crates/pb-control-collar"]
default-members = [".", "crates/pb-control-collar"]
resolver = "3"

[package]
name = "pb"
# existing package metadata remains here
```

The implemented package lives at:

```text
crates/pb-control-collar/
    Cargo.toml
    src/
        lib.rs
        vocabulary.rs
        mask.rs
        protocol/
            mod.rs
            dsml.rs
        json.rs
        tool.rs
        mutation/
            mod.rs
            snapshot.rs
            write.rs
            patch.rs
        analysis/
            mod.rs
            prefix.rs
            semantic.rs
            syntax.rs
        gate.rs
        receipt.rs
        diagnostics.rs
```

Only dependencies used by both packages should move to `[workspace.dependencies]`, and that move
should be mechanical and separately reviewable. `Cargo.lock` remains at the workspace root. The
workspace conversion must update developer and CI commands so formatting, Clippy, and tests cover
all members; relying accidentally on Cargo's implicit package selection is not acceptable.

Do not add a user-visible feature flag or `PB_*` environment control for the collar. Experimental
selection belongs in explicit harness arguments. Production enablement is a typed, load-resolved
backend capability and later, if user choice is genuinely needed, typed configuration mirrored by
`src/init.rs`.

## Baseline at design approval

The implementation started from these shipped boundaries:

- `Cargo.toml` was one `pb` package and depended on LLGuidance 1.7.6 directly.
- `src/inference/flashmoe/json_constraints.rs` owned the tokenizer-scoped LLGuidance factory and a
  request-local strict-JSON matcher.
- `src/inference/flashmoe/constraints.rs` owned the handwritten Qwen native tool-envelope and JSON
  prefix constraint.
- `src/inference/flashmoe/runtime/generation.rs` obtained LLGuidance's full-vocabulary mask before
  sampling. Qwen-family output can apply it in the resident Metal vocabulary head; DeepSeek applies
  the same mask to its existing full-logit output.
- That generation path deliberately set the ordinary native tool constraint to `None` for DeepSeek.
  DeepSeek tool output is parsed afterward as DSML by `src/inference/flashmoe/text.rs`.
- DeepSeek's tokenizer adapter already distinguished JoyAI GPT-2 byte-BPE bytes from model-native
  special tokens, and its session implementation can capture the complete model state for bounded
  exact-prefix reuse.
- `src/agent_core.rs` wrote validated `write_file` content only after generation and checked
  `apply_patch` through `git apply --check --recount` before `git apply --recount`.
- `src/lsp.rs` owned proactive language-server process and document-overlay infrastructure. That
  process authority should remain outside the collar crate even when its analysis results inform a
  collar session.

That baseline proved the tokenizer and sampling hooks existed. Phases 0–5 added the shared DSML
dialect, logical mutation boundary, virtual patch transaction, and complete-file syntax close gate.
The Phase 7 work described below was outside that original shipped boundary; its current opt-in
foundation is recorded separately from that historical baseline.

## Implemented production slice

The current implementation has these concrete boundaries:

- the root package and unpublished `pb-control-collar` are explicit workspace/default members using
  resolver 3; dependencies used by both packages are workspace-owned while exact-pinned Tree-sitter
  dependencies remain private to the collar;
- the collar owns tokenizer byte/control-token surfaces, full-vocabulary masks, the LLGuidance JSON
  factory/session, versioned manifests and analyzer/recovery interfaces, DSML parsing/probing,
  controller-snapshot virtual writes, canonical patches, syntax profiles, receipts, and content-free
  rejection codes;
- Qwen/GLM retains its qualified JSON-envelope adapter but delegates mutation closure and virtual
  result acceptance to the shared collar gate; DeepSeek uses the collar DSML parser and candidate
  probe over its full-logit output while preserving native `｜DSML｜` token identity;
- the controller supplies at most 32 MiB of fresh exact read evidence for mutation generation;
  FlashMoe refuses to decode an exposed mutation without that immutable snapshot;
- Rust, Python, TypeScript, TSX, JavaScript/JSX, HTML, and CSS use exact pinned Tree-sitter grammar
  versions, reject invalid UTF-8 plus error/missing nodes, and apply embedded-language checks for
  supported HTML script/style/JSON content;
- constrained patches are bounded to 32 files and 256 hunks, require exact counts, offsets, context,
  deletion bytes, and paths, and produce all virtual results before execution; and
- conservative UTF-8, delimiter, and Python-dedent prefix rules now reject only promoted local
  impossibilities; canonical patch lines additionally stream untouched base bytes and generated
  result bytes through the same prefix oracle between hunks;
- configured semantic providers can receive exact full-content overlays, compare baseline and
  candidate diagnostic debt, reject definite classified errors at mutation-payload closure, and
  repeat the transaction gate before publication. This is opt-in and defaults to disabled;
- llama.cpp has a full-vocabulary pre-truncation adapter for both supported dialects and shares the
  constrained mutation/executor path, but pinned live model/tokenizer/template qualification is
  still outstanding; and
- the executor independently reconstructs the mutation, verifies live bases and syntax, publishes
  file writes atomically, and requires exact `git apply --check`/`git apply` parity for constrained
  patches without `--recount`.

This implementation prevents an invalid supported complete file from closing and executing, and it
hard-masks a deliberately small set of proven-impossible prefixes. It does not claim exact grammar
extendability, expression-level type steering, complete project-wide symbol/type correctness, a
pinned production semantic profile, live-model llama.cpp parity, or model-state rollback. Those
remain independently promoted gates in Phases 6–9.

## Authority and correctness invariants

These invariants apply in every phase:

1. **The controller grants authority.** The collar receives only exposed tool schemas, terminal-tool
   declarations, allowed paths, immutable base bytes, and policies already authorized by pb.
2. **The collar does not read the workspace.** The controller creates a bounded snapshot before
   generation. This keeps inference deterministic and prevents generation-time path races.
3. **The executor remains authoritative.** A collar receipt is evidence for deterministic replay,
   not permission to skip path, schema, base-hash, capability, or final-content checks.
4. **Incomplete output never mutates.** EOS, cancellation, token exhaustion, parser failure, empty
   masks, analyzer failure, or an unfinished tool envelope cannot publish a partial mutation.
5. **Constraint failure is fail-closed.** A required constraint may not silently downgrade to
   unconstrained sampling or another model/backend.
6. **Accepted streaming and batch results agree.** The same wire transcript and snapshot must yield
   the same calls, virtual files, diagnostics, and receipt when replayed without a model.
7. **Masks precede sampling truncation.** A valid low-ranked token must remain reachable when the
   normal top-k frontier contains only invalid tokens.
8. **Only proven impossibility is hard-masked semantically.** Repairable or analyzer-unknown prefixes
   remain available until a construct or tool boundary makes the error definite.
9. **Diagnostics preserve privacy.** Durable events contain versions, digests, counts, states, and
   timings, not source text, patch text, argument values, or repository paths.
10. **No protocol/parser drift.** Generation constraints, the final wire parser, and the renderer's
    delimiter/escape definitions come from the same dialect implementation.

## Core request contract

The controller should construct one request-local manifest before inference:

```rust
pub struct CollarManifest {
    pub contract_version: ContractVersion,
    pub dialect: ToolDialect,
    pub mode: ToolConstraintMode,
    pub tools: Vec<ExposedTool>,
    pub terminal_tools: Vec<ToolName>,
    pub mutation_policy: MutationPolicy,
    pub workspace: WorkspaceSnapshot,
    pub language_profiles: Vec<LanguageProfile>,
    pub limits: CollarLimits,
}
```

The manifest contains canonical schema and policy digests. A `WorkspaceSnapshot` contains normalized
logical paths, file kinds, exact bytes where required, content hashes, and existence policy. It does
not contain open handles or grant permission to fetch another path. Snapshot construction, symlink
policy, ignore rules, repository boundaries, and size limits remain in `pb`.

The inference backend supplies a tokenizer vocabulary separately:

```rust
pub struct VocabularyEntry {
    pub token_id: u32,
    pub surface: TokenSurface,
}

pub enum TokenSurface {
    Bytes(Vec<u8>),
    Control {
        identity: ControlToken,
        visible_bytes: Vec<u8>,
    },
}
```

The distinction matters for DeepSeek. Its `｜DSML｜` marker is a tokenizer special token, while
ordinary byte-BPE tokens may end in the middle of a UTF-8 sequence. Protocol control terminals must
be able to match token identity; payload analysis must receive incrementally decoded logical bytes.

## Unified tool-event layer

JSON and DSML must converge before mutation handling:

```rust
pub enum ToolEvent<'a> {
    PreludeBytes(&'a [u8]),
    BeginCall { name: ToolName },
    BeginArgument {
        name: ArgumentName,
        encoding: ArgumentEncoding,
    },
    ArgumentBytes(&'a [u8]),
    EndArgument,
    EndCall,
    EndBatch,
}

pub enum ArgumentEncoding {
    JsonString,
    JsonValue,
    DsmlRawString,
    DsmlJson,
}
```

`ArgumentBytes` are logical bytes after the selected wire decoder has handled JSON escapes, Unicode
escapes, or DSML's one exact closing-tag escape. Mutation code must not parse raw JSON or search raw
DSML itself.

The dialect owns:

- model-native envelope phases and control markers;
- tool-name and parameter-name restriction;
- schema-supported scalar/container encodings;
- duplicate and missing parameter rejection;
- logical string decoding and delimiter escaping;
- terminal-call completion;
- incomplete/capped-output classification; and
- final parsing into the same canonical calls used during streaming.

Chat-template placement and model roles remain in `pb`, but the template must obtain tool-envelope
examples and delimiter constants from the dialect rather than duplicating them.

## Dependency-aware arguments

Some arguments determine how later arguments can be constrained. The manifest therefore needs a
canonical wire-order contract in addition to ordinary JSON Schema semantics:

```rust
write_file: path -> content
replace_file: path -> content
apply_patch: patch
```

For `write_file`, the path must be complete and authorized before content begins. Otherwise the
collar cannot select a language profile while the content is being emitted. The protocol compiler
may enforce this order even though JSON object properties or DSML parameters are semantically
unordered after parsing.

For a retry already bound to an exact target, the controller may provide that target out of band;
the emitted path must still match it. For `apply_patch`, paths are discovered by the canonical patch
parser and must resolve only to snapshot entries or explicitly permitted creations.

## Constraint session

The sampling-facing API must support both hard exclusion and future steering:

```rust
pub struct ConstraintStep {
    pub hard_mask: TokenMask,
    pub logit_biases: Vec<TokenBias>,
    pub stop: Option<StopReason>,
    pub state: CollarState,
}

pub trait ConstraintSession {
    fn next(&mut self) -> Result<ConstraintStep, CollarError>;
    fn probe(&mut self, token: u32) -> Result<TokenDecision, CollarError>;
    fn commit(&mut self, token: u32) -> Result<Vec<ToolEvent<'_>>, CollarError>;
    fn finish(&mut self, reason: FinishReason) -> Result<CollarReceipt, CollarError>;
}
```

`TokenMask` is owned by the collar contract rather than exposing an LLGuidance bitset type to pb.
The LLGuidance adapter can fill it efficiently. FlashMoe maps it to its resident vocabulary mask or
DeepSeek full-logit mask; llama.cpp maps it to its sampler integration. Sparse logit biases are
optional and never override a hard exclusion.

The implementation composes layers in this order:

```text
wire grammar hard mask
    -> logical argument decoder
    -> tool-schema and argument-order state
    -> mutation transaction state
    -> syntax/type boundary probes
    -> hard-mask intersection and semantic biases
    -> backend sampler
    -> atomic commit to every state machine
```

LLGuidance remains the preferred static grammar engine. Workspace-aware patch applicability and
language semantics remain explicit dynamic state machines around it; they must not be encoded as an
unmaintainable generated grammar.

## Mutation transaction

The collar operates on an in-memory virtual workspace. A transaction records every source event and
can create a deterministic prepared result without touching disk:

```rust
pub enum SourceEvent<'a> {
    BeginFile { path: LogicalPath, language: LanguageId },
    KnownBytes(&'a [u8]),
    GeneratedBytes(&'a [u8]),
    DeleteKnownBytes(&'a [u8]),
    EndFile,
}
```

Known and generated bytes remain distinct for diagnostics, caching, and patch correctness, but the
language analyzer sees their exact concatenated virtual result.

### `write_file`

Once `path` closes, the transaction resolves the pinned language profile. Decoded `content` bytes
stream into the virtual file and analyzer. Intermediate syntax may be incomplete. A token is rejected
only when the relevant parser can prove that no permitted continuation at the current boundary can
recover, or when it attempts to close the argument/call with an invalid complete file.

The first implementation may be weaker but still useful: allow incomplete content, probe every
candidate that could close the content argument, and exclude all close/EOS candidates until the
complete virtual file passes the pinned syntax profile. This immediately guarantees that an invalid
file cannot become an executable completed call without requiring a perfect prefix-language parser.

`replace_file` and exact-target edit operations should reuse the same file transaction after
`write_file` is qualified.

### `apply_patch`

Generation and execution must share a `PatchStream` and `PreparedPatch`; Git is not the streaming
correctness oracle.

The initial constrained dialect is a canonical text-only subset of unified diff:

- normalized authorized repository-relative paths;
- explicit create, modify, and delete policy;
- exact hunk old/new counts rather than `--recount` repair;
- ordered, non-overlapping, in-range hunks;
- exact context and deletion bytes at declared base offsets;
- explicit missing-final-newline handling; and
- no binary patches, submodules, symlink changes, mode-only changes, renames, copies, or ambiguous
  quoted path syntax.

For each affected file the stream performs:

1. emit unchanged known base bytes before the next hunk;
2. verify and emit context bytes;
3. verify and omit deletion bytes;
4. emit generated addition bytes;
5. continue from the base cursor to the next hunk; and
6. emit the remaining known base bytes when the file closes.

The same language analyzer therefore receives the complete virtual result even though only additions
were generated. A patch batch closes only when every affected supported file passes its configured
syntax gate and all patch counts/context checks are exact.

The existing Git-compatible path may remain for backends where constrained generation is not yet
enabled. It must not be silently substituted after a constrained canonical patch fails.

### Multiple calls

DeepSeek DSML and Qwen envelopes can contain multiple invokes. The manifest must declare whether a
batch is:

- independent and validated against one immutable snapshot;
- an ordered transaction where later calls see earlier virtual results; or
- restricted to a single mutation call.

The first release should prefer one mutation per generated batch unless executor behavior and
atomic rollback are made identical to an ordered virtual transaction. A final-valid workspace is
not sufficient if the real executor would reject an invalid intermediate call before a later repair.

## Language analysis

The analyzer contract must be transactional and more general than a byte-only syntax validator:

```rust
pub trait IncrementalAnalyzer {
    type Checkpoint: Copy;

    fn begin(&mut self, snapshot: ProgramSnapshot) -> Result<(), AnalysisError>;
    fn checkpoint(&mut self) -> Self::Checkpoint;
    fn apply(&mut self, event: SourceEvent<'_>) -> Result<Analysis, AnalysisError>;
    fn rollback(&mut self, checkpoint: Self::Checkpoint) -> Result<(), AnalysisError>;
    fn finalize(&mut self) -> Result<Analysis, AnalysisError>;
}

pub struct Analysis {
    pub viability: Viability,
    pub closure: ClosureVerdict,
    pub obligations: Vec<SemanticObligation>,
    pub biases: Vec<RepairIntent>,
}

pub enum Viability {
    Valid,
    Repairable,
    Impossible,
    Unknown,
}
```

The crate owns this synchronous token-time interface and the value types. `pb` owns any process,
thread, timeout, cancellation, overlay-document, or LSP lifecycle needed to implement a provider.
Do not make Tokio, a language server, or the live filesystem a requirement of the core crate.

### Syntax profiles

Start with pinned Tree-sitter grammars for:

- `.rs`;
- `.py` and `.pyi`;
- `.ts` and `.tsx`;
- `.js`, `.jsx`, `.mjs`, and `.cjs`;
- `.html` and `.htm`; and
- `.css`.

A complete-file success contract must be versioned and explicit: valid UTF-8 where required, the
parser consumes the entire virtual file, and the resulting tree contains no error or missing nodes.
Where a grammar's recovery behavior is too permissive, add a language-specific final parser or
toolchain gate before describing the result as syntax-valid.

HTML requires included-range handling for embedded script and style content. The profile must define
how script `type` values select JavaScript, TypeScript, JSON, or uninterpreted text. Python needs an
explicit indentation and newline policy rather than treating significant whitespace as a generic
token stream.

The first gate is complete-file validity at argument closure. Stronger rejection of impossible
prefixes comes only after per-language extension tests prove it sound.

### Future semantic steering

Semantic validity is not prefix-monotonic. A currently mismatched Rust or Python expression may be
extended with a conversion or method call before its statement closes. The collar should therefore:

- hard-mask only tokens proven to make a valid continuation impossible;
- retain expected types, visible symbols, imports, scopes, and unresolved obligations;
- apply soft biases toward tokens that discharge those obligations;
- probe statement, expression, item, function, file, and tool-call boundaries;
- prevent a boundary token such as `;`, newline, dedent, `}`, or tool close from committing a
  definite error; and
- run a final authoritative analyzer over the complete virtual workspace.

Rust can eventually use rust-analyzer or compiler-compatible analysis for names, types, traits,
ownership, `cfg`, and macro-aware checks. Python strictness must be defined relative to a pinned
analyzer and configuration; `Any`, dynamic imports, monkey-patching, descriptors, and runtime-only
dispatch yield `Unknown` rather than a false guarantee. TypeScript and JavaScript similarly require
a declared project/compiler configuration rather than an isolated-file claim when module resolution
or ambient types matter.

No full compiler should run once per vocabulary token. The fast path uses grammar masks, byte/token
classes, cached analyzer checkpoints, and boundary detection. Expensive analysis happens at a small
number of structural boundaries or on speculative spans.

### Guarantee ladder for later phases

The phrase "incremental validity" must not collapse several materially different claims. Later
phases promote capabilities one rung at a time and report the highest rung actually active for each
language, analyzer profile, backend, and request:

| Rung | Production claim |
| --- | --- |
| Complete syntax | Shipped: the exact completed virtual file passes the pinned complete-file syntax profile before execution. |
| Conservative prefix safety | No committed token violates any promoted lexical, delimiter, indentation, patch, or grammar rule that can prove the prefix has no valid continuation in its declared language subset. Unknown prefixes remain open. |
| Scoped symbol resolution | Against one immutable semantic world, the final mutation introduces no provider-classified definite unresolved-name, import, field, method, or module errors in the enforced scope. |
| Scoped type correctness | Against a pinned analyzer, project configuration, target, and dependency graph, the final mutation introduces no provider-classified definite type/ownership errors in the enforced scope. This is not a proof of runtime behavior. |
| Backend parity | Every qualified backend applies the same manifest, token mask/bias ordering, mutation replay, final analyzer gate, and executor revalidation contract for the same dialect. |

Exact extendability is a valid claim only where an LLGuidance-compatible or equivalent source
grammar describes the accepted subset and differential tests prove that the source grammar and the
complete-file parser agree. Tree-sitter's incremental edit support is valuable for caching trees and
finding structural boundaries, but its error recovery is not by itself an impossibility oracle. A
language can therefore ship conservative prefix rules before it ships an exact grammar-backed
prefix subset, without overstating the guarantee.

### Semantic-world and provider boundary

Semantic analysis needs a second, separately bounded snapshot. The existing mutation snapshot is
optimized for exact file publication and is intentionally too small and narrow to stand in for a
project graph. Before constrained decoding, `pb` should prepare a content-addressed
`SemanticWorldSnapshot` containing:

- exact bytes for open/changed files plus immutable content bindings and digests for every other
  in-repository file admitted to the analysis scope;
- the target language/toolchain and analyzer identity/version;
- project configuration, feature/`cfg`, compiler-option, Python-environment, and target-platform
  digests;
- manifest and lockfile digests plus an offline dependency-graph identity; and
- a baseline diagnostic snapshot and its completeness state.

The analyzer runs against an immutable shadow workspace or document overlay derived from that
snapshot, never against a mixture of live and virtual bytes. Dependency source, declarations,
typeshed data, sysroots, package metadata, and build artifacts may be mounted read-only from
controller-owned caches. Network access is disabled. The provider may inspect repository and
dependency data authorized for analysis without exposing that source to the model or durable
telemetry.

`pb-control-collar` remains synchronous and I/O-free. It owns portable value types such as
`AnalyzerCapability`, `SemanticWorldId`, `BoundaryProbe`, `ProviderVerdict`, `DefiniteErrorClass`,
and `UnknownReason`. A controller-owned `SemanticProviderBroker` owns LSP/native analyzer processes,
deadlines, overlays, and cache leases. The broker produces immutable verdicts that a collar session
can consume. The current blocking boundary pauses decode and uses a one-way lock order from the
request-local model runtime to an individual LSP session; analyzer code must never call back into the
model runtime. A future worker/queue implementation must preserve that acyclic order while removing
the LSP wait from the runtime critical section.

The LSP client now accepts controller-provided bytes through monotonically versioned full-content
`didOpen`/`didChange` updates. Each semantic transaction creates an exact bounded shadow copy from
the captured content snapshot and a fresh provider session rooted at that copy; it rejects
symlinks, special files, copy-time live-workspace drift, and any provider mutation of the shadow.
Generation-time analysis requires a complete pull-diagnostic response. Stale, partial,
quiet-period-only, timed-out, restarted, configuration-divergent, empty Rust crate-graph, or
detached-document results are `Unknown`, never clean evidence.

### Boundary steering policy

Semantic checks compose with sampling in two tiers:

1. Apply the wire grammar and cheap full-vocabulary prefix mask before any sampling truncation.
2. Detect candidate tokens that close an expression, statement, item, file, tool argument, or tool
   call. Probe only the bounded ranked frontier, widen until enough valid candidates exist or the
   vocabulary is exhausted, and perform final top-k/temperature sampling only after those probes.

A definite error may hard-reject the boundary candidate that makes it observable. A repairable or
unknown result may create an obligation and sparse bias, but cannot remove the candidate. Symbol
completion, hover, signature help, and public API indexes are steering inputs, not hard-mask
authority: macros, dynamic dispatch, ambient declarations, forward items, and generated sources can
make a finite completion list incomplete.

This policy handles the motivating type mismatch without making the earlier counterexample unsound.
For example, a Rust expression with incompatible operands remains extendable until the model tries
to close the expression or statement; a provider-confirmed `type-mismatch` can reject that close and
leave conversion/method-call tokens reachable. An unresolved item that can be declared or imported
later remains an obligation until file/tool closure. If the already committed prefix can be repaired
only by changing earlier tokens, correctness still fails closed at closure; Phase 9 recovery is the
optional mechanism for rewinding and steering earlier.

Blocking boundary probes are the first production implementation. External analysis is not run for
every vocabulary token, and speculative decode is not a prerequisite for semantic correctness. A
required semantic profile that times out, loses its process, exceeds its snapshot/boundary budget,
or cannot establish a complete baseline prevents semantic closure. An advisory profile may return
`Unknown` and retain the shipped syntax-only behavior, but receipts and user-visible status must not
describe that request as semantically constrained.

## Checkpoint and rollback model

There are three distinct checkpoints and they must not be conflated:

1. **Grammar/analyzer checkpoint:** cheap request-local state used while probing candidate tokens.
2. **Mutation checkpoint:** virtual-file cursors, patch state, and semantic obligations.
3. **Decode checkpoint:** model KV/recurrent/DeepSeek complete state plus generated tokens and sampler
   state, used only when already committed tokens must be resampled.

Candidate probing before commit needs only the first two. This should be the initial implementation:
the model state has not yet consumed the candidate, so rejecting a semicolon, newline, close tag, or
EOS does not require model rollback.

Speculative multi-token generation or delayed external analysis requires the third checkpoint. The
backend must advertise one of:

```rust
pub enum DecodeRecovery {
    CandidateProbeOnly,
    ReplayFromBoundary,
    SnapshotAndRestore,
}
```

DeepSeek's existing in-memory session snapshots capture the complete four-stream, raw/compressed KV,
compressor, indexer, frontier, and hidden state needed for exact restoration. They are currently
bounded cross-turn checkpoints and may be too expensive for frequent branching. A future ephemeral
decode checkpoint must be separately budgeted and qualified; it must not reuse or evict the durable
two-session/two-checkpoint policy implicitly. Until then, DeepSeek uses candidate-before-commit
boundary checks and may omit speculative semantic resampling.

## DeepSeek design gate

DeepSeek support adds a wire dialect and backend adapter, not a second mutation or analyzer stack.
The current code already derives constraint bytes for its pinned JoyAI GPT-2 byte-BPE vocabulary and
applies LLGuidance's strict-JSON bitset over full logits. Ordinary tool calls remain post-hoc DSML
parsing and therefore define the missing integration.

The DSML phase must qualify all of the following:

- `｜DSML｜` and other model-native control tokens are matched by token identity;
- normal byte-BPE tokens can cross UTF-8 boundaries without forcing whole-prefix `String` decoding;
- thinking/prose, tool-required entry, invoke, parameter, and terminal phases are explicit;
- exposed tool names and parameters are constrained before sampling;
- `string="true"` streams raw logical text and reverses only the exact documented escaped closing
  parameter tag;
- `string="false"` uses a nested JSON constraint appropriate to the parameter schema;
- mutation dependency order such as `path` before `content` is enforced;
- complete terminal DSML calls stop semantically without waiting for EOS;
- truncated DSML is classified as a named incomplete action and never executed;
- multiple invokes follow the manifest's declared batch policy;
- hard masks and semantic biases are applied to full logits before top-k sampling; and
- the streaming dialect and final DSML parser accept and reject the same corpus.

DeepSeek experiments use explicit harness selection and the pinned local profile. No hidden model
family override, relaxed parser, alternate runtime, or unconstrained fallback is permitted.
Structured DeepSeek checkpoint keys include the exact rendered stable-root token digest. A
controller tool-schema or authority narrowing therefore starts a cold bounded session instead of
attempting to restore incompatible complete Metal state; unchanged roots retain exact prefix reuse.
Raw harness sessions keep their base identity so they can continue to qualify raw prompt extension.

## Executor handoff

A completed constraint session produces a content-free receipt plus a prepared mutation held in
request-local memory:

```rust
pub struct CollarReceipt {
    pub contract_version: ContractVersion,
    pub dialect_version: DialectVersion,
    pub manifest_sha256: Digest,
    pub transcript_sha256: Digest,
    pub base_files: Vec<FileDigest>,
    pub result_files: Vec<FileDigest>,
    pub patch_sha256: Option<Digest>,
    pub language_results: Vec<LanguageResult>,
    pub terminal_state: TerminalState,
}
```

Before mutation, `pb` must:

1. revalidate tool capability and arguments through the ordinary executor path;
2. normalize and authorize paths again;
3. compare every live base hash with the snapshot receipt;
4. replay or independently verify the complete prepared mutation;
5. rerun the final configured syntax/semantic gate over exact result bytes; and
6. publish the mutation atomically according to existing tool semantics.

Snapshot drift aborts the call. The executor does not ask the collar to fetch new bytes and does not
apply the generated patch to a newer file opportunistically.

## Performance contract

Correctness comes first, but the collar must remain usable with large vocabularies and slow local
models:

- precompute tokenizer surfaces, control-token identities, schema grammar, and stable base parse
  state during request preflight;
- compute the complete protocol hard mask before top-k;
- compose bitsets without materializing decoded strings for every candidate;
- cache candidate probes by analyzer-state digest and token ID;
- invoke expensive analyzers at construct boundaries, not for every vocabulary entry;
- bound virtual workspace bytes, affected files, hunks, analyzer checkpoints, speculative depth, and
  diagnostic counts;
- record protocol-mask, dynamic-probe, analyzer, rollback/replay, and total sampling time separately;
  and
- treat an empty valid-token set as a named failure rather than widening without a bound.

Protocol-only constraints should establish a measured low-overhead baseline. Syntax and semantic
qualification records should report tokens/second and energy against the same prompt, sampler, model,
and resident/streamed graph. No isolated DeepSeek, Q4-only, or output-head fast path should fork the
shared FlashMoe scheduling architecture.

## Test strategy

Most correctness evidence belongs in the new crate and must run without a model.

### Protocol and tokenizer tests

- Replay every byte boundary and every real tokenizer token boundary for complete and incomplete
  Qwen JSON and DeepSeek DSML calls.
- Split JSON escapes, four-digit Unicode escapes, UTF-8 scalars, DSML attributes, special tokens,
  delimiters, and escaped closing tags at every possible point.
- Assert that every accepted streaming transcript is accepted by the final parser with identical
  canonical calls.
- Assert that malformed names, parameters, types, duplicates, trailing text, illegal escapes,
  repeated structural whitespace, payload-limit stops, EOS, cancellation, and token exhaustion fail
  with stable terminal states.
- Keep a real Qwen/GLM tokenizer corpus and the pinned DeepSeek JoyAI tokenizer corpus. A valid token
  below the normal candidate frontier must remain selectable.

### Mutation and patch tests

- Compare streaming `write_file` output with batch-decoded content for every chunking.
- Compare `PatchStream`, batch `PreparedPatch`, and executor replay results byte-for-byte.
- Fuzz hunk counts, offsets, ordering, overlap, context, deletions, EOF markers, paths, creation,
  deletion, and multi-file boundaries.
- Prove that out-of-range offsets, stale context, unauthorized paths, snapshot drift, unsupported
  patch features, and partial final hunks never yield a receipt.
- Preserve the current Git-compatible behavior as a differential compatibility corpus without
  allowing Git's `--recount` behavior into the canonical constrained dialect.

### Language tests

- Maintain positive, negative, incomplete-but-repairable, and impossible-prefix corpora for all six
  language families.
- Assert that every completed accepted result passes the same pinned batch parser.
- Exercise Python indentation, Rust raw strings/macros/attributes, TypeScript JSX and ambient types,
  JavaScript modules/templates/regex literals, HTML raw-text elements and embedded ranges, and CSS
  escapes/custom properties/nesting.
- Test unchanged base prefixes, generated replacements, deletions, and additions through the same
  analyzer interface.
- Add semantic boundary fixtures such as a repairable string/integer mismatch followed by a rejected
  statement close, unknown Python values, forward declarations, imports, and multi-file symbol
  changes.

### State and backend tests

- Inject cancellation and errors between probe, sample, commit, analyzer update, model forward, and
  receipt publication; all state machines must remain aligned or the request must abort.
- Drive a fake LSP through stale document versions, partial pull results, delayed/missing push
  publications, restart, cancellation, response limits, and configuration changes; none may be
  mistaken for a clean semantic result.
- Run pinned rust-analyzer/Rust, TypeScript Language Service, and Pyright qualification suites
  against immutable shadow workspaces with offline dependency caches. Include dependency changes,
  lockfile/config invalidation, missing stubs/sources, and intentionally hostile Rust build scripts
  or proc macros to prove the sandbox boundary.
- Verify deterministic replay with greedy sampling and stable sampler-state restoration for any
  backend that advertises rollback.
- For DeepSeek snapshot-and-restore, require complete hidden/logit parity across nonzero prefixes and
  semantic-boundary rollback before enabling speculation.
- For llama.cpp, compare manual full-vocabulary mask/bias/apply/accept sampling with the existing
  sampler chain when the collar admits every token, and prove stateful samplers see one acceptance.
- Run the narrow FlashMoe release smoke after backend integration, plus focused Qwen and DeepSeek
  `write_file`/`apply_patch` harness cases whose output is independently parsed from disk. Add the
  same cases for every qualified llama.cpp model/template profile.

Property tests and fuzz targets should use deterministic seeds in ordinary CI, with longer corpora in
scheduled or explicit qualification runs. Crashes, panics, timeouts, and allocation-limit failures
are constraint failures, never permission to execute the output.

## Telemetry and privacy

Invocation evidence may record:

- collar, dialect, patch, and language-profile versions;
- schema, manifest, transcript, base, and result digests;
- constraint mode, promoted guarantee rung, backend recovery capability, and provider capability;
- analyzer/toolchain version, semantic-world/config/dependency digests, baseline completeness, and
  content-free unknown-reason/error-class counts;
- counts of protocol and semantic candidate rejections;
- boundary probes, rollbacks/replays, affected files, hunks, and generated/known bytes;
- terminal state and analyzer result class; and
- preflight, mask, analyzer, replay, and total decode timing.

It must not record source bytes, patch bodies, argument values, logical paths, symbol names, analyzer
messages containing source excerpts, prompts, or model logits. Detailed local harness artifacts may
retain explicitly requested test data only inside the harness scratch contract.

## Delivery phases

Each phase is independently reviewable and leaves the existing executor checks in place.

### Phase 0 — Lock contracts and baselines

Implementation status: **complete.** Existing Qwen, LLGuidance JSON, DSML, mutation, and patch
fixtures were retained and extended with collar-specific closure, truncation, and stale-base cases.

- Add deterministic fixtures for current Qwen native constraints, strict LLGuidance JSON, DeepSeek
  DSML parsing, `write_file`, and Git-backed `apply_patch` behavior.
- Record tokenizer identities, supported schemas, terminal states, low-ranked-token behavior, decode
  overhead, and current malformed/truncated-call handling.
- Finalize versioned manifest, event, mask, analysis, recovery, and receipt types in tests before
  moving production code.

Gate: the corpus explains every current acceptance/rejection outcome and runs without loading a
model where model behavior is not under test.

### Phase 1 — Create the workspace and pure crate

Implementation status: **complete.** The root and `pb-control-collar` are the two explicit workspace
and default members; the internal crate forbids unsafe code and owns no I/O authority.

- Add the non-virtual workspace with explicit members/default members and resolver 3.
- Add `pb-control-collar` with vocabulary, mask, protocol-neutral event, limits, digest, diagnostic,
  and receipt types.
- Move no live behavior initially. Add build/format/Clippy/test coverage for every workspace member.
- Then move the existing tokenizer-scoped LLGuidance JSON wrapper behind parity tests without
  changing sampling output.

Gate: existing JSON artifacts and native tool behavior are byte/token identical; all repository-wide
checks and the release smoke pass.

### Phase 2 — Unify Qwen wire constraints

Implementation status: **complete for the production boundary.** The proven Qwen envelope/schema
adapter remains at the FlashMoe sampling edge, while LLGuidance state, mutation closure, virtual
results, limits, rejection codes, and future analyzer contracts are collar-owned. This avoids a wire-
language expansion while removing mutation semantics from the backend adapter.

- Replace the handwritten Qwen-only sampling surface with the collar dialect/session API.
- Compile exposed tool schemas and canonical argument order during request preflight.
- Preserve full-vocabulary widening, payload-limit classification, terminal semantic stop, and
  executor validation.
- Replay the streaming transcript through the final collar parser and compare canonical calls.

Gate: the existing Qwen constraint corpus passes with no broader accepted language, and protocol-only
decode overhead is measured and accepted.

### Phase 3 — Virtual `write_file` and final syntax closure

Implementation status: **complete.** The initial gate is deliberately at payload/tool closure. It
guarantees valid supported completed files without claiming the stronger impossible-prefix behavior
reserved for Phase 6.

- Add controller-owned workspace snapshots and language-profile resolution.
- Stream decoded content through the virtual file transaction.
- Initially constrain candidate argument/call closure and EOS so an invalid complete supported file
  cannot execute.
- Qualify Rust, Python, TypeScript, JavaScript, HTML, and CSS complete-file gates.
- Revalidate the prepared bytes and base digest in the executor.

Gate: across unit, property, tokenizer, and harness corpora, every executed supported `write_file`
result passes its pinned parser and every capped/incomplete result leaves the workspace untouched.

### Phase 4 — Canonical streaming `apply_patch`

Implementation status: **complete for the initial closure gate.** `PatchStream` is chunk-boundary
independent and fail-closed, then prepares one exact in-memory transaction at closure. It does not
yet claim early hunk-boundary viability or checkpointing; those optimizations can be added behind the
same stream/executor contract without changing accepted patch semantics.

- Implement the exact patch subset, virtual base streaming, context/deletion verification, hunk
  accounting, and multi-file syntax closure.
- Use the same `PreparedPatch` for generation replay and executor verification.
- Preserve the unconstrained Git-compatible path as an explicitly separate compatibility mode.

Gate: patch fuzzing and differential tests find no streaming/batch/executor divergence; stale offsets,
context, hashes, and unsupported features fail before mutation.

### Phase 5 — DeepSeek DSML constraints

Implementation status: **complete and production-qualified.** DSML mutation
payloads use JSON strings inside `string="false"` parameters so their closure is unambiguous even
when source contains DSML-looking text. Candidate probing preserves special-token identity and
widens before sampling truncation.

- Implement DSML as a collar dialect over token identities and incremental bytes.
- Constrain tool names, ordered parameters, schemas, raw strings, nested JSON, terminal close, and
  truncation using the shared mutation/analyzer stack.
- Apply masks and biases over DeepSeek full logits before top-k.
- Start with `CandidateProbeOnly`; do not add speculative rollback to the production contract.

Gate: parser/constraint equivalence, real-tokenizer split tests, strict truncation containment, syntax-
valid `write_file` and `apply_patch` harness cases, repository-wide checks, release build, and the
required FlashMoe smoke all pass on the pinned profile.

### Production evidence ledger

Production promotion requires every required row below. Each recorded pass covers the exact current
workspace, not an earlier phase branch.

| Evidence | Status | Current result |
| --- | --- | --- |
| Collar unit and chunk-boundary corpus | Passed | 52 deterministic tests, including all six language families plus TSX/JSX, byte/random chunk equivalence, monotonic hard rejection, constant-time deep checkpoints, branch rollback caches, partial patch-line probing, embedded HTML languages, virtual write replacement, exact multi-hunk/create/delete patches, malformed patch corpus, Qwen/DSML payload closure, immutable semantic identities/debt and receipts, and control-token identity |
| Qwen native constraint corpus | Passed | 13 focused tests, including closure rejection for invalid syntax and the one-mutation batch bound |
| DeepSeek DSML renderer/parser corpus | Passed | 3 focused root tests plus 2 collar DSML tests for typed parameters, ordered JSON-string mutation history, and closure boundaries |
| Workspace format/check/Clippy | Passed | `cargo fmt --all -- --check`, `cargo check --workspace --all-targets -j 1`, and the repository warning/correctness Clippy gate |
| Executor and event focused tests | Passed | Snapshot freshness, exact patch/Git differential result, inexact hunk rejection, and additive/backward-compatible event round trip |
| Workspace all-target tests | Passed | On 2026-07-27, 1,465 root tests passed with 23 device/environment tests ignored, 2 environment-contract tests passed, and all 52 collar tests passed |
| Web and documentation tests | Passed | 76 web tests passed; mdBook and link validation checked 59 pages and 98 rendered files |
| Production asset/release build | Passed | On 2026-07-27, web assets and the optimized macOS arm64 release binary rebuilt successfully |
| Required FlashMoe one-token smoke | Passed | On 2026-07-27, the current release binary exited zero and printed the existing raw Qwen baseline `5` for `2+2=` |
| Phase 6–8 foundation integration | Passed | Conservative prefix and patch-stream tests, Qwen and DeepSeek payload-close semantic rejection, exact monotonic fake-LSP overlays and diagnostic debt, llama.cpp full-vocabulary pre-top-k masking, shared event compatibility, workspace check, and strict Clippy all pass |
| Phase 6 Qwen/DeepSeek tokenizer-prefix qualification | Passed | Corpus SHA-256 `1b4568f8…9215`; Qwen tokenizer `be756060…5506` ran 173 token-prefix/rollback probes and DeepSeek JoyAI tokenizer `263ab7b3…f3ba` ran 180; each replayed 704 deterministic chunkings and measured 1 µs p95 against the 1,000 µs budget |
| Phase 6 remaining promotion evidence | Pending | Scheduled long fuzz/property runs, outer Qwen/DSML logical-prefix extraction profiling, and refreshed pinned end-to-end write/patch throughput still gate promotion beyond the implemented conservative rules |
| Phase 7 shadow/evidence foundation | Passed | Semantic analysis uses a fresh exact bounded shadow tree and isolated LSP session for each transaction, rejects symlinks and post-copy drift, permits only an LSP-specific read-only analysis-root mount, and emits independently validated content-free generation and final-executor receipts |
| Phase 7 pinned Rust provider attempt | Failed safely | `ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:07b26526…173d` (rust-analyzer 1.96.0) never produced a non-empty crate graph or document membership within the required barrier; the qualifier returned `Unknown`/provider unavailable, required mode rejected, and the profile remained disabled |
| Phase 7 pinned semantic profiles | Pending | A provider that passes the checked-in Rust corpus, plus digest-pinned TypeScript/Pyright project matrices, compiler/profile parity, multi-file/dependant scope, cancellation/latency, and offline dependency resolution still gate any default semantic guarantee |
| Phase 8 live llama.cpp profiles | Pending | Pinned model/tokenizer/template write/patch/malformed/truncation and FlashMoe differential runs still gate a backend-parity claim |
| Pinned DeepSeek direct mutation qualification | Passed | The checked-in `fixtures/control-collar/` inputs produced syntax-valid `answer.py` after 2 candidate rejections and an exact snapshot-bound one-line patch after 8, reporting 1 file and 16 snapshot bytes; an alternate capped patch attempt reported 7 `invalid_patch` closure rejections and executed no call |
| Pinned DeepSeek strict workflow | Passed | An 11-invocation delivery crossed tool-schema narrowing, reused unchanged exact roots, cold-started changed roots, executed constrained `write_file`, passed review, and ended `contract_status=satisfied`, `verified_completed=true` |
| DeepSeek candidate-probe budget | Passed | On the same patch prompt and pinned checkpoint, current-release constrained decode sustained 7.678 tokens/s versus 9.358 tokens/s unconstrained, a 18.0% reduction within the accepted 25% qualification ceiling |

### Phase 6 — Stronger prefix syntax constraints

- **6A — Freeze the prefix contract.** Add versioned rule provenance, capability, boundary, verdict,
  obligation, checkpoint, and unknown-reason types. Distinguish exact grammar-backed extendability
  from conservative local rules in receipts and telemetry. Add replay fixtures before changing masks.
- **6B — Add the cheap local oracle.** Incrementally decode candidate logical bytes and track UTF-8,
  strings/raw strings/templates, escapes, comments, delimiter stacks, Python indentation/newline
  state, HTML tag/raw-text state, and CSS/JS nesting. Probe from cheap copy-on-write checkpoints and
  hard-reject only impossible transitions such as an invalid closer, impossible escape, illegal
  indentation transition, or final-byte sequence that cannot be completed within the remaining
  argument budget.
- **6C — Cache syntax trees and discover boundaries.** Apply exact Tree-sitter edits and reuse the
  previous tree for the virtual document. Use changed ranges and language-specific queries to find
  expression, statement, item, function, embedded-language, and file boundaries. Parser error nodes
  create repair obligations; they do not prove impossibility unless a separate promoted rule does.
- **6D — Make patch streaming real.** Replace closure-only buffering with a line/chunk state machine
  that commits canonical headers and hunk lines as soon as complete, verifies counts/context/base
  cursors immediately, and emits `Known`/`Generated` source events into the same analyzer used by
  writes. Checkpoint at file and hunk boundaries. A provisional syntax error cannot reject a hunk
  close when an ordered later hunk could still repair it.
- **6E — Promote languages independently.** Start with Rust and TypeScript/JavaScript delimiter and
  lexical rules, then Python indentation, then HTML/CSS and embedded-language transitions. Add an
  exact LLGuidance-compatible source-grammar subset only where grammar/final-parser differential
  tests justify the stronger extendability claim; do not block broader valid language syntax merely
  to make the grammar easier.

Gate: every valid real-tokenizer corpus program remains reachable under every token split; every
hard-rejected rule has a positive proof fixture and repairable counterexample; streaming and batch
mutation results are identical; random chunking and rollback are deterministic; no supported
completed result bypasses the shipped final syntax gate; p95 cheap-probe CPU time is below 1 ms; and
the pinned end-to-end decode-throughput reduction remains within the existing 25% ceiling.

### Phase 7 — Semantic steering and final semantic gates

- **7A — Build the semantic provider contract and shadow world.** The controller-owned broker,
  immutable bounded shadow workspace, isolated provider session, exact overlay versions, baseline
  diagnostics, provider health, and cancellation are implemented. A content-addressed world cache
  and request queue remain optional performance work after correctness qualification. Analyzer I/O
  and Tokio stay out of `pb-control-collar`.
- **7B — Establish diagnostic-delta semantics.** Initially enforce only a clean, complete baseline
  for the affected project targets. Then add a diagnostic-debt ledger that allows a mutation to fix
  existing errors while rejecting new definite errors, using in-memory code/range/source mapping and
  persisting only content-free hashes and counts. An incomplete baseline can steer softly but cannot
  establish a semantic guarantee.
- **7C — Promote Rust first.** Pin rust-analyzer, Rust toolchain, target, Cargo metadata/lockfile,
  feature/`cfg`, proc-macro, and build-script policy. Use native rust-analyzer diagnostics such as
  type mismatch, unresolved field/method/item, privacy, mutability, and ownership classes for bounded
  boundary probes. Run a final compiler-compatible check in a sandboxed shadow workspace for the
  stronger profile. Because rust-analyzer flycheck, Cargo build scripts, and proc macros can execute
  code, they require explicit no-network/read-only-workspace/ephemeral-output authority; disabled or
  unavailable expansion is `Unknown`, not success.
- **7D — Promote TypeScript, then configured JavaScript.** Prefer a pinned TypeScript Language
  Service sidecar with `ScriptSnapshot`/versioned virtual files, exact `tsconfig` options, module
  resolution, JSX mode, ambient declarations, and project references. Use syntactic and semantic
  diagnostic APIs for boundary and final checks. JavaScript receives a type guarantee only when
  `checkJs`, JSDoc, or another explicit project profile makes the relevant values known; otherwise
  dynamic results remain `Unknown`.
- **7E — Promote Python under an explicit profile.** Pin Pyright plus Python version/platform,
  execution environment, strictness, typeshed/stub paths, and project configuration. Start with
  annotated or inferred-known code under standard/strict diagnostics. `Any`, `Unknown`, dynamic
  imports, monkey-patching, descriptors, and runtime-only dispatch cannot support a hard type claim.
- **7F — Resolve dependency APIs and steer names.** First let each pinned provider resolve offline
  dependency sources/declarations/stubs directly from read-only caches. Later materialize portable
  public-symbol/type fact packs keyed by provider version, target, configuration, and lockfile so
  common completions do not require repeated process queries. Bias tokenizer sequences for visible
  symbols, qualified names, imports, fields, and methods; never hard-exclude an identifier merely
  because it is absent from a completion list.
- **7G — Revalidate at execution.** Reconstruct the exact prepared virtual workspace, verify live
  base and semantic-world identities, rerun the authoritative final provider, and only then publish
  atomically. A generation-time receipt never substitutes for final executor validation.
- **7H — Make semantic evidence durable and auditable.** Separate generation-boundary and
  final-executor receipts are implemented with contract/provider/world/config/dependency/document
  digests, versions, completeness, verdict classes, reason/count fields, budgets, and timings.
  Receipts identify their scope (document, affected target set, or complete project) and exclude
  paths, source, patches, diagnostic messages, prompts, and symbol names. Generation telemetry
  cannot be promoted into final publication evidence.

Gate for each language/profile: a pinned analyzer/toolchain and project-config digest; a complete
baseline policy; explicit definite/repairable/unknown diagnostic classes; string-plus-integer,
wrong-call, missing-field/method, import, forward-declaration, shadowing, macro/dynamic, ambient-type,
and multi-file fixtures; no new definite semantic errors across write/replace/edit/patch; no false
hard rejection in the valid/repairable corpus; offline dependency resolution; bounded cancellation
and process failure; final executor parity; and separately accepted boundary latency plus end-to-end
throughput budgets. Rust, TypeScript/JavaScript, and Python ship independently. HTML and CSS retain
their syntax claim until a separately scoped semantic provider is justified.

### Phase 8 — llama.cpp control-collar parity

- **8A — Add a native vocabulary adapter.** Build the collar vocabulary from exact llama.cpp token
  bytes, marker-only special-token identities, and every end-of-generation token. Bind dialect and
  chat-template qualification to the model/tokenizer/template identity rather than assuming every
  llama.cpp model speaks Qwen JSON or DeepSeek DSML.
- **8B — Put the collar before truncation.** For constrained tool generation, obtain the full
  `LlamaTokenDataArray`, set denied logits to negative infinity, apply sparse biases, perform bounded
  semantic frontier probing/widening, then run top-k/temperature/distribution sampling. Refactor away
  from `LlamaSampler::sample`, whose combined sample-and-accept operation is too late for a dynamic
  external mask; commit the collar and accept the selected token into stateful llama samplers exactly
  once before decoding it into the context.
- **8C — Share the production mutation path.** Construct the same manifest/snapshots, reuse Qwen and
  DSML dialect implementations, stop semantically on complete calls, and execute prepared writes and
  canonical patches through the same revalidation path. Once a model/template profile is qualified,
  remove its access to the broader `--recount` compatibility behavior and never fall back after a
  collar rejection.
- **8D — Differentially qualify profiles.** Replay identical vocabulary surfaces and transcripts
  through FlashMoe and llama.cpp adapters, compare hard masks, biases, events, receipts, parser
  results, and executor decisions, then run pinned live write/patch and malformed/truncation cases.

Gate: masks are applied over the full vocabulary before sampling truncation; sampler RNG/penalty/
grammar state is accepted exactly once; every qualified dialect passes tokenizer-split and replay
parity; mutation and semantic guarantees match FlashMoe; session-prefix reuse remains exact; decode
overhead is measured against the same model/prompt; and unsupported model/template profiles fail
preflight or retain an explicitly reported unconstrained compatibility mode.

### Phase 9 — Optional speculative recovery

Semantic correctness ships first with blocking candidate-before-commit probes. Speculation is a
latency optimization for cases where an external analyzer is slower than decode or where repair
requires changing earlier committed tokens.

- Add one bounded in-flight speculative branch from the last structural boundary. Snapshot collar,
  grammar, mutation, analyzer-overlay version, sampler, generated-token, and decode state together.
- For DeepSeek, add ephemeral complete-state snapshots with a separate memory budget and eviction
  policy from cross-turn session checkpoints. For llama.cpp, qualify deterministic replay from a KV
  boundary first; add in-memory state copies only if wrapper/upstream support proves cheaper and exact.
- If the provider accepts, publish the speculative branch into the main session. If it rejects,
  atomically restore every layer, hard-mask the rejected boundary/branch decision, and resample. A
  timeout under a required profile discards the branch and fails closed.
- Bound branch depth, speculative tokens, memory, analyzer requests, replay work, and cancellation
  time. Never persist speculative source or symbol content in events.

Gate: restored logits and sampler outputs are bit-for-bit equal under deterministic settings;
accepted and rejected branches leave identical collar/mutation/LSP version state to non-speculative
execution; memory and latency bounds survive cancellation and analyzer death; and disabling
speculation changes performance only, never the accepted output language or executor guarantee.

## Owner decisions

The repository and technical research are sufficient to begin Phase 0 and the behavior-preserving
workspace work in Phase 1. The project owner accepted the recommended defaults below on 2026-07-26;
they are now requirements for their dependent phases.

### Production milestone

**Decided:** the first production milestone is Phases 0 through 5: the workspace/core extraction,
Qwen constraints, virtual `write_file`, canonical `apply_patch`, the six pinned syntax profiles, and
DeepSeek DSML parity. Stronger impossible-prefix checks, type-aware steering, speculative recovery,
and llama.cpp parity are separately qualified follow-ons.

Requiring Phase 7 semantic analysis before the first production release would substantially enlarge
the soundness, dependency, project-overlay, performance, and language-configuration contract. It
would also delay the already valuable completed-file syntax and patch-validity guarantee.

### Patch and batch compatibility

**Decided:** constrained generation uses the documented canonical text-only patch subset and
initially permits at most one mutation call per generated batch. The existing Git-compatible
parser/executor path remains available for explicitly unconstrained backends, but pb never falls back
to it after a constrained patch is rejected.

Supporting the complete Git patch language or ordered atomic multi-mutation batches in the first
release requires materially more path, mode, rename/copy, rollback, and executor-equivalence work.

### Enablement and crate status

**Decided:** `pb-control-collar` remains unpublished and workspace-internal. Each qualified syntax
profile is automatically enabled on a qualified FlashMoe dialect/backend. Preflight fails closed when
a request promises an unavailable required constraint. Experiments use explicit harness arguments;
the initial release adds no user-visible environment or configuration toggle.

Publishing the crate or making the collar opt-in would create additional API stability, support,
configuration, migration, and documentation commitments that are not needed to prove the production
behavior inside pb.

## Open decisions and required probes

Phases 0–5 resolved their acceptance-affecting choices below. Later-phase choices remain open until
their named phase produces evidence:

| Decision | Latest phase | Evidence required |
| --- | --- | --- |
| Tree-sitter versions | Resolved in Phase 3 | `tree-sitter` 0.25.10; Rust 0.24.2; Python 0.25.0; TypeScript/TSX 0.23.2; JavaScript 0.25.0; HTML 0.23.2; CSS 0.25.0, all exact-pinned |
| Additional final parser beyond Tree-sitter | Resolved for Phase 3 | Error/missing nodes and UTF-8 are rejected; HTML adds explicit element closure plus supported embedded-language parsing. Compiler/type/project validity is explicitly outside the claim. |
| Ordered atomic multi-mutation batches versus one mutation per generated batch | Phase 4 | Executor rollback design and virtual/executor differential tests |
| Canonical patch path and EOF syntax | Resolved in Phase 4 | LF text, unquoted exact `a/`/`b/` paths, `/dev/null`, exact hunk ranges and context, Git's no-final-newline marker, optional matching `diff --git`, and only exact `100644` create/delete metadata |
| DeepSeek candidate-probe latency budget | Resolved in Phase 5 | The default production ceiling is at most 25% decode-throughput loss on the same prompt/checkpoint; the pinned patch qualification measured 18.0% |
| Prefix guarantee scope | Phase 6 | Per-rule extension-property corpus, valid-program tokenizer splits, grammar/final-parser differential results, and named conservative versus exact capability receipts |
| Semantic baseline policy and rollout mode | Phase 7 | Clean-world and diagnostic-debt corpora, incomplete-baseline behavior, typed project configuration, and user-visible guarantee wording |
| Rust native-analysis versus compiler-check boundary | Phase 7 | Diagnostic-class parity, macro/build-script sandbox evidence, target/feature/`cfg` matrix, and final executor cost |
| Dependency public-symbol fact packs | Phase 7 | Provider/direct-resolution parity, lockfile/config invalidation, cache size/privacy, and completion steering benefit |
| llama.cpp model/template qualification | Phase 8 | Token/special/EOG identity, dialect rendering, full-vocabulary mask ordering, stateful-sampler acceptance, and live mutation corpus |
| Whether DeepSeek ephemeral complete-state snapshots beat deterministic replay | Phase 9 | Snapshot copy cost, memory peak, restore parity, branch depth, and end-to-end energy |
| Rust semantic provider boundary | Phase 7 | rust-analyzer/compiler parity corpus including macros, traits, `cfg`, and workspace edits |
| TypeScript/JavaScript project overlay provider | Phase 7 | Module-resolution, ambient-type, JSX, and multi-file corpus |
| Python strict profile and `Unknown` policy | Phase 7 | Annotated/unannotated corpus, false-rejection audit, and pinned analyzer configuration |

Unsupported language profiles remain protocol-constrained and executor-validated; they must not be
reported as syntax- or semantic-constrained. A supported profile whose required analyzer cannot be
prepared fails preflight rather than downgrading silently.

## Documentation and release obligations

This document remains an engineering record. As phases ship:

- update `docs/architecture/workflows.md` for generation, tool, executor, and terminal behavior;
- update `docs/architecture/user-contracts.md` for visible guarantees and failure states;
- update `docs/architecture/security.md` for snapshot, capability, parser, and executor trust
  boundaries;
- update `docs/architecture/local-privacy.md` for analyzer processes, source overlays, persistence,
  and telemetry;
- update `docs/flashmoe-architecture-parity-plan.md` whenever FlashMoe sampling, model-family behavior,
  output-head masking, or DeepSeek state handling changes;
- update the matching user chapters only when behavior is actually shipped or configurable; and
- update `src/init.rs` with any future project configuration rather than adding environment flags.

The implementation qualification sequence includes `deno task build:web` before release builds,
workspace-wide formatting/Clippy/tests, web and documentation tests, the macOS arm64 release build,
and the required one-token FlashMoe smoke after backend changes.

## Completion criteria

The project is complete only when:

1. Qwen/GLM and DeepSeek use one shared mutation/analyzer stack through separate tested dialects.
2. Every completed supported `write_file` and constrained `apply_patch` execution is bound to an
   unchanged authorized snapshot and a valid exact result.
3. Streaming parsing, batch replay, and executor verification are differential-test equivalent.
4. Invalid or incomplete syntax, patches, envelopes, analyzer failures, cancellation, and token caps
   cannot produce a mutation.
5. Low-ranked valid tokens remain reachable and measured decode overhead stays within an explicitly
   accepted qualification budget.
6. Rust, Python, TypeScript, JavaScript, HTML, and CSS each have pinned profiles and real-tokenizer
   positive, negative, repairable, and patch corpora.
7. Any semantic guarantee identifies its analyzer/configuration boundary and does not present unknown
   dynamic behavior as proven safe.
8. Backend rollback, where enabled, restores model, sampler, collar, mutation, and analyzer state
   atomically and exactly.
9. Durable telemetry contains no source, patch, argument, path, prompt, or symbol content.
10. Curated architecture, security, privacy, user, FlashMoe, harness, and release documentation match
    the behavior actually enabled in production.

## Primary references

- [Cargo workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [LLGuidance repository](https://github.com/guidance-ai/llguidance)
- [LLGuidance grammar syntax](https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md)
- [Tree-sitter advanced parsing](https://tree-sitter.github.io/tree-sitter/using-parsers/3-advanced-parsing.html)
- [Tree-sitter syntax nodes](https://tree-sitter.github.io/tree-sitter/using-parsers/queries/1-syntax.html)
- [Language Server Protocol 3.18](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.18/specification/)
- [rust-analyzer diagnostics](https://rust-analyzer.github.io/book/diagnostics.html)
- [rust-analyzer configuration](https://rust-analyzer.github.io/book/configuration)
- [TypeScript Compiler and Language Service APIs](https://github.com/microsoft/TypeScript/wiki/Using-the-Compiler-API)
- [Pyright configuration and diagnostic profiles](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)
- [llama.cpp sampling and grammar surface](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md)
- [`git apply`](https://git-scm.com/docs/git-apply)
