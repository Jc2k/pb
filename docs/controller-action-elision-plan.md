# Controller-owned deterministic action elision

Status: **Default-off production implementation complete; truthful controller rendering is the only
production representation, while non-off defaults and transcript renderings remain unpromoted**

This follow-on explores whether pb can save local-model invocations by executing uniquely determined
observations and bookkeeping in the controller. It extends the
[verified task-completion reliability plan](task-completion-reliability-plan.md) and its typed
work-unit controller. The mechanism is available to normal daemon, desktop, and web workflows
through a typed, default-off user policy. Prompt-transcript research remains confined to the hidden
harness, and a non-off default is not a shipped guarantee.

## Implementation status

| Step | Current state |
| --- | --- |
| E0 | **Configurable:** hidden harness rendering enum, typed receipts, truthful durable provenance, and prompt-only compatibility call IDs. |
| E1 | **Qualified representation screen:** the [initial review screen](benchmarks/controller-action-elision-e1.md) and [byte-locked two-tier read qualification](benchmarks/controller-action-elision-e2.md) support the truthful controller block and provide no evidence for transcript-shaped production rendering. |
| E2 | **Configurable:** controller observations carry validated operation, coverage, fingerprints, ranges, prompt bytes, and authority effects alongside ordinary stage evidence. |
| E3 | **Configurable:** candidate prompts use the active model's renderer/tokenizer and admit observations only below 55% of usable prompt capacity without compacting their bytes. |
| E4 | **Configurable:** exact active small UTF-8 files can be fully observed and seed read-before-write evidence; ineligible inputs fall back to native reads. |
| E5 | **Configurable:** exact failed-diagnostic anchors can produce hash-bound ranges and range-confined edits; replacement and unobserved edits remain blocked. |
| E6 | **Configurable:** fresh review may receive all required `inspect_change` results at once, without controller-authored assessments or verdicts. |
| E7 | **Configurable:** a successful final mutation may carry model-authored completion fields; structurally empty mutation-forbidden work can close as controller-owned no-change. |
| E8 | **Configurable, separately default off:** narrowly eligible tracked clean deletions retain controller origin and Git recovery and require `safe` plus an explicit user opt-in. |
| E9 | **Implemented, default off:** typed user policy is server-owned, skipped in request persistence, reapplied to restored tasks, and has a one-step `off` rollback. |
| E10 | **Implemented:** action IDs are content-derived, prompt admission revalidates workspace/path identity, and ordinary read eligibility failures fall back to native tools. |
| E11 | **Implemented and screened on 4B/7B:** the hidden four-arm continuation command preserves fixture/configuration, source/model identity, semantic and rendered prompt digests, events, and artifacts and rejects non-locked controller arms. The broader promotion matrix remains open. |
| E12 | **Default-off production gate passed; default promotion open:** typed policy, rollback, provenance, fallback, and scripted safety checks ship without promoting a non-off default or transcript representation. |

## Outcome

pb should spend model invocations only on choices or judgments that require model intelligence. A
controller may elide an action when its result is locally computable, uniquely determined, bounded,
non-authorizing, and independently auditable. The optimization must preserve verified completion,
fresh evidence, allowed-path restrictions, review independence, managed commit ownership, and local
privacy.

One question remains deliberately empirical: a tool-trained local model may behave better when a
controller-executed read is rendered in the familiar assistant-tool-call/tool-result shape than when
the same bytes appear in a prose controller block. pb measures that possibility, including the
undisclosed compatibility arm, but does not make safe production rollout depend on it. The truthful
controller block is the sole production candidate. A transcript rendering remains harness-only
unless the locked experiment independently clears its stricter provenance and behavior threshold.

## Production closure decision

Production behavior is owned by the local pb controller, not by request payloads or model output.
It uses a typed user setting rather than a hidden environment toggle:

```toml
[agent]
action_elision = "off"          # off | review_only | safe
controller_delete_elision = false
```

The rollout contract is:

- `off` preserves native model tool selection and is the immediate rollback;
- `review_only` permits only fresh, exact controller-owned review observations;
- `safe` additionally permits eligible reads, range observations, and structural closure fusion;
- deletion requires `safe` plus the separate explicit boolean, and defaults to disabled; and
- normal production always renders an explicit controller block. Transcript choices remain
  accepted only on the hidden harness.

The first release defaults to `off`. Promotion to a non-off default is a separate evidence-backed
change: `review_only` requires the repeated review matrix, while `safe` also requires the locked read,
range, stale-state, and closure matrices. This makes the implementation deployable and recoverable
without silently converting incomplete qualification into a default guarantee.

## Truth boundary

Controller execution and prompt representation are separate facts.

- The durable event stream records the actual actor: model, controller, deterministic workflow, or
  user.
- A controller action must execute successfully before any result is rendered to the model.
- A prompt-only assistant tool call never becomes a model `ToolCall` event, model-authored intent,
  progress credit, approval, review judgment, or check receipt.
- Every injected result remains bound to its current path and workspace fingerprints.
- A bounded excerpt never claims that the model observed the complete file.
- Prompt compaction cannot convert a controller action into model provenance.
- No compatibility transcript may represent mutation, approval, publication, a passing check, or a
  review verdict.

The implementation must distinguish:

1. **executor evidence** — pb observed enough exact state to enforce concurrency and mutation
   safety; and
2. **model observation coverage** — the exact full content, ranges, metadata, or failure actually
   included in the model prompt.

## Existing bounds

The current controller reads at most 8 MiB through `read_file` and the existing text mutation tools.
A carried complete-file evidence entry is limited to 16 KiB, with 24 entries and 64 KiB total per
bundle. A rendered tool result is limited to 16,000 prompt characters and may be compacted when the
measured prompt crosses its soft context limit. Action elision must respect or tighten these bounds;
it must not silently increase them.

## Eligibility classes

| Class | Examples | Initial policy |
| --- | --- | --- |
| Controller-observable | exact active-path read, metadata, bounded search result, changed-path inspection | eligible after the rendering experiment |
| Controller-derivable | plan digest, current fingerprint, actual touched paths, no-change fact, selected check IDs | eligible when projected as controller-owned facts |
| Controller-executable | uniquely accepted tracked-file deletion, no-change transition | experimental only; requires a later explicit gate |
| Model-required | edit content, resolve ambiguity, plan scope, substantive review, findings, semantic summary | never synthesized by the controller |
| User-required | approval, external publication, destructive expansion, subjective acceptance | never elided |

## E0 — experiment-only provenance and rendering surface

Add a typed harness-fixture field rather than an environment toggle or production setting:

```text
observation_rendering = native | controller_block | disclosed_tool_transcript | compatibility_tool_transcript
```

The field is accepted only by the hidden evaluation surface. It selects prompt representation, not
authority. All controller arms execute the same bounded read, produce the same content hash and
observation coverage, seed the same executor evidence, and differ only in prompt messages.

Every controller arm emits a durable receipt containing at least:

```text
schema_version
action_id
actual_origin = controller
prompt_representation
stage and work_unit_id
operation and normalized path
workspace/path/content fingerprints
coverage = full | ranges | metadata_only | none
observed and prompt byte counts
included ranges and their hashes
included_in_prompt
authority_effects
fallback or invalidation reason
```

Synthetic call IDs live only in the prepared prompt. They cannot collide with real call IDs, enter
the durable model transcript as authored calls, or be replayed as new evidence from the read cache.

Promotion proof:

- serialization, checkpoint, compaction, and event/journal tests preserve actual origin;
- no prompt renderer changes gate state or the result bytes;
- an interrupted or failed controller read creates no successful synthetic transcript;
- production accepts only the truthful controller block and never exposes rendering selection.

## E1 — compatibility-transcript experiment

### Hypothesis

A tool-trained local model may produce a more correct immediate continuation after a familiar
assistant-tool-call/tool-result sequence than after an equivalently informative prose block. The
benefit, if any, must be separated from the invocation saved by controller execution.

### Locked arms

1. **Native read.** The model selects and calls `read_file`; this is the behavioral and efficiency
   control.
2. **Explicit controller block.** pb executes the read and inserts a canonical labelled observation
   in the dynamic user/context suffix.
3. **Disclosed tool transcript.** A system line says that pb executed the deterministic call on the
   model's behalf, followed by a prompt-only assistant tool call and the exact tool result.
4. **Compatibility tool transcript.** The prompt contains the assistant tool call and result without
   the disclosure line; durable provenance still says controller.

Model, model digest, chat template, context size, generation cap, sampling, seed, task, contract,
workspace bytes, evidence, result truncation, and cache conditions remain locked. The three
controller arms use identical tool schemas and byte-identical results. The native arm initially
exposes the read required by its evidence state; after a successful read, its result bytes and
post-read continuation are compared with the controller arms. Its extra generation is reported
separately rather than normalized away.

Arm preparation is immutable and independently checkable. A canonical fixture root is hashed before
any run, cloned once per arm, and rejected if its manifest differs. The eligible controller
observation is prepared from those frozen bytes and its exact operation result is saved as a fixture
artifact. All controller arms consume that same byte sequence; action IDs, timestamps, check
durations, scratch paths, and other volatile values are excluded from the compared prompt suffix.
The evaluator saves the complete rendered prompt digest, observation-result digest, model/runtime
digest, fixture manifest, sampling settings, event log, and final artifact manifest for each arm.

The read experiment starts at the immediate-continuation boundary: an accepted plan and unique
active path are fixed before the compared generation. In the native arm the model must request the
read; in controller arms the exact result is already present. This avoids measuring whether the
model happened to propose the expected plan instead of measuring its response to the four
representations.

### Fixture matrix

The initial screen uses small deterministic tasks where the next read is uniquely selected:

- modify one small existing UTF-8 file;
- delete one tracked unchanged file and modify a second small file;
- repair an exact path named by a failed diagnostic;
- review one small changed file;
- receive a complete but irrelevant allowed-path read; and
- receive a read failure.

Safety fixtures then cover partial large-file content, stale evidence, binary input, more than one
plausible target, and a controller result that is invalidated before generation. These cases test
whether a compatibility transcript causes inappropriate confidence or mutation.

Run the four-arm screen once per fixture in fresh scratch roots. Advance the best two controller
representations to three locked repeats over the qualifying TC3 modify, delete/modify, diagnostic,
and review cases. Preserve every scratch root and independently audit journal, events, exact delta,
checks, review, commit, and clean status.

### Measurements

Functional and safety measures:

- correct first action after the observation;
- redundant reread rate;
- correct target and mutation-tool selection;
- malformed, repeated, or rejected actions;
- verified completion and exact artifact checks;
- unsupported claims after partial, irrelevant, stale, or failed observations;
- false model attribution, false evidence, forbidden mutation, and false completion.

Efficiency measures:

- model invocations and stage steps;
- rendered, cached-prefix, and fresh-prefill tokens;
- generated tokens;
- prompt bytes attributable to the representation;
- wall time and energy; and
- compaction and context-limit incidence.

### Selection rule

No representation advances with any false evidence, false attribution, authority expansion, or
unsafe continuation. Among safe candidates, prefer the disclosed transcript when it is within one
verified completion of the best controller arm in the screen and within five percentage points in
the repeated series. An undisclosed compatibility transcript requires a repeatable improvement of
at least ten percentage points across two independent series before it merits a separate design
decision; it is never promoted implicitly.

If neither transcript materially improves behavior over the explicit controller block, production
uses the explicit block. A model-family-specific win remains evaluation evidence until another
supported local tier confirms it or the rendering is explicitly scoped by typed model capability.

## E2 — production observation receipt

Production introduces a versioned controller observation in workflow state using the truthful block.
E1 can still justify a later, separately reviewed harness-to-production rendering change.
It records origin, current fingerprints, coverage, prompt representation, persistence, and
invalidation. The event stream and journal show the receipt independently of its model-facing
rendering.

Likely implementation boundaries:

- `src/workflow/evidence.rs` owns the typed receipt and coverage;
- `src/agent_core.rs` selects, executes, validates, and renders an eligible observation;
- `src/agent_context.rs` measures and reserves context budget;
- `src/events.rs`, harness reports, and web event types expose provenance;
- read-cache replay preserves result provenance without re-executing authority effects.

## E3 — dynamic observation budget

Before injecting content, pb measures the base prompt with the actual model tokenizer. It preserves
the generation reserve, safety margin, and compaction target, then assigns the remaining observation
budget only to the active work unit. Evidence is selected in this order:

1. complete current content;
2. exact failed-diagnostic ranges;
3. exact typed contract anchors when a future contract schema carries them;
4. metadata-only receipt; or
5. the existing model read/search surface.

Tool schemas remain canonical; all observation content stays in the dynamic suffix. If inclusion
would cross the target, pb reduces coverage or declines elision rather than shrinking the generation
reserve or dropping authoritative anchors.

## E4 — small-file read elision

For an exact accepted-plan path whose complete UTF-8 bytes fit the carried-evidence and measured
prompt budgets, pb:

1. reconciles the work-unit ledger;
2. reads and hashes the complete current bytes;
3. records a full controller observation;
4. includes those exact bytes through the E1-selected rendering;
5. seeds the same fingerprint in the write gate; and
6. begins generation with the unit mutation-ready and without redundant `read_file` exposure.

Any read failure, path-kind mismatch, context shortfall, or stale fingerprint falls back to the
existing evidence-needed state. A successful eligible task must save at least one model invocation
with identical final bytes and verification.

## E5 — localized large-file observation

Files larger than the full-observation budget use explicit coverage rather than a false complete-read
claim. pb may stream/hash the file for executor safety, but the model receives only deterministic
ranges and metadata. A range-bound edit records the file hash, byte interval, interval hash, UTF-8
boundaries, and exact old-text or patch anchor. It cannot authorize replacement outside observed
ranges.

Initial production range sources are limited to exact failed-diagnostic locations. Changed hunks are
already exact review observations rather than edit authority. Symbols and contract anchors become
eligible only when a trusted typed producer is added; pb never infers them from prose. If no exact
source exists, pb retains `ripgrep` and ranged `read_file` instead of guessing relevance.

## E6 — fresh review inspection elision

At isolated code-review entry, pb may compute `inspect_change` material before generation. When every
required changed path fits, the fresh reviewer receives all exact inspections and may submit its own
assessment immediately. If any path has only partial or metadata coverage, `inspect_change` remains
available and the review terminal remains hidden until the existing coverage contract is satisfied.

The controller never supplies assessments, findings, severity, or verdict. Deleted, renamed,
symlink, and binary paths keep their explicit manifest representations.

## E7 — implementation closure fusion

To remove a separate bookkeeping generation without fabricating model judgment, the active mutation
schema may accept optional completion material: accepted-step statuses, summary, and semantic commit
subject. If the atomic mutation succeeds and structurally completes the queue, pb combines those
model-authored fields with controller-projected plan identity, actual paths, no-change fact, and
current fingerprint. Invalid or premature completion material is ignored or rejected without
advancing the workflow.

A contracted no-change task may use a controller-owned structural receipt after plan review, but
checks and any substantive review remain unchanged.

## E8 — conservative automatic deletion

Automatic deletion is last and separately default-off. It requires a unique active delete in an
accepted and freshly reviewed plan, deliver intent, explicit contract/policy allowance, a tracked
unchanged file or symlink, exact baseline/invocation identity, and no adopted or untracked bytes.
Directories, ambiguous resolution, dirty content, untracked files, and user-owned partial work are
ineligible. Symlink deletion unlinks the link and never follows it.

Product promotion requires `safe` mode plus the separate local deletion opt-in. The setting is not
itself delete authority: all freshness, uniqueness, tracked-clean, accepted-plan, allowed-path, and
Git-recovery conditions still have to pass. A model tool call remains the fallback; the controller
does not broaden destructive authority merely to save a step.

## E9 — production policy ownership and rollback

Add the typed user settings above and copy their effective values into each server-owned agent
request. Client/session payloads and restored checkpoints cannot enable or broaden the policy.
Harness fixtures continue to select their explicit arm independently. Configuration validation,
effective-value reporting, and documentation must make these facts observable:

- absence means `off` and deletion disabled;
- an unknown mode is rejected;
- deletion is inert unless `safe` is also effective;
- changing to `off` affects newly started work without data migration; and
- no mode adds network access, telemetry, remote inference, or remote persistence.

## E10 — invalidation, fallback, and idempotence

Controller observations use content-derived action IDs rather than clocks or scratch-root names.
Immediately before prompt admission, pb revalidates the workspace, path identity, content hash, and
coverage hashes recorded by the receipt. A mismatch discards the candidate and returns to native
tool selection without progress credit.

Missing, unreadable, disallowed, oversized, binary, symlinked, or context-ineligible content is a
normal eligibility miss rather than a workflow failure. It cannot produce a successful prompt
transcript or mutation evidence. Repeated preparation over identical state yields the same receipt
identity and cannot grant duplicate authority. Cancellation and checkpoint restore repeat the same
validation before reuse.

## E11 — byte-locked evaluator and promotion evidence

Add a deterministic harness fixture for the immediate-continuation experiment and machine-readable
arm comparison. It rejects non-identical controller result bytes, prompt inputs that differ outside
the declared representation, fixture mutation before generation, or missing provenance artifacts.
It reports the native extra invocation separately and compares first continuation, verified final
artifact, safety events, tokens, latency, and energy where available.

Preserved qualification covers at least:

- small full-file modify, irrelevant full read, missing/read-failure, and context rejection;
- exact diagnostic range, large file, huge line, invalid UTF-8, stale mutation, and multiple paths;
- review of small, large, deleted, renamed, symlink, and binary changes;
- fused completion with valid, invalid, incomplete, and stale model-authored fields;
- no-change work with required and forbidden mutation contracts; and
- deletion success plus dirty, untracked, ambiguous, stale, symlink, and recovery cases.

Run scripted coverage first, then preserved real-model trials on each supported local-model tier.
Model limitations, experiment defects, and pb defects are classified separately. Promotion evidence
records exact commits, runtime/model digests, configs, fixture manifests, scratch roots, event logs,
checks, artifact diffs, and audit conclusions.

## E12 — production promotion gate

The production code may ship default-off once scripted safety, configuration, provenance, fallback,
and rollback tests pass. A non-off default requires its complete E11 matrix, zero false authority or
completion, independently identical artifacts, and a repeatable end-to-end latency or energy win.
The undisclosed transcript additionally retains E1's cross-series ten-point requirement and needs a
separate explicit trust-boundary decision.

Deletion stays default-off even if its opt-in matrix passes. Any safety regression rolls the
effective mode back to `off`; a review-only regression also disables `review_only`. Rollback does
not delete receipts or rewrite event history, so the audit trail continues to state what actually
executed.

## Edge-case matrix

| Condition | Required behavior |
| --- | --- |
| UTF-8 file at or below 16 KiB and within measured prompt budget | full observation may elide the read |
| Small file that does not fit remaining context | bounded coverage; keep targeted reads available |
| UTF-8 file from 16 KiB through 8 MiB | full executor hash, deterministic ranges only, range-bound edit |
| File above 8 MiB | metadata/streaming hash only; localized command, replan, or explicit action |
| Binary or invalid UTF-8 | metadata and hash; no automatic text mutation |
| Minified file or one huge line | UTF-8-aligned byte windows rather than line-only ranges |
| Symlink | `lstat`, never implicit follow; unlink only when deletion is explicitly eligible |
| File changed after observation | invalidate receipt and mutation evidence before execution |
| File missing or recreated | complete delete, block modify for replan, or re-observe the new identity |
| Several paths exceed total budget | active path only; never eager-load the queue |
| Permission or read failure | structured controller failure; no successful transcript or evidence |
| Prompt compaction | rebuild current observation from workflow state; retain origin and coverage |
| Sensitive or secret-declared path | no eager read unless exact typed task authority permits it |
| Untracked or adopted deletion | never automatically delete |
| Cancellation | preserve prior checkpoint; grant no observation or mutation progress |

## Qualification ladder

1. Scripted tests prove origin, coverage, budget, invalidation, fallback, terminal gates, policy
   ownership, and rollback.
2. E11 proves that E1 controller arms have byte-identical observation results and locked inputs;
   representation outcomes are preserved research evidence.
3. The truthful block qualifies one small modify and one delete/modify task 3/3 against native
   controls before `safe` can become a default.
4. E5 qualifies oversized, partial, binary, UTF-8-boundary, stale, and multi-file containment.
5. E6 qualifies fresh review across small, large, deleted, renamed, symlink, and binary paths.
6. E7 qualifies closure without missing or synthesized semantic fields.
7. E8 qualifies only an explicit default-off production opt-in after destructive-action review.
8. Run the final-source TC3 corpus and supported model tiers.
9. Independently audit source, preserved journal/events, artifacts, checks, Git recovery, latency,
   and energy before changing any default.

Every promotion requires zero false verified completions, forbidden mutations, evidence grants,
review claims, or model attribution; identical independently checked artifacts; clean semantic
commits where required; and a measured wall-time and energy improvement rather than call-count
reduction alone.

## Documentation and configuration obligations

Any shipped behavior updates the curated workflow, user-contract, security, and local-privacy
chapters in the same commit. The user setting must update configuration documentation; no new project
field is introduced. No `PB_*` environment switch is permitted. Experiment-only rendering stays on
the typed hidden harness surface until a promotion gate explicitly moves it into supported
configuration.
