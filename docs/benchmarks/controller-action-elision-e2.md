# Controller action-elision E2 production qualification

Captured: 2026-07-22

Plan: [Controller-owned deterministic action elision](../controller-action-elision-plan.md)

This qualification tests the accepted-plan read boundary with one immutable fixture and four
renderings. It closes the byte-lock defect from E1 and informs the production representation
decision. It does not promote action elision from its default-off setting.

## Locked protocol

The hidden evaluator resets the same Git workspace to one fixture commit before every arm. It
starts from an accepted plan whose unique active work unit modifies `existing.txt`, hashes the
fixture and exact model artifact, records the pb source commit and dirty-source tree fingerprint,
and preserves every semantic generation input, actual rendered-prompt digest, event, and final
artifact. It rejects the run unless the three controller arms have identical result bytes, tool
schemas, normalized non-representation inputs, and controller action IDs.

Both final tiers used eight bounded workflow steps, 8,192 context tokens, 512 maximum generated
tokens per turn, temperature 0, top-k 1, seed 0, and all available Metal layers.

| Tier | Model artifact SHA-256 | Configuration SHA-256 | Source-tree fingerprint |
| --- | --- | --- | --- |
| 4B | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` | `13cc6047c31bd05ced755b2494f96be0ee7724a0fd01cc23d0b365be83ef0727` | `90209a7b53a2bfff2dd8de132205b117c64938cd15c7e6798c9a61c18a2d1ed1` |
| 7B | `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` | `43582968c394ec9eb0bd57b4242f6d28c683378d288eb1812b2167eeebd4a90f` | `90209a7b53a2bfff2dd8de132205b117c64938cd15c7e6798c9a61c18a2d1ed1` |

The recorded source commit is `9a205314768c2d7f022199c607e9d803ce76ed74`; the source-tree fingerprint
identifies the uncommitted production-hardening implementation used to build the evaluated binary.
The exact running pb executable SHA-256 is
`c4d8698b759485b836e1ead9d079f880a0ac35772990b05c20ada612da93d369`. The fixture SHA-256 is
`fee347f44f47e3038e34613458fe2329d463ee0b60814be3477e9163eeb7bfb4`.

## Protocol result

Both tiers passed every hard lock:

- controller observation results were byte-identical;
- the native read result and each controller result had SHA-256
  `15e5637a329474efa601b719488b0bca324a72b629ba3c0aeea423d537554ba9`;
- normalized inputs outside the declared representation were identical;
- controller tool schemas and content-derived action IDs were identical; and
- actual rendered prompt digests were recorded separately for every generation.

The earlier host run exposed a pb defect: a controller full-file read omitted the strict native
workflow-fingerprint suffix. The controller renderer now uses the exact native bounded-read result
and fingerprint suffix, with a regression test. A second evaluator reporting defect preferred a
later model-authored reread over the initial controller result; the recorder now gives the initial
observation precedence. The final runs below include both corrections.

## Behavioral result

| Tier | Arm | Outcome | Model calls | Tool calls | Prompt tokens | Generated tokens | Model wall | Energy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | Native | step limit | 8 | 4 | 22,410 | 4,016 | 41,916 ms | 873.7 J |
| 4B | Controller block | step limit | 7 | 3 | 19,633 | 3,393 | 40,884 ms | 316.8 J |
| 4B | Disclosed transcript | step limit | 10 | 2 | 38,669 | 4,531 | 89,157 ms | 200.4 J |
| 4B | Compatibility transcript | step limit | 6 | 1 | 19,018 | 3,028 | 71,769 ms | 346.2 J |
| 7B | Native | step limit | 7 | 5 | 17,312 | 2,077 | 54,066 ms | 967.4 J |
| 7B | Controller block | ready | 5 | 4 | 12,437 | 1,593 | 46,088 ms | 192.8 J |
| 7B | Disclosed transcript | step limit | 6 | 5 | 17,487 | 3,072 | 72,201 ms | 101.3 J |
| 7B | Compatibility transcript | step limit | 8 | 6 | 25,197 | 2,967 | 72,732 ms | 89.7 J |

Model wall is the sum of recorded inference durations, not end-to-end elapsed time. Energy is
retained for audit but is not used to rank arms because this was one sequential run per arm.

On 4B every arm made the requested semantic change and produced the same 17-byte artifact, but all
four stopped at their bounded step limit and pb did not infer completion from their correct files.
The controller block saved one invocation. Across the last three otherwise locked runs, the
compatibility arm used six, eight, then six calls, while the disclosed arm used seven, seven, then
ten and reached `ready` only in the middle run. This trajectory variance makes the transcript arms
unsuitable for selection from sparse call-count or latency results.

On 7B, native and controller block produced the same 17-byte artifact SHA-256
`97b38b2ebda1ca4cf4ea291005d97d07c7053db2aed3ef866c04b49ecfb3448d`. The controller block saved
two invocations and about 14 seconds of recorded model time, reached `ready`, completed independent
model-authored review, and created a task-owned semantic commit. Its first model action redundantly
requested `read_file`; because that tool was intentionally not exposed after full controller
observation, pb rejected it without executing it and the model recovered.

The two 7B transcript arms retained a trailing newline, so their 18-byte artifact SHA-256 was
`dc5ed57b50f4f32321b34961f763c47c50ac952276a8b3e27404ed9af8ed308e`. Both were semantically
acceptable, but neither was byte-identical to native and both hit the step limit. The undisclosed
compatibility form was the worst arm by model calls on this tier. This is evidence against using a
fabricated-looking transcript to improve tool-trained model behavior.

## Provenance, safety, and Git audit

- Every synthetic read was recorded only as `controller_observation` with
  `actual_origin=controller`; no synthetic call became a model `tool_call` event.
- Controller read authority was bound to the accepted work unit, exact path/content/workspace
  fingerprints, full coverage, and current-state revalidation before generation.
- Models still authored every mutation, implementation submission, review assessment, and verdict.
  Controller observations granted no approval, check, review, commit, or completion authority.
- All mutations stayed within `existing.txt`. No arm touched another path.
- Failed, unavailable, or redundant model actions earned no false progress. Step and token bounds
  terminated every incomplete run predictably.
- Only the final 7B controller-block arm completed review and the managed commit gate. Every
  incomplete arm remained dirty in its isolated fixture workspace and was reported as step-limited
  rather than successful. A preceding 4B disclosed run did complete, illustrating trajectory
  variance rather than a stable transcript advantage.
- Independent SHA-256 checks of every preserved artifact agree with the evaluator summary.

## Classified observations

### P1 — Native/controller result mismatch in the first host run

- Classification: pb defect.
- Evidence: `/private/tmp/pb-action-elision-prod-4b-8k-host-20260722` omitted the workflow content
  fingerprint from the controller result.
- Impact: invalidated the intended representation-only comparison.
- Disposition: fixed in the final source and covered by exact-result regression tests.

### P2 — Initial summary selected a later reread hash

- Classification: pb defect.
- Evidence: `/private/tmp/pb-action-elision-prod-4b-8k-fix-20260722` preserved correct prompt bytes,
  but its summary preferred a later model `read_file` event over the first controller observation.
- Impact: misleading comparison reporting without changing model behavior or authority.
- Disposition: fixed; both final summaries report native/controller byte identity correctly.

### Bounded local models did not always close correct work

- Classification: model limitation and positive containment evidence.
- Evidence: all four final 4B arms and three final 7B arms produced the requested semantic file
  change but reached their step limit before all workflow gates; preceding 4B runs varied.
- Impact: no false success; additional calls can erase the expected latency win on weaker arms.
- Disposition: accepted model limit. Do not weaken completion or review gates.

### Sandboxed Metal trials could not create an inference context

- Classification: experiment error.
- Evidence: the three sandboxed roots listed below ended with `engine_error` before generation.
- Impact: they test neither representation behavior nor model quality.
- Disposition: excluded from behavior results and preserved as environment diagnostics.

## Decision

The truthful controller block remains the only production representation. It matched native bytes
on both supported local tiers, improved the 7B outcome, and reduced calls on both. The experiment
does not support either transcript-shaped arm, especially the undisclosed compatibility form.

The default remains `off`. This two-tier screen plus scripted safety tests is sufficient to ship
the typed, reversible, opt-in production mechanism, but not to make `review_only` or `safe` the
default. Default promotion still requires the repeated small-modify series and the full range,
review, closure, deletion, recovery, and final-source corpus in the plan. Deletion also remains a
separate explicit opt-in under `safe` regardless of future read promotion.

## Preserved evidence

| Classification | Scratch root |
| --- | --- |
| Valid final 4B run | `/private/tmp/pb-action-elision-prod-4b-final3-20260722` |
| Valid final 7B run | `/private/tmp/pb-action-elision-prod-7b-final3-20260722` |
| Valid pre-executable-digest 4B run | `/private/tmp/pb-action-elision-prod-4b-final2-20260722` |
| Valid pre-executable-digest 7B run | `/private/tmp/pb-action-elision-prod-7b-final2-20260722` |
| Valid pre-step-record 4B run | `/private/tmp/pb-action-elision-prod-4b-final-20260722` |
| Valid pre-step-record 7B run | `/private/tmp/pb-action-elision-prod-7b-final-20260722` |
| Valid pre-report-fix 4B run | `/private/tmp/pb-action-elision-prod-4b-8k-fix-20260722` |
| Valid defect-discovery host run | `/private/tmp/pb-action-elision-prod-4b-8k-host-20260722` |
| Excluded sandbox experiment error | `/private/tmp/pb-action-elision-prod-4b-8k-20260722` |
| Excluded sandbox experiment error | `/private/tmp/pb-action-elision-prod-7b-8k-20260722` |
| Excluded sandbox experiment error | `/private/tmp/pb-action-elision-prod-7b-20260722` |

The two final roots contain configuration and fixture manifests, per-arm event streams, all semantic
generation inputs and rendered-prompt digests, final artifacts, and the machine-readable summary.
