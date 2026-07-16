# Small-model agent S2 focused-evidence checkpoint

Captured: 2026-07-16

Plan: [Small-model agent reliability plan](../small-model-agent-reliability-plan.md)

S2 removes duplicated repository material from planning and isolated code review without changing
the complete graph, content fingerprint, check ledger, read-evidence, or artifact-validation
authorities. This checkpoint is deterministic and model-free; real-model effectiveness remains an
S6 measurement.

## Production behavior

- Planning and plan review receive a deterministic repository brief capped at 16,000 characters.
  It retains the complete graph SHA-256 and explicit omission counts.
- Code review receives a changed-path manifest capped at 16,000 characters and selected-check
  evidence summaries capped at 8,000 characters, not a full diff plus duplicate full current
  files.
- `inspect_change(path)` is available only in code review and returns at most the active tool-result
  and configured per-inspection byte budgets. It includes status, previous path, content kind,
  checked fingerprint, focused diff hunks, bounded numbered current context, and relevant check
  evidence. An undersized budget fails before earning read evidence.
- Successful current-text inspection records the same fresh path evidence as `read_file`.
- Deleted and binary paths do not claim text-read evidence. Pure renames are deterministically
  paired by the task-baseline/current path fingerprints.
- The complete normalized graph still validates every plan component/check ID, and the full
  checked workspace/check ledger still validates every code-review submission.

## Deterministic acceptance

| Fixture | Proof | Result |
| --- | --- | --- |
| RV1 | Three changed 5,000-line files produce a manifest-only initial review prompt; each later focused inspection remains under 12,000 characters and contains hunks/current context/fingerprint | pass |
| RV1 reduction | Constructed S2 review prompt is asserted at least 40% smaller than the same prompt with S0's complete diff plus duplicate complete current files | pass |
| RV2 | New, deleted, pure-renamed, and binary paths have distinct manifest and inspection representations; unavailable text earns no read evidence | pass |
| RV3 | `submit_code_review` is hidden before inspection, a premature submission reports the exact missing path, and the terminal tool appears only after `inspect_change` earns fresh evidence | pass |
| RB1 | A synthetic 180-component/check/task polyglot graph yields identical repeated briefs at or below 16,000 characters with its full graph hash, manifests, and omission accounting | pass |
| Scope/bounds | Ordinary agents never receive `inspect_change`; oversized manifests and undersized inspections fail before generation/evidence, and UTF-8 output respects the configured byte cap | pass |

The existing plan-artifact validator continues to reject component/check IDs absent from the full
trusted graph. Existing review validation continues to require current selected-check evidence,
fresh path reads for every current changed text path, findings restricted to the task delta, and the
exact checked content fingerprint.

## Reproduction

```bash
cargo test agent_repository::tests
cargo test rv1_large_review_prompt_uses_manifest_and_focused_inspection
cargo test rv3_code_review_terminal_is_unavailable_until_changed_text_is_inspected
cargo test agent_core::tests
cargo run --quiet -- harness eval --jsonl /tmp/pb-harness-s2-scripted.jsonl
cargo run --quiet -- harness eval --suite small-model \
  --jsonl /tmp/pb-small-model-s2-scripted.jsonl
```

The RV1 comparison deliberately measures constructed prompt material, not tokenizer behavior: the
same task, plan, implementation accounting, fingerprint, and check facts are held constant while
only S0's duplicated review payload is replaced by the S2 manifest. Exact tokenizer/context
accounting remains covered by S1.
