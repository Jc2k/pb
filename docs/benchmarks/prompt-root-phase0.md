# Prompt-root Phase 0 baseline

Status: **Single-case diagnostic; the Phase 0 three-language gate remains open.**

Revision `f6fd5a71` added privacy-safe exact rendered-root identity and the first non-overlapping
FlashMoe refill counters. Before changing controller root layout or retention, the release binary
ran the locked Rust registry case directly with Qwen3-Coder-Next MLX Q4, temperature 0, top-k 1,
seed 0, seven steps, and 1,792 generated tokens per turn. The machine-readable record is
`../../fixtures/harness-usability/baselines/2026-07-25-prompt-root-phase0-rust.json`.

The independent auditor reran the official check and immutable-fixture check, matched the recorded
semantic commit to Git, confirmed the clean worktree and allowed path, and classified the result as
positive evidence. The run reached verified-clean completion in four model invocations.

| Result | Value |
| --- | ---: |
| Rendered prompt tokens | 18,523 |
| Complete eligible root tokens | 9,488 |
| Reused root tokens | 9,488 (100%) |
| Fresh suffix tokens | 9,035 |
| Root-hit invocations | 4/4 |
| Refill lookup | 810 ms |
| Metal state hydration | 9 ms |
| Fresh suffix prefill | 78,978 ms |
| Snapshot capture | 48 ms |
| Wall time | 172,786 ms |
| Energy | 2.456 Wh |

The four roots represented planning, plan review, implementation mutation, and code review. Their
token digests were all distinct, as required by their different exact tool authority. Complete disk
root restoration worked; fresh suffix prefill still dominated measured refill time by two orders of
magnitude over lookup and hydration.

One excluded setup run used the Deno orchestration wrapper and could not acquire a Metal device.
The direct release process used Metal normally, so the excluded result is an experiment error, not
a pb or model failure.

This run is deliberately not promoted to the Phase 0 gate. It predates the split disk-decode and CPU
validation/allocation timers plus queue and durable-completion counters, and it covers only Rust.
The locked Rust/Python/React rerun remains required before Phase 0 closes.
