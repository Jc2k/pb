# Qwen3-Coder-Next prefill qualification

Status: **Locked baseline; chunk-owner command measured and not promoted**

This is the repeatable native-runner performance contract derived from the preserved Typing Defense
evaluation. It does not duplicate private prompt bodies. A qualification run renders the current
strict stage prompt from the named fixture/task, records the resulting token count and SHA-256, and
must retain that rendered hash beside its result. The descriptor hashes below lock geometry and
generation policy even when a prompt body cannot be checked in.

| Geometry | Target rendered tokens | Descriptor SHA-256 | Preserved baseline |
| --- | ---: | --- | ---: |
| short control | about 512 | `2b2d39f4c7de2c681e64c66106ddd293726fa823fc30a1d3e710692a1ddc8a6e` | record on qualification host |
| Typing Defense planning frontier | 4,354 | `661114ed9d0da114d36ba19bcf24e170f1dc8a235440557bd6137661d8988c26` | 689.6–690.2 s |
| Typing Defense plan-review frontier | 6,500–7,000 | `e8dc4e485108ff049bf1517aeaef52becda4710c8beab260226ab545deff89bc` | roughly 12–20 min |

Each descriptor is the SHA-256 of
`qwen3-coder-next-prefill-v1|<geometry>|<tokens>|non-thinking|top-k=1|temp=0`.
The before run IDs are `1784491831209-39233-0`, `1784500308030-56072-0`, and
`1784503415217-61220-0`; their classification and task contract remain in
[Qwen3-Coder-Next Native Agent Evaluation](qwen3-coder-next-agent.md).

Use the release Apple-Silicon binary and the resident native source. Capture the resource summary
and detailed timing output, then repeat with a 32 GiB limit to force the existing streamed graph:

```bash
target/aarch64-apple-darwin/release/pb harness infer \
  --model hf://mlx-community/Qwen3-Coder-Next-4bit \
  --no-thinking --max-tokens 1 --temperature 0 --top-k 1 \
  --resource-summary "<rendered prompt>"

target/aarch64-apple-darwin/release/pb harness infer \
  --model hf://mlx-community/Qwen3-Coder-Next-4bit \
  --metal-working-set-limit-mib 32768 \
  --no-thinking --max-tokens 1 --temperature 0 --top-k 1 \
  --resource-summary "<rendered prompt>"
```

The result must identify `Qwen3NextMoe`, K=10, effective thinking off, fresh/cached token counts,
prefill and decode separately, `resident_complete_corpus` or `streamed_parallel_pread`, and the
prefill command kind. The current `qwen_chunked_token_batch` is an ownership/command-boundary
migration step and is not the target layer-major command.

Promotion requires scalar/batch state and greedy-token parity for zero/restored prefixes and all
chunk boundaries, resident zero-read behavior, streamed scheduler ownership, no more than 5%
additional request/session reserve, at least 5x over the 4–5k scalar baseline, and no more than 120
seconds at that frontier. Until all gates pass, record the measurement and leave the target
layer-major command unpromoted.

## 2026-07-20 chunk-owner measurement

The release arm64 binary passed the resident and forced-32-GiB raw `2+2=` smoke with the greedy
continuation `5`. Both runs resolved `Qwen3NextMoe`, K=10, and effective thinking off. The default
graph reported `resident_complete_corpus`; the constrained graph reported
`streamed_parallel_pread` with no live transient buffers or in-flight commands at request end.

A resident short-control run then used a raw prompt containing 512 space-separated `a` tokens.
The rendered prompt SHA-256 was
`1c9b7a4ea873d5032662593a9694a561b18f43b6b9fc403029650fbf1ff4f37f`. It measured 512 fresh
tokens in 30,770 ms, or 16.64 token/s, with `qwen_chunked_token_batch`, eight resource boundaries,
and 44,974,407,680 bytes allocated after load. The request completed with balanced resource counts.

This result does not meet the layer-major promotion gate. Even a deliberately optimistic linear
extrapolation with no sequence-length cost puts the 4,354-token frontier above 261 seconds, versus
the required 120 seconds. The 4,354-token and full agent/browser reruns were therefore not started:
the current command cannot satisfy the locked gate, and another expensive run would not change
that decision. The next performance experiment must be the actual layer-major Qwen graph described
in the plan, not a larger chunk or a weaker threshold.

The same release build profiled four greedy raw tokens with detailed per-layer timings. The three
decode tokens took 470.0 ms (6.38 token/s). The fused attention/projection bucket accounted for
391.7 ms (83%), sampling for 37.8 ms, combine/norm for 16.3 ms, and expert compute for 6.5 ms.
Accordingly, no decode change was promoted: a sampling-only tweak cannot meet the 1.5x gate. The
next decode experiment must reduce the common Qwen attention/projection command and synchronization
cost while preserving both resident and streamed graph ownership.
