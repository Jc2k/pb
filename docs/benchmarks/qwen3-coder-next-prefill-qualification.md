# Qwen3-Coder-Next prefill qualification

Status: **Layer-major command promoted; integrated browser-agent rerun pending**

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
prefill command kind. Promoted long prompts report `qwen_layer_major_matrix`; scalar controls
report `scalar_token`.

Promotion required scalar/batch state and greedy-token parity for zero/restored prefixes and all
chunk boundaries, resident zero-read behavior, streamed scheduler ownership, no more than 5%
additional request/session reserve, at least 5x over the 4–5k scalar baseline, and no more than 120
seconds at that frontier. The 2026-07-20 result below closes those deterministic and performance
gates; artifact quality remains independently gated by the full agent/browser evaluation.

## 2026-07-20 layer-major promotion

The final release arm64 binary prepared a layer-major graph only for the exact
`Qwen3NextMoe` resident-affine-Q4/fixed-affine-Q4 capability. `auto` keeps short suffixes and every
incompatible supported layout on the scalar graph. Compatible suffixes of at least 32 tokens use a
resource-resolved layer-major chunk of at most 8,192 rows; the calculation preserves a 512 MiB
safety margin and caps graph scratch at 5% of the current resident/session basis. An explicit
unsupported or unreservable layer-major qualification fails before execution rather than retrying
the scalar graph.

Exact parity was checked inside one loaded runtime. State fingerprints cover the final hidden row,
aggregate and per-layer full-attention KV, aggregate and per-layer router/recurrent records, and
aggregate and per-layer Metal linear-attention buffers.

| Expert graph | Prefix | Forced chunk | Result |
| --- | ---: | ---: | --- |
| complete resident corpus | 0 | 17 | exact after 40 fresh tokens |
| complete resident corpus | 10 cached + 30 fresh | 7 | warm-prefix and restored-prefix exact |
| forced 32 GiB streamed scheduler | 0 | 17 | exact after 40 fresh tokens |
| forced 32 GiB streamed scheduler | 10 cached + 30 fresh | 7 | warm-prefix and restored-prefix exact |

The resident scheduler tests observe zero issued or positioned reads and reuse the same mapped
expert slots. The streamed tests observe one positioned read for each member of the sorted unique
layer union on every request; a repeated request reads the union again, proving that the graph did
not add an application expert cache. Both live graphs finished with zero transient expert buffers
and zero in-flight commands.

The deterministic 4,354-token raw geometry is the ASCII sequence `a a ... a` with no trailing
space (input SHA-256
`a524b9f622ba729c0e9ecfea470bb4f002d57f37e3e97e8d4ade5827e40d9a26`). It isolates the same locked
frontier size from changing agent prompt prose; the integrated stage prompt remains part of the
subsequent browser-agent acceptance run. One resident graph boundary produced:

| Metric | Result | Gate |
| --- | ---: | ---: |
| fresh tokens | 4,354 | 4–5k frontier |
| prefill wall time | 59,685 ms | at most 120,000 ms |
| prefill throughput | 72.949 token/s | report |
| preserved scalar wall time | 689,600–690,200 ms | baseline |
| speedup vs 689,600 ms | 11.554x | at least 5x |
| resident baseline allocation | 44,981,485,568 bytes | baseline |
| layer-major peak allocation | 46,330,593,280 bytes | report |
| additional allocation | 1,349,107,712 bytes (2.9993%) | at most 5% |
| resource boundaries | 1 | report |
| live transient buffers / commands | 0 / 0 | required |

The greedy continuation was `a`, and telemetry reported `Qwen3NextMoe`, K=10,
`resident_complete_corpus`, `qwen_layer_major_matrix`, and thinking disabled. Relative to the
preserved baseline, all three promotion thresholds pass without changing quantization, expert
ownership, or sampling.

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

## 2026-07-21 host-boundary baseline

Phase-zero instrumentation for the device-resident continuation counts actual buffer copies,
tracked readbacks, and Metal command leases in the shared resource owner. Native generation
telemetry takes monotonic snapshots around prefill, so model load and sampling are excluded.

The resident control forced the shipped layer-major matrix command over the raw 40-token ASCII
prompt `a a ... a` (no trailing newline, SHA-256
`10bee2ef5c59d7909d35adf25157506b0be2ce8aec863a876a43f95e3a2057d5`). It used one 40-row
chunk and returned the greedy continuation `a`.

| Metric | Result |
| --- | ---: |
| prefill wall time | 793 ms |
| prefill throughput | 50.4075 token/s |
| Metal command buffers | 239 |
| host upload | 348,977,168 bytes |
| host readback | 186,818,560 bytes |
| expert strategy | `resident_complete_corpus` |
| request-end active/transient/in-flight | 0 / 0 / 0 |

The approximate five command buffers per transformer layer and more than 535 MB of host traffic on
only 40 rows confirm that the promoted command is still host-synchronized between matrix phases.
The existing traversal, parity, and scheduler evidence remains valid; promotion of a
device-resident successor now additionally requires zero hidden/normed transfers between layers.
