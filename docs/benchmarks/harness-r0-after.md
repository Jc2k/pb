# FlashMoe R0 after-state

Captured: 2026-07-13

This report repeats the protocol in `harness-r0-baseline.md` after the Metal ownership ledger and
working-set policy were implemented. Detailed timings and resource summaries were disabled for the
throughput trials. Resource summaries were enabled only for the lifecycle runs.

## Configuration

- Target: `aarch64-apple-darwin`, release profile
- Host: MacBook Pro (Mac16,6), Apple M4 Max, 16 cores, 64 GB
- Power: AC power
- Model: `hf://mlx-community/Qwen3.5-397B-A17B-4bit`
- Prompt: `List the integers from 1 to 100 in order, separated by commas.`
- Sampling: temperature `0`, top-k `1`, seed `1`
- Generated tokens per isolated sample: `16`
- Decode tokens per sample: `15`

Each throughput sample ran in a new process. Samples 1–3 are warmups and samples 4–10 are the
measured set.

## Throughput samples

| Sample | Use | Backend load ms | Total ms | Prefill/TTFT ms | Decode ms | Decode tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | warmup | 365 | 22,220 | 12,959 | 9,259 | 1.620 |
| 2 | warmup | 356 | 19,754 | 11,436 | 8,316 | 1.804 |
| 3 | warmup | 358 | 21,937 | 12,968 | 8,968 | 1.673 |
| 4 | measured | 359 | 22,095 | 13,084 | 9,009 | 1.665 |
| 5 | measured | 358 | 22,071 | 13,125 | 8,944 | 1.677 |
| 6 | measured | 360 | 22,121 | 13,103 | 9,016 | 1.664 |
| 7 | measured | 371 | 22,415 | 13,201 | 9,212 | 1.628 |
| 8 | measured | 360 | 22,239 | 13,156 | 9,081 | 1.652 |
| 9 | measured | 367 | 22,263 | 13,161 | 9,100 | 1.648 |
| 10 | measured | 366 | 22,197 | 13,165 | 9,031 | 1.661 |

Measured medians:

- decode throughput: `1.661 tok/s` (MAD `0.009 tok/s`)
- generation elapsed: `22,197 ms` (MAD `76 ms`)
- prefill/TTFT: `13,156 ms`
- decode elapsed: `9,031 ms`

The before-state median was `1.590 tok/s`; the after-state is 4.5% higher and comfortably above the
predeclared `1.542 tok/s` regression floor. Generation elapsed was 21.8% lower. These measurements
show that R0 maintained the performance gate and observed better throughput under the same
protocol; they do not establish that resource accounting alone caused the improvement.

## Ownership defect and narrow smoke

The first ownership-invariant run identified 15 deferred CMD3 submissions that were removed from
an `Option` before a later guard chose not to wait for them. The retained resources matched the
control-flow error exactly: 420 general buffers (28 per submission) and 60 transient expert buffers
(4 per submission). Reordering the guard and adding drop-time submission cleanup fixed the leak.

The required one-token narrow smoke exited 0 and produced a token. At its final snapshot:

- active general buffers: `0`
- transient expert buffers: `0`
- in-flight commands: `0`
- pooled buffers: `36` (`1,468,480` bytes)
- driver allocation/high-water: `6,630,211,584` bytes
- declared working-set limit: `50,096,509,748` bytes

## Repeated-prompt regression

The before-state process was killed with exit 137 during prompt 5. The after-state process completed
all ten identical 32-token prompts:

- generated tokens: `320`
- elapsed: `317,684 ms`
- decode tokens: `310`
- decode throughput: `1.658 tok/s`
- driver allocation at prompts 1 and 10: `6,630,211,584` bytes
- ledger high-water: `6,658,089,536` bytes
- final active general/transient/in-flight counts: `0 / 0 / 0`

The driver allocation, pool size, and zero-live-work invariants were unchanged from prompt 1 through
prompt 10. Allocation and release counters continued increasing because expert staging is streamed,
but live resources did not accumulate.

## Sustained soak and constrained limit

The separate 128-generated-token run completed in `89,046 ms` at `1.681` decode tok/s. Its final
snapshot again reported zero active general buffers, zero transient expert buffers, and zero
in-flight commands, with the same `6,630,211,584` byte driver allocation.

With `--metal-working-set-limit-mib 6000`, generation exited non-zero before an allocation and
reported a structured `FlashMoe Metal resource limit would be exceeded` diagnostic. It included the
requested allocation, current allocation (`6,628,769,792` bytes), configured limit
(`6,291,456,000` bytes), device recommendation, pool-drain result, high-water, and ledger live
bytes. The process was not terminated by the OS.

## Disposition

R0 passes its release soak, constrained-abort, lifecycle-balance, narrow-smoke, and throughput
acceptance criteria.
