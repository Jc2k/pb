# FlashMoe R0 before-state

Captured: 2026-07-13T16:20:58Z

This is the pre-implementation baseline for R0 in
`docs/harness-improvement-plan.md`. It separates throughput measurement from the resource failure
that motivated the milestone.

## Configuration

- Commit: `bfa83a34981dd571a37bd20dec288e258092854e`
- Release binary SHA-256: `d53cc55f22fd767191c597f70fd4cb3df1082c3e0e08dd97d47fd3b341926f5a`
- Target: `aarch64-apple-darwin`
- Host: MacBook Pro (Mac16,6), Apple M4 Max, 16 cores, 64 GB
- Power: AC power, battery fully charged
- Model: `hf://mlx-community/Qwen3.5-397B-A17B-4bit`
- Prompt: `List the integers from 1 to 100 in order, separated by commas.`
- Sampling: temperature `0`, top-k `1`, seed `1`
- Generated tokens per isolated sample: `16`
- Decode tokens per sample: `15`
- Detailed timings: disabled

Each throughput sample ran in a new process so the backend and its Metal resources were released
between trials. The first three samples are warmups; samples 4–10 are the measured set.

## Throughput samples

| Sample | Use | Backend load ms | Total ms | Prefill/TTFT ms | Decode ms | Decode tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | warmup | 406 | 24,485 | 16,041 | 8,442 | 1.777 |
| 2 | warmup | 366 | 24,902 | 15,683 | 9,216 | 1.627 |
| 3 | warmup | 417 | 25,637 | 16,681 | 8,951 | 1.676 |
| 4 | measured | 363 | 28,391 | 18,939 | 9,448 | 1.588 |
| 5 | measured | 363 | 28,410 | 18,992 | 9,414 | 1.593 |
| 6 | measured | 360 | 28,413 | 18,976 | 9,434 | 1.590 |
| 7 | measured | 358 | 28,290 | 18,888 | 9,398 | 1.596 |
| 8 | measured | 355 | 28,356 | 18,839 | 9,515 | 1.576 |
| 9 | measured | 356 | 28,420 | 18,913 | 9,506 | 1.578 |
| 10 | measured | 390 | 25,945 | 16,757 | 9,183 | 1.633 |

Measured medians:

- decode throughput: `1.590 tok/s` (MAD `0.006 tok/s`)
- generation elapsed: `28,391 ms` (MAD `29 ms`)
- prefill/TTFT: `18,913 ms`
- decode elapsed: `9,434 ms`

The R0 after-state fails the performance gate if median decode tok/s is below `1.542 tok/s` (3%
below the baseline) and the regression also exceeds the combined before/after MAD.

## Repeated-prompt resource failure

Before collecting isolated samples, one process loaded the same backend once and attempted ten
identical 32-token prompts. It was killed with exit status `137` during prompt 5. The first four
completed results were:

| Prompt | Generated | Total ms | Prefill/TTFT ms | Decode ms | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 41,329 | 21,328 | 19,999 | 1.550 |
| 2 | 32 | 39,809 | 19,182 | 20,626 | 1.503 |
| 3 | 32 | 39,789 | 19,165 | 20,622 | 1.503 |
| 4 | 32 | 38,908 | 18,482 | 20,425 | 1.518 |

No prompt-5 summary or structured Metal high-water evidence was emitted. This is direct P0 evidence
for R0: the current runtime can be terminated by the OS during a bounded repeated-prompt session,
and the existing allocation-failure diagnostic does not run early enough to preserve a useful
resource ledger.

## After-state protocol

After R0:

1. Repeat the isolated three-warmup/seven-measured 16-token series with the same configuration.
2. Compare median decode tok/s and elapsed time using the gate above.
3. Repeat the ten-prompt, 32-token in-process run. It must either finish within the declared Metal
   cap or abort before OS termination with a structured resource-limit event.
4. Run the separate 128-token soak required by R0 and preserve its ledger/high-water summary.
