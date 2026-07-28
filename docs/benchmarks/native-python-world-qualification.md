# Native Python world qualification

- Date: **2026-07-28**
- Profile SHA-256: `aeeca87bd8032bede70a174dd124ee341de33beeed1739998c11d94b8134966e`
- Provider: **Astral `ty` 0.0.6**
- Artifact: **optimized macOS arm64 release binary**
- Host surface: **arm64, macOS 26.5.2 (25F84)**

## Result

The version 1 process-isolated matrix passed its default ceilings: 60 seconds for a complete cold
pre-inference barrier, 20 seconds for a warm or exact process-cache request, 20 seconds for each
independent final replay, and 1 GiB whole-process peak resident memory.

| Case | First-party / dependency files | Primed queries | Native load / prime | Complete cold barrier | Warm / process cache | Invalid / valid replay | Process peak / incremental peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 4 / 7 | 9 | 1 / 11 ms | 49 ms | 14 / 14 ms | 16 / 15 ms | 39,550,976 / 22,970,368 bytes |
| Representative | 1,024 / 515 | 1,537 | 22 / 42 ms | 337 ms | 67 / 67 ms | 73 / 74 ms | 53,657,600 / 36,978,688 bytes |
| Large | 10,000 / 5,003 | 15,001 | 280 / 371 ms | 3,106 ms | 601 / 602 ms | 662 / 660 ms | 192,380,928 / 175,718,400 bytes |

The source/dependency byte pairs were 405/288, 113,625/29,752, and
1,109,961/290,056. The matrix therefore stresses project and dependency module count more than
large individual syntax trees.

## What the timings include

`cold_millis` is the wall clock at the production readiness boundary. It includes the Git-backed
content snapshot, ignored local-environment discovery, deterministic dependency walk and hashing,
immutable shadow construction, `ty` database load and complete first-party/dependency priming,
post-prime source and dependency revalidation, a fresh request database, and streaming-layer
construction. No prompt reservation, invocation accounting, or model entry occurs before this
returns.

`load_millis` and `prime_millis` are the narrower readiness-receipt components reported by the
language crate. `warm_millis` repeats live identity/dependency capture and constructs and primes a
fresh request database from the already prepared world. The process-cache arm creates a new
controller lifecycle and proves it receives the same exact world without a second cold load.

Each replay recaptures the live source and dependency identities, constructs another independent
request database, overlays a real changed candidate, and applies the production final semantic
gate. The invalid arm must reject a string-parameter function called with an integer; the valid arm
must accept a changed string call. Thus these measurements cover the executor-side replay contract,
not only analyzer startup.

Every case runs in a fresh child process. Peak memory is the operating system's whole-process
maximum resident set; incremental peak subtracts the maximum already observed after fixture
construction. It is not an allocation-accounting claim for `ty` alone.

## Reproduction

Build the web assets and optimized binary, then run:

```console
deno task build:web
cargo build --release --target aarch64-apple-darwin
target/aarch64-apple-darwin/release/pb harness native-world-qualify --language python
```

The command emits only content-free identities, counts, byte totals, timings, and memory values.
Each child creates and removes its own synthetic Git workspace and ignored `.venv`.

## Scope

This is an accepted lifecycle/scaling baseline for simple, fully annotated static module graphs. It
does not establish a universal project latency promise, a complex-AST throughput bound, editable or
out-of-project environment support, native-extension authority, dynamic Python soundness, or a
semantic false-rejection rate. Those remain separate Phase 7E/7H qualification work.
