# Native Python world qualification

- Date: **2026-07-28**
- Profile SHA-256: `a6093b4fa3f4b762d432291226aee27c111afae72745aa0dede09a64ee0ac0bc`
- Provider: **Astral `ty` 0.0.6**
- Artifact: **optimized macOS arm64 release binary**
- Host surface: **arm64, macOS 26.5.2 (25F84)**

## Result

The version 2 process-isolated matrix passed its default ceilings: 60 seconds for a complete cold
pre-inference barrier, 20 seconds for a warm or exact process-cache request, 20 seconds for each
independent final replay, 120 seconds for serialized-overlay stress, 1 GiB whole-process peak
resident memory, and 512 MiB of current resident growth retained after stress.

| Case | First-party / dependency files | Primed queries | Native load / prime | Cold | Warm / cache | Invalid / valid replay | Stress replays / total / maximum | Peak RSS / retained growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 4 / 7 | 9 | 1 / 11 ms | 50 ms | 16 / 15 ms | 18 / 17 ms | 65 / 516 / 41 ms | 44,105,728 / 2,621,440 bytes |
| Representative | 1,024 / 515 | 1,537 | 21 / 41 ms | 317 ms | 67 / 68 ms | 74 / 73 ms | 33 / 1,469 / 219 ms | 59,342,848 / 4,931,584 bytes |
| Large | 10,000 / 5,003 | 15,001 | 282 / 375 ms | 3,121 ms | 612 / 606 ms | 666 / 670 ms | 9 / 4,126 / 1,178 ms | 201,850,880 / 14,499,840 bytes |

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

The stress arm uses 4 workers for tiny and representative and 2 for large. They submit alternating
invalid and valid replays through the same production lifecycle, which serializes access to the
writable Salsa overlay. Tiny runs 16 replays per worker, representative 8, and large 4; the reported
counts include one final accepted recovery replay after all workers join. Every decision matched,
and the recovery proves a rejecting branch did not leak candidate state into the next request.

Every case runs in a fresh child process. Peak memory is the operating system's whole-process
maximum resident set; incremental peak subtracts the maximum already observed after fixture
construction. Retained growth is the saturating difference between current RSS immediately before
and after stress. Neither is an allocation-accounting claim for `ty` alone.

## Reproduction

Build the web assets and optimized binary, then run:

```console
deno task build:web
cargo build --release --target aarch64-apple-darwin
target/aarch64-apple-darwin/release/pb harness native-world-qualify --language python
```

The command emits only content-free identities, counts, byte totals, timings, and memory values.
Each child creates and removes its own synthetic Git workspace and ignored `.venv`.

The same production lifecycle also has deterministic focused regression coverage for a fully
observed repository-local plain-path editable, an exact user-authorized external editable, and an
exact user-authorized external environment. Those cases prove native import/call-shape resolution,
repository-versus-user authority isolation, external-byte drift rejection, cold recapture, and
independent final replay. They share the limits exercised above but are not additional timing rows
in this version 1 resource matrix.

## Scope

This is the accepted lifecycle, serialization, reclamation, and scaling baseline for the shipped
Python profile over simple, fully annotated static module graphs. It does not establish a universal
project latency promise, a complex-AST throughput bound, a separate large external-editable resource
profile, dynamic/import-hook/native-extension authority, dynamic Python soundness, or semantic
authority beyond the separately qualified corpus.
