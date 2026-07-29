# Native Rust world qualification

- Date: **2026-07-28**
- Profile SHA-256: `738a7ceef37d5c020f4ec89b67d306448ff01cd8aba9fabd44da98c1baeddd42`
- Provider: **rust-analyzer crates 0.0.344**
- Semantic profile: **exact Rust v2**
- Artifact: **optimized macOS arm64 release binary**
- Host surface: **arm64, macOS 26.5.2 (25F84)**

## Result

The version 2 process-isolated matrix passed its default ceilings: 180 seconds for a complete cold
pre-inference barrier, 20 seconds for a warm or exact process-cache request, 30 seconds per
independent replay, 240 seconds for serialized-overlay stress, 4 GiB whole-process peak resident
memory, and 512 MiB current resident growth retained after stress.

Each fixture is an offline two-target Cargo workspace. The app target depends on a local dependency
target; the candidate replacement resolves and calls the dependency's public API. The invalid replay
passes a string to its `i32` parameter and must be rejected, while the valid replay changes the
integer call and must be accepted.

| Case | App / dependency files | Primed queries | Native load / prime | Cold | Warm / cache | Invalid / valid replay | Stress replays / total / maximum | Peak RSS / retained growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 6 / 6 | 21 | 246 / 1,148 ms | 2,250 ms | 7 / 7 ms | 160 / 9 ms | 65 / 148 / 12 ms | 872,955,904 / 180,224 bytes |
| Representative | 258 / 130 | 397 | 231 / 1,157 ms | 2,302 ms | 12 / 11 ms | 171 / 18 ms | 33 / 182 / 36 ms | 879,722,496 / 0 bytes |
| Large | 2,050 / 1,026 | 3,085 | 251 / 1,270 ms | 2,841 ms | 49 / 46 ms | 263 / 115 ms | 9 / 638 / 183 ms | 938,967,040 / 1,703,936 bytes |

The source/dependency byte pairs were 511/330, 25,711/8,908, and 204,911/71,652. Every arm reported
two Cargo targets and the exact Rust v2 deep profile before any replay was accepted.

## Lifecycle and serialization boundary

The cold timing crosses the ordinary production readiness boundary: verified Git-backed shadow,
offline Cargo metadata, sysroot and dependency graph, immutable read world, independently writable
Salsa world reconstructed from the loaded inputs, primed project queries, identity revalidation, and
a request-local streaming layer. Model entry cannot precede this barrier. Warm and process-cache
arms prove that request construction reuses the same exact identity without another Cargo load.

The stress arm uses 4 workers for tiny and representative and 2 for large. They submit alternating
invalid and valid complete replacements through the same production lifecycle, which serializes the
writable database and restores its baseline before releasing the lock. Tiny runs 16 replays per
worker, representative 8, and large 4. The reported count includes a final accepted recovery replay
after all workers join. Every expected decision matched.

Each case runs in a fresh process. Peak RSS is the operating system's whole-process maximum, not an
allocation claim for rust-analyzer alone. Retained growth is the saturating difference between
current RSS immediately before and after stress. The large arm retained 1.7 MB while remaining below
939 MB peak RSS.

## Reproduction

Build the web assets and optimized binary, then run:

```console
deno task build:web
cargo build --release --target aarch64-apple-darwin
target/aarch64-apple-darwin/release/pb harness native-world-qualify --language rust
```

The command emits content-free identities, counts, byte totals, timings, and memory values. Each
child creates and removes its own synthetic Git/Cargo workspace. It loads no model, invokes no LSP,
executes no build script or procedural macro, runs no generated code, and performs no network I/O.

## Scope

This is the accepted lifecycle, serialization, reclamation, and scaling baseline for the shipped
exact Rust v2 profile. It does not claim rustc equivalence, full borrow checking, procedural-macro or
build-script execution, source-topology authority, a universal project latency bound, or memory
behavior for substantially different dependency graphs. Those cases remain conservative `Unknown`
or separately qualified future profiles.
