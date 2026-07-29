# Native Rust semantic qualification

- Date: **2026-07-28**
- Corpus SHA-256: `30a19114d1b4071503d07977bafd198520ce9517c08dee6c81b331b2e03ce904`
- Provider: **rust-analyzer crates 0.0.344**
- Semantic profile: **exact Rust v2**
- Artifact: **optimized macOS arm64 release binary**
- Host surface: **arm64, macOS 26.5.2 (25F84)**

## Result

The checked-in version 2 corpus passed all production-path gates. Its 23 cases produced 10 allows
and 13 semantic rejections. Every case had the same outcome at generation closure and independent
execution-time replay, and the language-owned replay returned the exact expected promoted
diagnostic or conservative-unknown set.

| Category | Cases |
| --- | ---: |
| Positive | 4 |
| Promoted diagnostic | 11 |
| Baseline debt | 2 |
| Cross-crate transaction | 2 |
| Conservative unknown | 4 |

The corpus covers all ten promoted diagnostic classes: unresolved names/imports, missing
fields/methods, privacy, invalid calls, type mismatch, mutability, moved-from-reference ownership,
and trait contracts. It exercises replace, write, exact edit, and canonical multi-file patch calls.
The unknown cases require safe deferral for relative import context and create/delete source
topology rather than converting incomplete project facts into rejection authority.

Cold native readiness took 2,183 ms. Across the measured case, prefix, and rollback stages, latency
was 6 ms p50, 11 ms p95, and 170 ms p99/maximum. The run completed 2,758 exhaustive logical UTF-8
prefix probes and 1,472 deterministic rollback/full-replay branches. Once a prefix became
impossible, every longer prefix remained impossible.

## Project and lifecycle boundary

The corpus materializes one offline two-target Cargo workspace with a local dependency crate and a
preserved baseline error. The exact profile loads and primes one immutable rust-analyzer read world
plus one independently writable Salsa world before the first case. Complete modifications to
indexed sources are overlaid as one transaction; promoted diagnostic multisets are compared against
the baseline across local target modules, then the writable database is restored before another
case can use it.

Every candidate is checked through the production streaming gate, reconstructed independently at
the simulated executor boundary, and compared with a direct content-free diagnostic replay. The
qualifier runs no model or LSP, executes no build script or procedural macro, runs no generated code,
and performs no network I/O.

## Reproduction

Build the web assets and optimized binary, then run:

```console
deno task build:web
cargo build --release --target aarch64-apple-darwin
target/aarch64-apple-darwin/release/pb harness rust-semantic-qualify \
  --corpus fixtures/control-collar/semantic-rust-v2.json
```

The command emits only corpus/world/configuration/dependency digests, category and diagnostic
counts, allow/reject/unknown parity, prefix and rollback counts, and timings.

## Scope

This is differential evidence for the current exact Rust v2 promoted allowlist. It does not claim
rustc equivalence, full borrow checking, build-script or procedural-macro behavior, source-topology
authority, arbitrary `cfg`/target coverage, or runtime correctness. Those cases remain conservative
`Unknown` or separately qualified future profiles.
