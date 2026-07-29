# Native Python semantic qualification

- Date: **2026-07-28**
- Corpus SHA-256: `e073204a58aab27a8f52c17912734f138e7a6a239c2999d95841ece93f1f0b40`
- Provider: **Astral `ty` 0.0.6**
- Artifact: **optimized macOS arm64 release binary**
- Host surface: **arm64, macOS 26.5.2 (25F84)**

## Result

The version 1 checked-in corpus passed all production-path gates. Its 24 cases produced 9 allows
and 15 semantic rejections, and every case had the same outcome at generation closure and at the
independent execution-time replay. A third, language-owned replay returned the exact expected set
of promoted diagnostic codes for every case.

| Category | Cases |
| --- | ---: |
| Annotated | 10 |
| Unannotated | 3 |
| Frozen third-party dependency | 4 |
| Baseline debt | 2 |
| Multi-file transaction | 4 |
| Dynamic unknown | 1 |

The rejection expectations exercised `invalid-argument-type` 3 times, `invalid-assignment` 3,
`invalid-return-type` 2, `unresolved-attribute` 3, `unresolved-import` 2, and
`unsupported-operator` 3. The corpus covers create, replace, exact edit, and canonical multi-file
patch calls.

The final release run's native-world readiness barrier took 42 ms. Across the measured
warm-preparation, generation, execution-replay, direct-diagnostic, prefix, and rollback stages,
latency was 8 ms p50, 15 ms p95, and 16 ms p99/maximum. The same run completed 2,007 exhaustive
logical UTF-8 prefix probes and 1,536 deterministic rollback/full-replay branches. Once a prefix
became impossible, every longer prefix remained impossible. The accepted qualifier ceilings remain
deliberately loose at 60 seconds cold and 20 seconds per case stage; they are failure bounds, not
user-facing latency promises.

## Dependency and lifecycle boundary

The corpus materializes nine first-party files and a three-file static dependency image in an
ephemeral, ignored project-local `.venv`. The dependency contains a small, corpus-specific typed
HTTP client surface and package metadata; it is not a vendored full upstream distribution. Its
purpose is to prove that imports, callable signatures, attributes, and result types flow through
the same frozen native resolver used by production.

The native world is loaded and every captured source/stub module is primed before the first case.
Each case then gets a fresh request database. Its complete mutation is checked once through the
generation-time prefix/closure path, reconstructed independently immediately before the simulated
executor boundary, and replayed a third time to compare only promoted diagnostic-code deltas.
There is no live LSP, package installation, network access, model inference, or incomplete-document
diagnostic stream.

## Reproduction

Build the web assets and optimized binary, then run:

```console
deno task build:web
cargo build --release --target aarch64-apple-darwin
target/aarch64-apple-darwin/release/pb harness python-semantic-qualify \
  --corpus fixtures/control-collar/semantic-python-v1.json
```

The command emits only content-free identities, profile/category/diagnostic counts, parity and
prefix/rollback counts, and timings. The corpus contract rejects unsafe dependency paths, absent
annotated/unannotated/third-party allow-or-reject arms, unpromoted diagnostic expectations,
duplicate codes, missing mutation-tool coverage, and incomplete promoted-code coverage.

## Scope

This is differential evidence for the current six-code Python profile and the existing
string-plus-integer token proof. It does not make dynamic Python sound, prove the complete HTTP
client API, exercise the separately qualified editable/external-environment lifecycle profiles,
grant token-time authority to the five closure-only codes, or replace broader public-project and
independent external-oracle corpora. The checked
[external-oracle interface](native-python-external-oracle.md) materializes this corpus for such a
comparison without making an external checker a pb runtime dependency or authority.
