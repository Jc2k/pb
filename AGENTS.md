# AGENTS.md

## Commands

- `deno task build:web` - Build web UI assets (required before `cargo build`)
- `cargo fmt --all -- --check` - Verify Rust formatting
- `cargo clippy --all-targets --all-features -- -D warnings -A clippy::all -D clippy::correctness -D clippy::suspicious` - Enforce compiler warnings plus high-signal correctness and suspicious-code lints
- `cargo test --all-targets` - Run Rust tests
- `deno task test:web` - Run web UI tests
- `deno task test:docs` - Build the mdBook site and validate rendered links, assets, fragments, and PWA metadata
- `cargo build --release --target aarch64-apple-darwin` - Build release binary for macOS arm64

## Architecture

- **Entry point**: `src/main.rs` → `pb::run(Cli::parse()).async`
- **CLI commands** defined in `src/lib.rs`: `serve`, `queue`, `self`, `tray`, `projects`, `env`, `service`, `init`; the hidden `harness` surface is documented in `docs/harness.md`
- **Web UI**: React app in `webui/` served by Rust backend; builds to `webui/dist/`
- **Agent profiles**: `build|scout|review|explore|plan|ask|research` - see `src/agent_core.rs`
- **Documentation**: mdBook configured by `book.toml`; navigation and inclusion are defined in `docs/SUMMARY.md`

## Platform-specific behavior

- `pb tray` only works on macOS; other platforms return error
- Service management (`pb service`) uses launchd on macOS

## Critical conventions

1. **Semantic commits only**: `feat:`, `fix:`, `chore:`, etc.
2. **Update `src/init.rs`** when adding new per-project configuration fields
3. **Web UI requirements**: Must include `viewport-fit=cover`, `env(safe-area-inset-*)` CSS, and PWA meta tags
4. **Web UI tests**: Add or update `webui/src/**/*.test.ts` tests for web UI behavior changes, and run `deno task test:web` before committing
5. **Documentation parity**: When changing user-visible behavior or architectural guarantees, update the relevant curated chapter under `docs/user/` or `docs/architecture/` in the same commit and run `deno task test:docs`
6. **FlashMoe architecture**: When changing FlashMoe data flow, scheduling, expert I/O, Metal kernels, or model-family behavior, keep `docs/flashmoe-architecture-parity-plan.md` aligned with the target architecture and current migration status.
7. **No standalone environment controls**: User-visible pb behavior must use typed user/project configuration or an explicit CLI argument. Do not add `PB_*` environment flags or feature toggles. Parent-process environment reads belong only in `src/host_environment.rs` and must be limited to operating-system conventions or secret names explicitly declared by configuration. Child-process environment values and test-only process handoffs are separate, scoped mechanisms. `PB_GITHUB_CLIENT_ID` is an explicitly grandfathered build-time exception and must not be expanded or refactored without a dedicated decision.
8. **Rust owns the web boundary**: Lifecycle state, project/session collections, usage totals and requested usage-window aggregates, terminal transitions, Trinity chatter, and API failures are server-authored. React mutation handlers may clear local form state after success, but must not synthesize domain or lifecycle state, refetch to discover a mutation result, or add polling to compensate for a missing server model. Extend the typed response or a revisioned SSE snapshot instead; when a mutation changes a streamed collection, its response must use the same revisioned projection so delayed or unavailable SSE cannot leave the control stale. Stream subscription floors, revision publication, retained transitions, and derived usage must share one causal boundary; derive totals from the captured session projection and keep any retained window or generation cache explicitly bounded. Each connection receives only terminal-transition deltas newer than its advancing cursor, and only the live stream may authorize a new process generation. Loaded browser data remains usable during reconnect and must be distinguished from an initial request with no snapshot. A multi-await mutation transaction must be owned independently of the caller so disconnect or cancellation cannot split durable state, in-memory projections, accounting, and revision publication. A mutation must not return failure after its authoritative state has committed; represent incomplete ancillary cleanup as success with typed warnings. Every boundary change must keep Rust and TypeScript fields and enum values exact, update the browser runtime parser, and add a behavioral cross-language contract or regression test that would fail if the workaround returned.

## Documentation guidance

- Treat `docs/architecture/` as the current, curated description of shipped pb behavior. Detailed top-level plans and benchmark records preserve design history and evidence; they do not replace an update to the curated architecture.
- Update `docs/architecture/workflows.md` and `docs/architecture/user-contracts.md` when changing intent handling, workflow stages, tool authority, checks, review, commit ownership, terminal outcomes, or external publication boundaries.
- Update `docs/architecture/security.md` when changing capabilities, delegation, policy, workspace isolation, execution environments, credentials, service access, or trust boundaries.
- Update `docs/architecture/local-privacy.md` when changing inference placement, persistence, data ownership, network access, integrations, or any path by which data can leave the machine.
- Update the matching chapter under `docs/user/` when a command, configuration field, default, setup flow, integration, data location, or cleanup behavior changes.
- Preserve the site's status language: distinguish **Shipped**, **Configurable**, and **Design record** behavior. Never present a planned hardening item as an enforced guarantee.
- Add new chapters to `docs/SUMMARY.md`; keep links document-relative so the `/pb/` GitHub Pages base path continues to work.
- Do not commit the generated `site/` directory.

## FlashMoe guidance

- Treat `docs/flashmoe-architecture-parity-plan.md` as the source of truth for FlashMoe work.
- Keep the upstream `danveloper/flash-moe` data flow visible: 4-bit quality is the production target, K=4 routing is the Qwen3.5 baseline, expert I/O should stream through scheduler-owned parallel `pread`, and the OS page cache should be trusted.
- Do not chase isolated FlashMoe microbenchmarks or add Q4-only fast paths without a plan for all supported Qwen MoE variants to use the same data flow, scheduling, and CPU/GPU handoff.
- Do not reintroduce hidden FlashMoe environment toggles, application expert caches, cache migrations, or alternate runtimes without updating the architecture plan first.
- After FlashMoe backend changes, run the narrow smoke at minimum: `target/aarch64-apple-darwin/release/pb harness infer --raw --max-tokens 1 --top-k 1 --temperature 0 "2+2="`. It must exit 0 and print a sensible answer.

## Release process

1. PR merged to `main` → semantic-release tag created
2. GitHub Actions builds macOS arm64 binary from `dist/pb-aarch64-apple-darwin`
