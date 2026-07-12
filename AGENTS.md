# AGENTS.md

## Commands

- `deno task build:web` - Build web UI assets (required before `cargo build`)
- `cargo test --all-targets` - Run Rust tests
- `deno task test:web` - Run web UI tests
- `cargo build --release --target aarch64-apple-darwin` - Build release binary for macOS arm64

## Architecture

- **Entry point**: `src/main.rs` → `pb::run(Cli::parse()).async`
- **CLI commands** defined in `src/lib.rs`: `serve`, `queue`, `self`, `tray`, `projects`, `env`, `service`, `init`; the hidden `harness` surface is documented in `docs/harness.md`
- **Web UI**: React app in `webui/` served by Rust backend; builds to `webui/dist/`
- **Agent profiles**: `build|scout|review|explore|plan|ask|research` - see `src/agent_core.rs`

## Platform-specific behavior

- `pb tray` only works on macOS; other platforms return error
- Service management (`pb service`) uses launchd on macOS

## Critical conventions

1. **Semantic commits only**: `feat:`, `fix:`, `chore:`, etc.
2. **Update `src/init.rs`** when adding new per-project configuration fields
3. **Web UI requirements**: Must include `viewport-fit=cover`, `env(safe-area-inset-*)` CSS, and PWA meta tags
4. **Web UI tests**: Add or update `webui/src/**/*.test.ts` tests for web UI behavior changes, and run `deno task test:web` before committing
5. **FlashMoe architecture**: When changing FlashMoe data flow, scheduling, expert I/O, Metal kernels, or model-family behavior, keep `docs/flashmoe-architecture-parity-plan.md` aligned with the target architecture and current migration status.

## FlashMoe guidance

- Treat `docs/flashmoe-architecture-parity-plan.md` as the source of truth for FlashMoe work.
- Keep the upstream `danveloper/flash-moe` data flow visible: 4-bit quality is the production target, K=4 routing is the Qwen3.5 baseline, expert I/O should stream through scheduler-owned parallel `pread`, and the OS page cache should be trusted.
- Do not chase isolated FlashMoe microbenchmarks or add Q4-only fast paths without a plan for all supported Qwen MoE variants to use the same data flow, scheduling, and CPU/GPU handoff.
- Do not reintroduce hidden FlashMoe environment toggles, application expert caches, cache migrations, or alternate runtimes without updating the architecture plan first.
- After FlashMoe backend changes, run the narrow smoke at minimum: `target/aarch64-apple-darwin/release/pb harness infer --raw --max-tokens 1 --top-k 1 --temperature 0 "2+2="`. It must exit 0 and print a sensible answer.

## Release process

1. PR merged to `main` → semantic-release tag created
2. GitHub Actions builds macOS arm64 binary from `dist/pb-aarch64-apple-darwin`
