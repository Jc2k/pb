# AGENTS.md

## Commands

- `deno task build:web` - Build web UI assets (required before `cargo build`)
- `cargo test --all-targets` - Run tests
- `cargo build --release --target aarch64-apple-darwin` - Build release binary for macOS arm64

## Architecture

- **Entry point**: `src/main.rs` → `pb::run(Cli::parse()).async`
- **CLI commands** defined in `src/lib.rs`: `serve`, `queue`, `self`, `tray`, `projects`, `env`, `service`, `init`
- **Web UI**: React app in `webui/` served by Rust backend; builds to `webui/dist/`
- **Agent profiles**: `build|review|explore|plan|ask` - see `src/agent_core.rs`

## Platform-specific behavior

- `pb tray` only works on macOS; other platforms return error
- Service management (`pb service`) uses launchd on macOS

## Critical conventions

1. **Semantic commits only**: `feat:`, `fix:`, `chore:`, etc.
2. **Update `src/init.rs`** when adding new per-project configuration fields
3. **Web UI requirements**: Must include `viewport-fit=cover`, `env(safe-area-inset-*)` CSS, and PWA meta tags

## Release process

1. PR merged to `main` → semantic-release tag created
2. GitHub Actions builds macOS arm64 binary from `dist/pb-aarch64-apple-darwin`
