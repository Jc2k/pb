# pb

A local coding agent CLI with an optional web front end.

## Commands

- `pb self update` - self update from the latest GitHub release binary.
- `pb pull [model]` - pull model blobs with retry, batching, and parallel download.
- `pb agent <task> --model-dir /absolute/path/to/models --workdir /absolute/path/to/repo` - run the local in-process coding agent with streamed output and follow-up prompts.
- the local agent can use built-in workspace editing tools plus read-only `web_search(query)` and `web_fetch(url)` actions for public web research.
- `pb serve` - start a Rust web server and serve the embedded SPA for browser-based sessions.

## Web UI (`pb serve`)

`pb serve` hosts a Bootstrap-themed SPA that can:

- create sessions,
- list and re-attach to existing sessions from any device,
- stream typed live events over SSE,
- replay recent session history when you reconnect,
- continue existing sessions with follow-up prompts,
- show the same agent progress contract used by the CLI adapter.

## Build and test

```bash
deno task build:web
cargo test --all-targets
cargo build --release
```

## CI

GitHub Actions workflow (`.github/workflows/ci-release.yml`) builds the web UI assets, runs unit tests, performs semantic release tagging, and then produces an optimized macOS arm64 binary asset.
