# pb

A local coding agent CLI with an optional web front end.

## Commands

- `pb self install` - move the current binary to `~/.local/bin/pb`, install launchd agents for `pb serve` and the menu bar item, and start them after confirmation.
- `pb self uninstall [--delete-data]` - stop the launchd agents, remove their plists, delete the installed binary after confirmation, and optionally delete pb data/cache/config/state/log files.
- `pb self update` - self update from the latest GitHub release binary and restart or start the installed launchd agents when present.
- `pb pull [model]` - pull model blobs with retry, batching, and parallel download.
- `pb queue <task> --workdir /absolute/path/to/repo` - submit a daemon-backed session over the local Unix socket and stream its terminal events.
- `pb queue --session <session-id>` - attach the terminal event stream for any daemon session, including sessions started from the web UI.
- `pb queue --list` - list sessions currently known to the daemon.
- `pb queue --profile build|review|explore|plan|ask <task>` - choose the primary agent profile. Build sessions can spawn focused sub-agents with the same profile names so exploration, planning, Q&A, and review happen in fresh contexts and return only compact summaries to the primary agent.
- `pb init [--backend apple-containers|local]` - inspect a project and configure it; Apple containers are the default backend.
- `pb env pull <image>` / `pb env build` - configure the default Apple-container-backed project environment.
- `pb env local [--init <cmd>]` - force the project environment to run commands locally from the repository root, useful for macOS-only builds that cannot run inside Apple containers.
- `pb env start|status` - verify or inspect the configured project execution backend.
- the local agent can use built-in workspace editing tools plus read-only `web_search(query)` and `web_fetch(url)` actions for public web research. Build-profile agents can also call `sub_agent(profile,task,max_steps)` to delegate `explore`, `review`, `plan`, `ask`, or nested `build` work without bloating the primary conversation.
- `pb serve` - start a Rust web server, the local Unix-socket RPC endpoint, and the embedded SPA for browser-based sessions.
- `pb service start|stop|restart` - on macOS, control the installed launchd agents for `pb serve` and the menu bar item.

## Web UI (`pb serve`)

`pb serve` hosts a Bootstrap-themed SPA that can:

- create sessions,
- list and re-attach to existing sessions from any device,
- stream typed live events over SSE,
- replay recent session history when you reconnect,
- continue existing sessions with follow-up prompts,
- show the same agent progress contract used by the terminal queue adapter,
- report `/api/status` so the macOS menu bar item can show when any session is running.

`pb serve` also listens on a local Unix socket (override with `--socket-path`) so `pb queue` can submit sessions with the same defaults as the web form and attach to running session event streams.

## Build and test

```bash
deno task build:web
cargo test --all-targets
cargo build --release
```

## CI

GitHub Actions workflow (`.github/workflows/ci-release.yml`) builds the web UI assets, runs unit tests, performs semantic release tagging, and then produces an optimized macOS arm64 binary asset.
