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
- `pb queue --profile build|scout|review|explore|plan|ask <task>` - choose the primary agent profile. Build sessions can spawn focused sub-agents with the same profile names so exploration, planning, Q&A, and review happen in fresh contexts and return only compact summaries to the primary agent.
  The `scout` profile auto-detects the development environment from AGENT.md/AGENTS.md, README files, CI workflows, Dockerfiles, and language manifests instead of requiring a per-project `.pb/environment.toml`; it prefers containers for Linux/deployment-oriented projects and local execution for macOS-specific projects.
- `pb init [--backend apple-containers|local]` - inspect a project and configure it; Apple containers are the default backend.
- `pb env pull <image>` / `pb env build` - configure the default Apple-container-backed project environment.
- `pb env local [--init <cmd>]` - force the project environment to run commands locally from the repository root, useful for macOS-only builds that cannot run inside Apple containers.
- `pb env start|status` - verify or inspect the configured project execution backend.
- the local agent can use built-in workspace editing tools, a shared in-memory `todo(action,id,title,description,status,parent_id,note)` task list, plus read-only `web_search(query)` and `web_fetch(url)` actions for public web research. Configured MCP servers are discovered at session start and exposed to the model as `mcp_<server>_<tool>` tools alongside the built-ins. Build-profile agents can also call `sub_agent(profile,task,max_steps)` to delegate `explore`, `review`, `plan`, `ask`, or nested `build` work without bloating the primary conversation; plan and review sub-agents can add todos, and build sub-agents can complete them or create follow-ups.
- `pb serve` - start a Rust web server, the local Unix-socket RPC endpoint, and the embedded SPA for browser-based sessions.
- `pb service start|stop|restart` - on macOS, control the installed launchd agents for `pb serve` and the menu bar item.

## MCP server tools

Global MCP servers live in `~/.config/pb/config.toml` under `[mcp.servers.<name>]`. Repository-specific servers live in `.pb/mcp.toml` with the same table shape; repo entries override global entries with the same name, and `disabled = true` removes a global server for that repo.

```toml
[mcp.servers.docs]
command = "node"
args = ["/absolute/path/to/docs-mcp-server.js"]
working_directory = "/absolute/path/to/repo"
disabled = false

[mcp.servers.docs.env]
API_TOKEN = "..."
```

When a session starts, pb initializes each enabled stdio MCP server, calls `tools/list`, and adds those schemas to the LLM prompt. Tool calls are routed back through `tools/call` with the model-supplied arguments.

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


## GitHub MCP OAuth setup

`pb mcp setup github` uses a baked GitHub OAuth client ID and a localhost callback; it does not require a GitHub client secret or the `gh` CLI. Configure the release repository with the GitHub Actions secret `PB_GITHUB_CLIENT_ID` so the workflow can compile the client ID into release binaries.

Create a GitHub OAuth App with an authorization callback URL on the configured web port, for example `http://127.0.0.1:8311/auth/github/callback` for the default `web.port`. During setup, `pb` opens GitHub in the browser, waits for the callback on the existing `pb serve` port when it is already running (or starts a temporary listener on that port), exchanges the PKCE authorization code, and stores the resulting token in the user config directory for the GitHub MCP server.

## Tool policy configuration

Projects can define `.pb/policy.toml` to allow, deny, or ask before tool calls, including MCP tools. Rules are evaluated in order; the first matching rule wins, and calls that do not match a rule are allowed.

```toml
[[rules]]
name = "ask before shell in planning"
outcome = "ask" # allow | deny | ask
profiles = ["plan", "explore"]
tools = ["run_command", "mcp_github_*"]
question = "Allow this tool call?"

[rules.params]
command = { contains = "npm install" }

[[rules]]
name = "block recursive deletes in build"
outcome = "deny"
profiles = ["build"]
tools = ["run_command"]

[rules.params]
command = { regex = "rm\\s+-rf" }
```

Parameter filters use dot paths over the full tool argument JSON, including nested objects and array indexes, and support literal values or `{ equals, contains, glob, regex, exists }` matchers.
