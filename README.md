# pb

pb is a local-first coding agent for getting from an idea to a checked commit without assembling a
model runner, project sandbox, review loop, and user interface yourself. It runs the coding model on
your Mac, keeps project context and session history under your control, and comes with sensible
defaults for the ordinary work around a change: inspect, plan, edit, check, review, and commit.

Use it from a friendly web interface or stay in the terminal. Tell pb what you want to build or fix;
it works inside your repository, shows its progress, and leaves you with the evidence behind the
result. You can start with the defaults and introduce project-specific environments, policies, MCP
tools, or model settings only when you need them.

## Why pb

- **Local first.** Inference, repository access, history, and the control plane live on your machine
  by default. Integrations that can send data elsewhere are explicit and documented.
- **Less setup.** pb can inspect a repository, choose a practical execution environment, pull a local
  model, and provide both terminal and web entry points.
- **Sensible guard rails.** The workflow owns planning, checks, fresh-context review, completion, and
  commit creation instead of trusting a model to declare its own work done.
- **Clear when things go wrong.** Sessions are durable and reattachable, and results distinguish a
  completed change from no-change, blocked, and failed outcomes.

pb currently targets Apple silicon Macs. Start with the
[getting-started guide](https://jc2k.github.io/pb/user/getting-started.html), or see the short path
below.

```bash
pb self install
pb pull
cd /path/to/project
pb init
pb queue "Add validation for empty project names" --workdir "$PWD"
```

The user guide and architecture documentation are published at
[jc2k.github.io/pb](https://jc2k.github.io/pb/). The site explores pb's workflows, security model,
local privacy boundaries, and contracts with the user; its Markdown source lives in [`docs/`](docs/README.md).

## Commands

- `pb self install` - move the current binary to `~/.local/bin/pb`, install launchd agents for `pb serve` and the menu bar item, and start them after confirmation.
- `pb self uninstall [--delete-data]` - stop the launchd agents, remove their plists, delete the installed binary after confirmation, and optionally delete pb data/cache/config/state/log files.
- `pb self update` - self update from the latest GitHub release binary and restart or start the installed launchd agents when present.
- `pb pull [model]` - pull model blobs with retry, batching, and parallel download.
- `pb queue <task> --workdir /absolute/path/to/repo` - submit a daemon-backed session over the local Unix socket and stream its terminal events.
- `pb queue --session <session-id>` - attach the terminal event stream for any daemon session, including sessions started from the web UI.
- `pb queue --list` - list sessions currently known to the daemon.
- `pb queue --profile build|scout|review|explore|plan|ask <task>` - choose the primary agent profile. Sessions can ask read-only advisory profiles (`explore`, `review`, `plan`, `ask`, `research`, or `monitor`) for bounded fresh-context input. Mutating `build` and `scout` work remains with the primary agent and cannot be delegated.
  The `scout` profile auto-detects the development environment from AGENT.md/AGENTS.md, README files, CI workflows, Dockerfiles, and language manifests instead of requiring a per-project `.pb/environment.toml`; it prefers containers for Linux/deployment-oriented projects and local execution for macOS-specific projects.
- `pb init [--backend apple-containers|local]` - inspect a project and configure it; Apple containers are the default backend.
- `pb env pull <image>` / `pb env build` - configure the default Apple-container-backed project environment.
- `pb env local [--init <cmd>]` - force the project environment to run commands locally from the repository root, useful for macOS-only builds that cannot run inside Apple containers.
- `pb env start|status` - verify or inspect the configured project execution backend.
- the local agent can use built-in workspace editing tools, a shared in-memory `todo(action,id,title,description,status,parent_id,note)` task list, plus read-only `web_search(query)` and `web_fetch(url)` actions for public web research. Configured MCP servers are discovered at session start and exposed to the model as `mcp_<server>_<tool>` tools alongside the built-ins. Agents can call `sub_agent(profile,task,max_steps)` for bounded read-only advice without bloating the primary conversation; advisory work runs against an isolated workspace snapshot, cannot delegate again, and returns a structurally truncated result.
- `pb serve` - start a Rust web server, the local Unix-socket RPC endpoint, and the embedded SPA for browser-based sessions.
- `pb service start|stop|restart` - on macOS, control the installed launchd agents for `pb serve` and the menu bar item.

## Project conversation and delivery

The web composer separates **Discuss** from **Build**. Discuss is conversational and read-only: it
supports brainstorming, rubber-ducking, explanation, repository inspection, and bounded advisory
teammates without creating a branch or granting mutation tools. A discussion may propose a Build,
but only an explicit Build selection starts delivery.

Build uses a harness-owned workflow rather than asking the model to police itself. pb requires a
structured plan, a fresh-context plan critique, implementation, affected configured checks, an
isolated fresh-context code critique, bounded repair when needed, and finally a task-owned managed
commit. Stage-specific structured submissions and content fingerprints advance the workflow;
model prose, a teammate response, and a successful shell command do not. `run_command` remains
available only during implementation and repair as a journaled escape hatch.

`pb queue <task>` is an explicit Build request. The web interface can continue ordinary discussion
after Ready, No change, or a failed/blocked workflow without silently starting another delivery.
Read-only advisory profiles may compact investigation context, but build and scout cannot be
delegated and no advisory call can advance a workflow stage.

A successful delta-bearing Build exposes a durable evidence bundle that binds the managed commit
OID to the accepted plan, fresh code review, and selected check evidence. The web UI identifies the
reviewed commit as ready to publish, and `pb harness agent` records the bundle digest in its audit.
This is evidence only: pb does not push, open a PR/MR, wait for CI, or process provider feedback as
part of local delivery. Those operations require the separate approval-bearing publication
workflow described in
[`docs/external-publication-workflow-follow-on.md`](docs/external-publication-workflow-follow-on.md).

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

`pb serve` also listens on a local Unix socket configured with `web.socket_path`; `pb queue` and project commands use the same setting unless their `--socket-path` argument overrides it. The platform fallback uses `$XDG_RUNTIME_DIR/pb.sock` when available and `/tmp/pb-<uid>.sock` otherwise.

## Build and test

```bash
deno task build:web
deno task test:docs
cargo test --all-targets
cargo build --release
```

## CI

GitHub Actions workflow (`.github/workflows/ci-release.yml`) builds the web UI assets, runs unit and
documentation tests, performs semantic release tagging, deploys the static documentation to GitHub
Pages from `main`, and produces an optimized macOS arm64 binary asset for new releases.

### macOS code signing without a paid Apple Developer account

The release workflow supports an optional self-signed macOS code signing identity. This gives the release binary a Mach-O code signature without requiring Xcode or a paid Apple Developer account. It does not notarize the binary, and macOS Gatekeeper can still warn users when they download the release asset from the internet.

To create and export a self-signed identity using built-in macOS tools:

1. Open **Keychain Access**.
2. Choose **Keychain Access → Certificate Assistant → Create a Certificate...**.
3. Set **Name** to something recognizable, such as `pb self-signed release`.
4. Set **Identity Type** to **Self Signed Root**.
5. Set **Certificate Type** to **Code Signing**.
6. Create the certificate in the **login** keychain.
7. In Keychain Access, select the certificate and its private key, then choose **File → Export Items...**.
8. Save the export as `pb-codesign.p12` and protect it with a strong password.
9. Convert it to base64 for GitHub Actions:

   ```bash
   base64 -i pb-codesign.p12 | pbcopy
   ```

10. Add these repository secrets in GitHub:
    - `MACOS_CODESIGN_CERT_P12_BASE64`: the base64 text copied in the previous step
    - `MACOS_CODESIGN_CERT_PASSWORD`: the `.p12` export password

If `MACOS_CODESIGN_CERT_P12_BASE64` is not configured, the workflow falls back to ad-hoc signing with `codesign --sign -`, which also requires no Apple Developer account. The workflow also falls back to ad-hoc signing when an imported `.p12` is not usable for code signing.

If GitHub Actions reports that no code signing identity was found after importing the certificate, recreate the `.p12` and verify these details before updating the secret:

- The certificate was created with **Certificate Type** set to **Code Signing**. A generic SSL, S/MIME, or website certificate can import successfully but will not appear in `security find-identity -p codesigning`.
- The export includes both the certificate and its private key. In Keychain Access, expand the certificate row and select the certificate together with the nested private key before choosing **File → Export Items...**.
- The base64 secret is generated from the exported `.p12`, not from a `.cer`, `.pem`, or certificate-only export.
- Locally, the exported file should produce a code signing identity after import, for example `security find-identity -v -p codesigning <keychain>` should list the certificate name.

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
