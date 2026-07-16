# Configuration and integrations

pb has user-level defaults and repository-level configuration. The two scopes are deliberately
separate: personal preferences should not silently become project authority, and a checked-in
project file should not contain personal credentials.

## User configuration

Inspect or change common values with the CLI:

```bash
pb config show
pb config get web.listen
pb config set model.temperature 0.1
pb config set model.profile build
```

The user configuration file is `<config-dir>/pb/config.toml`. It covers:

- web listen address, port, and Unix socket path;
- model identifier, directory, context, sampling, and resource defaults;
- global MCP and LSP server definitions;
- an optional separate personal-memory repository.

The web server binds to `127.0.0.1:8311` by default. Changing `web.listen` to `0.0.0.0` exposes the
interface on available networks. pb does not add authentication or TLS merely because the address
changed, so treat non-loopback binding as an explicit trust decision.

## Project configuration

Project-specific files live below `.pb/` in the repository:

| File | Purpose |
| --- | --- |
| `.pb/environment.toml` | Selected local or container execution environment. |
| `.pb/environment.lock` | Resolved environment facts and evidence. |
| `.pb/workspace.toml` | Workspace components, executors, tasks, and affected checks. |
| `.pb/workflow.toml` | Workflow limits and configured task/check policy. |
| `.pb/policy.toml` | User/tool policy rules evaluated for this project. |
| `.pb/mcp.toml` | Project MCP servers and their declared capabilities. |

Not every project needs every file. `pb init` creates the environment foundation and preserves
existing configuration.

## Tool policy

Policy rules are checked in order; the first match wins. A rule can match profiles, tool names, and
nested tool arguments, then allow, deny, or ask. Calls that match no rule are allowed, so a policy
file is a selective control layer rather than a deny-by-default sandbox.

```toml
[[rules]]
name = "ask before publishing commands"
outcome = "ask"
profiles = ["build"]
tools = ["run_command"]
question = "Allow this command?"

[rules.params]
command = { regex = "(^|\\s)(git push|gh pr create)(\\s|$)" }

[[rules]]
name = "block recursive deletes"
outcome = "deny"
profiles = ["build"]
tools = ["run_command"]

[rules.params]
command = { regex = "rm\\s+-rf" }
```

Policy is evaluated in addition to workflow-stage capabilities. A policy can narrow an available
tool call; it cannot make a tool available in a stage where the harness forbids it.

## MCP and LSP integrations

List configured integrations or discover marketplace entries:

```bash
pb integrations list
pb integrations list --marketplace
```

Project MCP entries override global entries with the same name. A project entry with
`disabled = true` removes the corresponding global server for that repository.

Container-backed MCP services declare capabilities explicitly:

```toml
[mcp.servers.example]
container_image = "ghcr.io/example/project-mcp:latest"

[mcp.servers.example.capabilities]
workspace = "read_only" # none | read_only | read_write
network = "none"        # none | session | egress
cache_ids = []

[mcp.servers.example.capabilities.secret_env]
API_TOKEN = "EXAMPLE_API_TOKEN"
```

Workspace, network, cache, and secret capabilities default to none for a container service. Secret
values are resolved from the host environment at launch and are not written to project
configuration, command arguments, or the session ledger.

Host-command MCP servers are less isolated. Container-backed sessions require such integrations to
use a container image so pb can enforce workspace, cache, and network capabilities.

The GitHub setup flow uses a loopback OAuth callback and writes its token below the user
configuration directory with owner-only mode on Unix:

```bash
pb mcp setup github
```

Enabling any remote integration is also a privacy decision. Review [Your data and privacy](data-and-privacy.md)
before granting repository or egress access.
