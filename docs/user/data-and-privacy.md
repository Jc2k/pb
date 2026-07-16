# Your data and privacy

pb is local-first: model inference, orchestration, the web UI, and normal repository work run on
your machine. It is not unconditionally offline. Downloads, public research, remote MCP servers,
updates, and provider publication are explicit paths across the machine boundary.

## What is stored

| Data | Default location or owner | Lifecycle |
| --- | --- | --- |
| User settings and project registry | `<config-dir>/pb/` | Kept until changed or removed. |
| GitHub OAuth token | `<config-dir>/pb/github-token` | Kept locally; owner-only mode on Unix. |
| Model weights | `$XDG_DATA_HOME/pb/models` or `~/.local/share/pb/models` | Kept for reuse. |
| Project configuration | `<repository>/.pb/` | Owned by the repository; may be committed intentionally. |
| Session history and workflow checkpoints | Repository-local Git notes under `refs/notes/pb/sessions` | Kept locally until the session is deleted; Git notes are not pushed by ordinary branch pushes. |
| Session containers, networks, workspaces, and services | Runtime-managed, session-owned resources | Reconciled and removed at terminal cleanup or expiry. |
| Declared images and caches | Container/model runtime storage | May persist for reuse and garbage collection. |

The optional personal-memory setting points at a separate repository chosen by you. pb does not
silently turn one project repository into cross-project memory.

## When data can leave the machine

Data crosses the local boundary when you choose a feature that requires it:

- `pb pull` downloads model artifacts;
- `pb self update` checks and downloads GitHub release assets;
- public `web_search` or `web_fetch` tools send a query or URL to their configured service;
- a remote MCP server receives the arguments supplied to one of its tools;
- a container or service with `network = "egress"` can reach external networks;
- GitHub OAuth contacts GitHub and stores the returned token locally;
- any shell command or configured task can perform network activity if its execution environment
  and policy permit it.

The workflow capability matrix controls when research and command tools are exposed, and policy can
ask or deny particular calls. Those controls do not make arbitrary host commands a network sandbox.

## Keep the web interface local

The default listen address is loopback-only. Leave it at `127.0.0.1` unless you have a specific
reason to make pb reachable from another device.

```bash
pb config set web.listen 127.0.0.1
pb config set web.port 8311
```

If you bind to `0.0.0.0`, place pb behind a trusted network boundary. The current server does not
add user authentication or HTTPS for you.

## Reduce external exposure

For an offline-oriented project:

1. Pull model artifacts before disconnecting.
2. Keep `web.listen` on loopback.
3. Disable remote MCP servers in `.pb/mcp.toml`.
4. Give container services `network = "none"` unless they need more.
5. Add project policy rules that deny `web_search`, `web_fetch`, and network-capable shell commands.
6. Review configured tasks: a broad shell task is a broad authority grant.

For the architectural reasoning behind these controls, see the [local privacy model](../architecture/local-privacy.md)
and [security model](../architecture/security.md).

## Delete local data

Delete a finished daemon session with `pb queue --delete-session SESSION_ID`. This removes its pb Git
notes ref. `pb self uninstall --delete-data` removes the installed application together with pb's
known data, cache, configuration, state, and logs after confirmation. Review project-owned `.pb/`
files and container runtime storage separately when you need a full project-specific cleanup.
