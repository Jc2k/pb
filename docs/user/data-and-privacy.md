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
| llama.cpp session states | `<platform-cache>/pb/llamacpp-session-v1` or `<storage.cache_dir>/llamacpp-session-v1` | Owner-only, byte-budgeted restart states; may contain prompt tokens and derived attention state. |
| FlashMoe session and shared-prefix states | `<platform-cache>/pb/flashmoe-session-v1` or `<storage.cache_dir>/flashmoe-session-v1` | Owner-only, content-addressed and byte-budgeted KV/MLA/recurrent checkpoints; contains token ids and prompt-derived state. |
| Project configuration | `<repository>/.pb/` | Owned by the repository; may be committed intentionally. |
| Session history, workflow, Goal, and multi-Task checkpoints | Repository-local Git notes under `refs/notes/pb/sessions` | Kept locally until the session is deleted; may include accepted Task plans, budgets, local model qualification digests, active child state, and bounded exact bytes from complete small-file reads for cross-stage evidence; Git notes are not pushed by ordinary branch pushes. |
| Durable project memory | Repository-local `refs/pb/memory` | Evidence-backed Markdown outside the working tree and ordinary branch pushes; bounded to 500 entries and 4 MiB until the ref is changed or removed. |
| Session containers, networks, workspaces, and services | Runtime-managed, session-owned resources | Reconciled and removed at terminal cleanup or expiry. |
| Declared images and caches | Container/model runtime storage | May persist for reuse and garbage collection. |

The optional personal-memory setting points at a separate repository chosen by you. pb does not
silently turn one project repository into cross-project memory.

A native Python dependency-authority entry in user settings can grant one canonical workspace
read-only access to an exact external virtual environment or editable source root. pb copies only
bounded static analyzer inputs into an ephemeral local shadow, never adds those bytes to prompts or
durable events, and rechecks their content identity before a mutation executes. Repository-owned
configuration cannot create this grant.

## When data can leave the machine

Data crosses the local boundary when you choose a feature that requires it:

- `pb pull` downloads model artifacts;
- `pb self update` checks and downloads GitHub release assets;
- public `web_search` or `web_fetch` tools send a query or URL to the built-in search endpoint or
  fetched destination; pb bypasses ambient proxies and accepts only validated public HTTP(S)
  targets, but the request is still a disclosure;
- a remote MCP server receives the arguments supplied to an operator-declared read-only tool;
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
add user authentication or HTTPS for you. On macOS, a non-loopback listener is also advertised as
an HTTP service through Bonjour so supported Wake on Demand infrastructure can find and wake it.

## Reduce external exposure

For an offline-oriented project:

1. Pull model artifacts before disconnecting.
2. Keep `web.listen` on loopback.
3. Disable remote MCP servers in `.pb/mcp.toml`, or keep a minimal audited
   `capabilities.read_only_tools` list.
4. Give container services `network = "none"` unless they need more.
5. Add project policy rules that deny `web_search`, `web_fetch`, and network-capable shell commands.
6. Review configured tasks: a broad shell task is a broad authority grant.

For the architectural reasoning behind these controls, see the [local privacy model](../architecture/local-privacy.md)
and [security model](../architecture/security.md).

## Delete local data

Delete a finished daemon session with `pb queue --delete-session SESSION_ID`. This removes its pb Git
notes ref, including its active/completed Goal checkpoints, objectives, criteria, amendments, and
evidence, plus any accepted Task plan, queue state, budgets, child checkpoints, and results. Stopping
a Goal or Tasks run does not delete or roll back that data; use session deletion when you want the
persisted record removed. `pb self uninstall --delete-data` removes the installed application together with pb's
known data, cache, configuration, state, and logs after confirmation. Review project-owned `.pb/`
files and container runtime storage separately when you need a full project-specific cleanup.
Set `inference.llamacpp_session_cache_enabled` to `false` if prompt-derived llama.cpp state should
not be written to disk; remove the `llamacpp-session-v1` cache directory to discard existing
snapshots. Use `inference.flashmoe_session_cache_enabled` or remove `flashmoe-session-v1` for the
equivalent FlashMoe control. The corresponding `*_session_cache_max_bytes` settings bound each
cache independently. These are user-config values managed by `pb config get/set`, so the daemon and
CLI resolve the same policy regardless of their parent process environment.

`pb cache status` reports only the resolved versioned namespace path, backend/format label, enabled
state, budget, aggregate file/byte count, and oldest-file age. It does not decode checkpoints or
print recovered tokens, tensors, prompts, prompt-derived paths, or tool arguments. `pb cache clean`
lists exact selected namespaces without deleting by default; `--yes` removes only the selected
`llamacpp-session-v1` and/or `flashmoe-session-v1` tree beneath the configured cache root.
