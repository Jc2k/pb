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

## Inference session cache

**Shipped.** Both local text backends compare exact rendered token prefixes before reusing model
state. llama.cpp keeps one live context for the logical run and saves restartable state so a later
process can resume without prefilling the unchanged prefix. The cache key includes the model file
identity, context size, and a hash of the session id; exact token comparison invalidates changed
system instructions, tools, templates, messages, and compaction.

States live below the platform cache directory at `pb/llamacpp-session-v1/`. Set `PB_CACHE_DIR` to
move pb's cache root, or set `PB_LLAMA_SESSION_CACHE=off` to disable disk snapshots while retaining
the in-process context. `PB_LLAMA_SESSION_CACHE_MAX_BYTES` sets its byte budget; the default is 8
GiB. pb writes replacements through a temporary owner-only cache directory. These files can be
large because they contain the model's evaluated attention state and associated prompt tokens.

FlashMoe keeps up to two checkpoints for each of the two most recently used logical sessions in
memory by default: the safe rendered-prompt boundary and an evaluated generated-token head. Set
`PB_FLASHMOE_MEMORY_SESSIONS` to another positive count. If canonical chat rendering preserves
generated tokens exactly, the next pass resumes from the generated head; otherwise it falls back to
the safe boundary. A stable first-system-message prefix is also content-addressed and may be shared
by other sessions using the same model, tokenizer, template, tool schema, and system tokens.
Full-attention KV, compressed MLA KV, linear-attention conv/SSM state, final hidden state, and token
ids are stored together, so hybrid Qwen and GLM checkpoints restore the state their forward graph
actually needs.

The web/service path retains loaded FlashMoe runtimes across managed turns. It retains up to two
idle models by default and reaps an unused runtime after 15 minutes. Set
`PB_FLASHMOE_RESIDENT_MODELS` or `PB_FLASHMOE_IDLE_SECONDS` to change those process-memory bounds.

FlashMoe restart snapshots live below `pb/flashmoe-session-v1/`, or below
`$PB_CACHE_DIR/flashmoe-session-v1/`. They are model-fingerprinted, content-addressed, owner-only,
checksummed, and pruned to an 8 GiB byte budget. Set `PB_FLASHMOE_SESSION_CACHE_MAX_BYTES` to a
different positive byte count or `PB_FLASHMOE_SESSION_CACHE=off` to retain memory reuse without
writing prompt-derived state. Only the canonical prompt boundary is written for a session; the
speculative generated head stays in memory, avoiding a second large durable write per turn. A
checkpoint larger than the whole budget is skipped rather than making generation fail.

For llama.cpp text sessions, pb probes the requested context after loading an accelerated model.
If Metal can load the weights but cannot create that context, pb reports the degradation and
reloads the model CPU-only instead of failing before the first token. A CPU-only context failure is
still terminal rather than being hidden behind retries.

## GLM-5.2 with FlashMoe

**Configurable.** On Apple Silicon, pb can import GLM-5.2 checkpoints and run the baseline decoder
through FlashMoe. The preferred source is the indexed MLX MXFP4 checkpoint:

```bash
pb pull hf://mlx-community/GLM-5.2-mxfp4
pb config set model.model hf://mlx-community/GLM-5.2-mxfp4
```

Pull recognizes MLX MXFP4 tensors as packed E2M1 values with one E8M0 scale per 32 values. It
decodes one row at a time and requantizes it once into FlashMoe's canonical affine Q4 cache; the
runtime and expert scheduler never depend on the source checkpoint's container or quantization.
When a checkpoint publishes `chat_template.jinja` separately from `tokenizer_config.json`, pull
preserves it alongside the tokenizer and FlashMoe uses it for non-raw chat generation. `--raw`
deliberately bypasses that model-specific conversation framing and is intended for completion and
diagnostic prompts rather than chat-quality validation.
The Colibri checkpoint remains a compatible alternative:

```bash
pb pull hf://jlnsrk/GLM-5.2-colibri-int4
pb config set model.model hf://jlnsrk/GLM-5.2-colibri-int4
```

That adapter imports packed offset-binary int4 tensors and preserves Colibri's int8 input/output
matrices as resident BF16. Both source snapshots are very large. Plan for enough space for the
source and runtime cache during conversion, or pass `--flashmoe-prune-source-shards` to delete
source safetensor shards after the runtime cache is complete. The indexed MLX checkpoint has
completed full-size cache construction plus deterministic prefill and decode validation. Support
remains Configurable because it requires an explicitly selected local checkpoint and Apple Silicon.

The baseline supports requests through the checkpoint's `index_topk` boundary (2,048 tokens in the
published snapshot). Longer contexts require GLM's DSA indexer and are rejected explicitly; DSA and
the optional MTP speculative head are not currently implemented.

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

On macOS, Safari Technology Preview 247 or newer includes its own local MCP server. pb does not
ship a separate WebDriver browser-control layer. Configure the preview server for the current
project with:

```bash
pb mcp setup safari
```

The command points the project MCP entry at Safari Technology Preview's `safaridriver --mcp`.
Before starting an agent, enable **Developer > Enable remote automation and external agents** in
Safari Technology Preview. Use `--driver-path` if the application is installed somewhere other
than `/Applications`. Because this is a host-command MCP server, use it with a local execution
environment rather than a container-backed session.

Enabling any remote integration is also a privacy decision. Review [Your data and privacy](data-and-privacy.md)
before granting repository or egress access.
