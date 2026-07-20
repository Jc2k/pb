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

- web listen address, port, Unix socket path, and the macOS work-queue sleep preference;
- model identifier, directory, context, sampling, and resource defaults;
- global MCP and LSP server definitions;
- an optional separate personal-memory repository.

The web server binds to `127.0.0.1:8311` by default. Changing `web.listen` to `0.0.0.0` exposes the
interface on available networks. pb does not add authentication or TLS merely because the address
changed, so treat non-loopback binding as an explicit trust decision.

## Keep macOS awake while pb works

**Shipped.** On macOS, `web.prevent_sleep_while_working` defaults to `true`. While a queued session
is actually running, the service uses the native IOKit power-management API to hold a
`PreventUserIdleSystemSleep` assertion. It releases that assertion by ID when no queued work remains
or when the active session pauses for a user answer, and reacquires it when processing resumes. No
helper process is launched. The assertion does not prevent the display from turning off.

Use **Settings → Prevent sleep while working** in the web interface to change the preference. A
change takes effect immediately, including during an active task, and is then saved to the user
configuration. The equivalent CLI commands are:

```bash
pb config set web.prevent_sleep_while_working true
pb config set web.prevent_sleep_while_working false
```

## Wake a Mac by opening pb

**Shipped.** An installed macOS service declares its configured HTTP listener as a
[launchd socket](https://developer.apple.com/documentation/xpc/launch_activate_socket). launchd
opens the socket before pb starts, publishes a network-visible listener as `_http._tcp` through
Bonjour, and hands the descriptor to pb. A direct `pb serve` development process instead binds the
address itself and uses the native DNS-SD API to publish the same Bonjour service. It does not need
a LaunchAgent, so the web interface and Bonjour path remain testable during development.

Loopback is deliberately not advertised. To opt into LAN access and wake-on-HTTP, expose the
otherwise unauthenticated web server and refresh an installed service after changing its socket:

```bash
pb config set web.listen 0.0.0.0
pb self refresh-service # omit this for a direct `pb serve`
```

Then enable macOS [**Wake for network access**](https://support.apple.com/en-gb/guide/mac-help/mh27905/mac)
and open `http://<mac-hostname>.local:8311/` from another device on the local network. A different
configured port replaces `8311` in both the launchd socket and URL.

**Configurable host dependency.** pb registers the HTTP service with the macOS mechanisms that can
participate in Wake on Demand; it cannot guarantee that a particular sleeping Mac will wake.
Success still depends on supported network hardware, the macOS power setting, laptop lid and power
state, and a reachable Bonjour Sleep Proxy when the client is on another subnet. Binding to
`0.0.0.0` also exposes task history and controls without adding authentication or TLS, so use only a
trusted network or put pb behind an authenticated proxy.

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

DeepSeek V4 Flash is deliberately excluded from that reuse. Its four hyperconnection
streams plus raw, compressed, and indexer KV form one typed state that the current snapshot format
cannot represent. Every ordinary DeepSeek request therefore starts from a freshly reset
request-scoped state. Supplying `--session-id` is a named capability error; pb does not silently
restore a partial Qwen/GLM snapshot, downgrade to uncached session execution, or advertise cached
prompt tokens for it. The managed agent backend recognizes the load-resolved capability and sends
ordinary request-scoped turns, preserving multi-turn tool execution without claiming cache reuse.

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

## Qwen3-Coder-Next with FlashMoe

**Configurable.** On Apple Silicon, the default model is the native affine-Q4 checkpoint:

```bash
pb pull
pb config set model.model hf://mlx-community/Qwen3-Coder-Next-4bit
```

The shorter `qwen3-coder-next` name resolves to the same source. Pull builds the normal FlashMoe
dense store and fixed whole-expert packs. The one-time import keeps the large MLX affine-Q4
matrices quantized and expands the checkpoint's small affine-int8 routers and shared gates to
resident BF16. At load, pb compares the complete 48-layer, 512-expert
corpus plus transient/session headroom with the Metal working-set limit. A fitting Mac keeps every
expert mapped and resident for the graph lifetime; a larger corpus or smaller memory budget uses
the existing parallel positioned-read scheduler and OS page cache. There is no partial expert
cache, first-use retention, or second scheduler.

The hidden inference/benchmark harness can lower the Metal limit for experiments. That limit is
installed before graph preparation, so it participates in the resident-versus-positioned-read
decision rather than changing accounting after the model has already been placed.

Qwen3-Coder-Next retains its checkpoint top-10 routing and hybrid attention schedule. It is a
non-thinking model, so pb disables thinking for both prompt measurement and generation rather than
spending the output budget on unsupported reasoning markers. It can return several independent
native tool calls in one assistant turn; pb validates the full batch and runs parallel-safe calls
together before asking the model again. An explicit Qwen GGUF path continues to use llama.cpp.
Once a native FlashMoe source has been selected, a cache or graph-load failure is reported directly
instead of silently running a different backend.

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

## DeepSeek V4 Flash with FlashMoe

**Shipped for the pinned profile and bounded live sessions.** On Apple Silicon, pb recognizes exactly the
published DeepSeek V4 Flash IQ2_XXS/Q2_K checkpoint:

```bash
pb pull hf://antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
pb config set model.model hf://antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Pull rejects other repository files and validates the complete 43-layer metadata, tensor, dtype,
shape, quantization, compression-schedule, expert, and JoyAI tokenizer contract before atomically
publishing a source-independent FlashMoe cache. Resident tensors stay in one mmap-backed store.
Each layer's 256 routed experts is split into a fixed, page-aligned whole-expert slot. Decode asks
the shared scheduler to stream the selected six from SSD. Layer-major prompt prefill instead reads
each sorted unique expert selected anywhere in that layer's prompt once, using at most eight
parallel positioned reads and request-scoped Metal staging. Neither path creates an application
expert cache; the operating system page cache remains authoritative. The published source is 86.72
GB; retaining both it and the
prepared resident/expert cache used about 162 GiB in the 2026-07-18 validation run, so check free
space before pulling.

Load resolves the complete graph and required fused Metal kernels before the first token. Prompts
below 32 tokens retain the exact token command; prompts of 32 tokens or more use the graph's
layer-major Metal prefill command. A live session with an exact cached token prefix restores its
complete Metal state and applies the same calculation to the appended suffix. This fixed
matrix-tile calculation is not an error fallback.
Top-6 is
part of the checkpoint contract: `--flashmoe-active-experts` cannot reduce it. There is no llama.cpp
fallback, CPU component fallback, DS4 process, hidden top-2 mode, or alternate generation loop.
DeepSeek is text-only in this profile. Exact prompt-prefix reuse is memory-only and LRU-bounded to
two logical sessions with two exact checkpoints each. Structured stages retain the first rendered
message as their stable base plus the latest complete prompt; raw sessions retain the prompt plus
an evaluated generated-token head. Stage identity also binds the system prompt and tool schema, so
tool results, validation corrections, and truncation retries can extend the stable base without
sharing state across different workflow stages. A reused session whose rendered tokens do not
extend any retained checkpoint fails with `DeepSeek V4 session prefix mismatch`;
there is no silent fresh-prefill fallback. DeepSeek checkpoints are not written to the Qwen/GLM
disk-session format. The implementation has local graph, routing, ABI, all-target, and
Metal shader compilation evidence plus a complete published-checkpoint cache build and real-model
Metal load/prefill/decode. A raw arithmetic smoke emitted `4`; all four continuation cases enforced
by the pinned upstream reference match exactly, including a 3,844-token prompt that crosses the
top-512 indexed-attention frontier; and a real two-turn DSML request executed a native tool call.
On the validation M4 Max, the final 3,844-token prefill measured 153.8 tok/s versus 234.3 tok/s for
the pinned ds4 one-expert cold SSD-streaming control. The same short Italian control measured 5.36
prefill and 8.23 decode tok/s in pb versus 3.62 and 6.72 in ds4.
The upstream reference excludes its `long_memory_archive` case because the hosted API and official
graph disagree after the official Hadamard and FP4-indexer path, so pb does not count that excluded
case as a supported-graph failure or as parity evidence.

## Project configuration

Project-specific files live below `.pb/` in the repository:

| File | Purpose |
| --- | --- |
| `.pb/environment.toml` | Selected local or container execution environment. |
| `.pb/environment.lock` | Resolved environment facts and evidence. |
| `.pb/workspace.toml` | Workspace components, executors, tasks, and affected checks. |
| `.pb/workflow.toml` | Workflow limits and configured task/check policy. |
| `.pb/goal.toml` | Durable Goal ceilings; cannot enable Auto, automatic continuation, or publication. |
| `.pb/policy.toml` | User/tool policy rules evaluated for this project. |
| `.pb/mcp.toml` | Project MCP servers and their declared capabilities. |

Not every project needs every file. `pb init` creates or preserves the environment, workspace,
strict-workflow, and durable-Goal foundations.

## Goal policy

**Shipped.** `.pb/goal.toml` is a versioned, hashed project ceiling. The default created by
`pb init` is equivalent to:

```toml
version = 1

[limits]
max_milestones = 8
max_workflows = 12
total_model_invocations = 120
total_generated_tokens = 100000
wall_time_minutes = 120
```

The Goal setup sheet defaults to the smaller Standard allowance (5 milestones, 8 workflows, 80
model invocations, 60,000 generated tokens, and 90 minutes). Compact narrows it further; Extended
uses the default project ceiling. Advanced and API-supplied limits must remain inside both this
document and pb's built-in hard ceilings. A malformed, unknown-version, unknown-field, zero, or
expanding configuration fails closed.

This file is repository-owned narrowing data. It deliberately has no fields for Auto activation,
automatic continuation, path/tool/network expansion, credentials, or publication. Those choices
remain user/session authority and Goal mode always stops at local evidence.

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
