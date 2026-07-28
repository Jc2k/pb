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
- intrinsic deterministic controller-action safety policy;
- storage roots and inference session-cache policy;
- FlashMoe in-memory session and resident-runtime bounds;
- global MCP and LSP server definitions;
- an optional separate personal-memory repository.

CLI, web, and marketplace mutations take a cross-process lock, re-read the latest configuration,
and atomically replace the file. Concurrent changes to different settings or integrations therefore
do not silently overwrite one another. Project MCP marketplace writes and the GitHub/Safari setup
commands use the same pattern for `.pb/mcp.toml`. Integration removal validates that the selected
entry is container-backed inside that transaction, so trying to remove a host-command server
through the container-integration UI cannot delete it before returning an error.

The web server binds to `127.0.0.1:8311` by default. Changing `web.listen` to `0.0.0.0` exposes the
interface on available networks. pb does not add authentication or TLS merely because the address
changed, so treat non-loopback binding as an explicit trust decision.

The daemon and all CLI clients resolve the Unix socket in the same order: a command-specific
`--socket-path`, `web.socket_path`, `$XDG_RUNTIME_DIR/pb.sock`, then `/tmp/pb-<uid>.sock`. The final
fallback uses the numeric user ID rather than the mutable `USER` environment variable.

pb-owned workspaces, leases, and managed cache records use the platform-local data directory by
default. Set `storage.state_dir` to a non-empty absolute path to relocate them.

## Deterministic controller actions

**Shipped, intrinsic.** pb performs a narrowly defined set of uniquely determined local actions
without spending a model turn on choosing them. This is workflow behavior, not a preference: there
is no `agent.action_elision` or `agent.controller_delete_elision` setting, request field, environment
toggle, or web switch.

Eligible complete reads, diagnostic ranges, and review inspections are supplied to the local model
as an explicit pb-owned context block. They are never rendered as a fabricated assistant tool call.
Missing, unreadable, binary, symlinked, stale, oversized, or context-ineligible files fall back to
ordinary model tools. Automatic deletion is limited to a unique accepted-plan deletion whose file
or symlink is tracked, clean, unchanged, fresh, bounded, allowed, and Git-recoverable; directories,
untracked or adopted content, dirty files, and stale fingerprints are rejected.

Older configuration files containing the retired `[agent]` keys still load. pb ignores those values
and omits the section the next time it saves the configuration. `pb config get/set` rejects the
retired keys rather than implying that they control current workflow safety.

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

States live below the platform cache directory at `pb/llamacpp-session-v1/`. Configure an absolute
`storage.cache_dir` to move pb's inference-cache root. Set
`inference.llamacpp_session_cache_enabled` to `false` to disable disk snapshots while retaining the
in-process context. `inference.llamacpp_session_cache_max_bytes` sets its byte budget; the default
is 8 GiB. pb writes replacements through a temporary owner-only cache directory. These files can
be large because they contain the model's evaluated attention state and associated prompt tokens.

```bash
pb config set storage.cache_dir /absolute/path/to/pb-cache
pb config set inference.llamacpp_session_cache_enabled false
pb config set inference.llamacpp_session_cache_max_bytes 8589934592
```

FlashMoe keeps up to two checkpoints for each of the two most recently used logical sessions in
memory by default: the safe rendered-prompt boundary and an evaluated generated-token head. Set
`flashmoe.memory_sessions` to another positive count. If canonical chat rendering preserves
generated tokens exactly, the next pass resumes from the generated head; otherwise it falls back to
the safe boundary. A stable first-system-message prefix is also content-addressed and may be shared
by other sessions using the same model, tokenizer, template, tool schema, and system tokens.
Full-attention KV, compressed MLA KV, linear-attention conv/SSM state, final hidden state, and token
ids are stored together, so hybrid Qwen and GLM checkpoints restore the state their forward graph
actually needs. JSON-constrained Task-artifact calls have no logical session or reusable generated
head, but they publish the same validated stable-system prefix after prefill. The JSON constraint,
dynamic Task evidence, and generated artifact remain outside that checkpoint and are rebuilt for
every call.

Shared prompt roots use an LRU bounded by bytes rather than by a guessed stage count. The default
in-memory prompt-root budget is 4 GiB; set `flashmoe.memory_prompt_root_max_bytes` to another
positive byte count. When a dirty root crosses that budget, pb attempts the configured durable
cache write before releasing the memory copy. Durable successes and failures are reported in the
session metrics; a failed write is discarded under the memory bound and causes a truthful fresh
prefill if that root is needed again.

DeepSeek V4 Flash is deliberately excluded from that reuse. Its four hyperconnection
streams plus raw, compressed, and indexer KV form one typed state that the current snapshot format
cannot represent. Every ordinary DeepSeek request therefore starts from a freshly reset
request-scoped state. Supplying `--session-id` is a named capability error; pb does not silently
restore a partial Qwen/GLM snapshot, downgrade to uncached session execution, or advertise cached
prompt tokens for it. The managed agent backend recognizes the load-resolved capability and sends
ordinary request-scoped turns, preserving multi-turn tool execution without claiming cache reuse.

The web/service path retains loaded FlashMoe runtimes across managed turns. It retains up to two
idle models by default and reaps an unused runtime after 15 minutes. Set
`flashmoe.resident_models` or `flashmoe.idle_seconds` to change those process-memory bounds.

FlashMoe restart snapshots live below `pb/flashmoe-session-v1/` in the configured inference-cache
root. They are model-fingerprinted, content-addressed, owner-only, checksummed, and pruned to an 8
GiB byte budget. Set `inference.flashmoe_session_cache_max_bytes` to a different positive byte count
or `inference.flashmoe_session_cache_enabled` to `false` to retain memory reuse without writing
prompt-derived state. Only the canonical prompt boundary is written for a session; the speculative
generated head stays in memory, avoiding a second large durable write per turn. A checkpoint larger
than the whole budget is skipped rather than making generation fail.
Checkpoint and manifest reads reject symlinks and oversized records. Pruning removes a checkpoint
from session manifests before deleting it, so a published manifest does not retain a dangling
reference. A successfully validated checkpoint or session-manifest restore refreshes its disk LRU
recency, so roots that remain useful across tasks survive ahead of newer but unused state. Updating
that retention hint is best effort: a valid read-only cache hit remains usable even when its
timestamp cannot be changed.

Use `pb cache status` to inspect the resolved llama.cpp and FlashMoe versioned session namespaces,
their configured budgets, file counts, byte totals, and oldest-file age without decoding cached
tokens or tensors. `pb cache clean` is a dry run. Select `--backend llama-cpp` or
`--backend flash-moe`, inspect the exact versioned path, then pass `--yes` to remove only that
namespace.

```bash
pb config set flashmoe.memory_sessions 2
pb config set flashmoe.memory_prompt_root_max_bytes 4294967296
pb config set flashmoe.resident_models 2
pb config set flashmoe.idle_seconds 900
pb config set inference.flashmoe_session_cache_enabled false
pb config set inference.flashmoe_session_cache_max_bytes 8589934592
pb cache status
pb cache clean --backend flash-moe
```

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
together before asking the model again. Strict native stages constrain function names and the
supported argument-schema subset while tokens are sampled. Mutation calls additionally use a
bounded controller snapshot and the shared syntax/canonical-patch completion gate, then validate the
completed result again at the executor boundary. This is automatic for qualified FlashMoe profiles;
there is no environment or configuration switch.

When a real local backend exposes a path that may edit Rust in a Cargo project, pb prepares a pinned
native Rust semantic world before the model starts. The first request can therefore spend noticeable
time loading Cargo metadata, the sysroot, dependencies, and HIR; exact later requests reuse it, and
ordinary edits to existing `.rs` files refresh it incrementally. Cargo/configuration/dependency or
file-topology changes rebuild before the next inference. Preparation is offline and never downloads
dependencies. Missing local dependency sources or a workspace that changes while loading fail the
model turn rather than silently dropping the Rust layer. This behavior is automatic and has no
environment toggle. Its guarantee is intentionally narrow: exact qualified import and selected
literal/call-shape contradictions can be rejected, while build-script/procedural-macro and deeper
type/ownership facts remain unknown. If cancellation arrives during this preparation, pb will not
start the model afterward. Cold loads run in a request-independent local worker; the initiating
request and queued requests poll cancellation every 100 ms, and only one cold Rust load runs at a
time. Cancelling stops the request wait, but the embedded analyzer cannot interrupt every
in-progress Cargo, VFS, or HIR query, so the exact local build may finish in the background and warm
the bounded process cache.

A compatible Qwen3-Coder-Next affine-Q4 graph processes fresh suffixes of
at least 32 tokens with true layer-major Metal prefill. The live Metal working-set snapshot chooses
a chunk of at most 8,192 rows while retaining a 512 MiB safety margin and limiting graph scratch to
5% of the resident/session basis; shorter suffixes or insufficient headroom keep the scalar token
command. Full-attention KV, hybrid recurrent state, row-local top-10 routes, and greedy output are
bitwise qualified against scalar prefill across fresh, restored-prefix, and forced chunk-boundary
runs. This does not alter expert placement: a resident graph issues no expert reads, and a streamed
graph obtains each unique layer expert through the existing parallel positioned-read scheduler.
There is no execution-error fallback or second expert cache. An explicit Qwen GGUF URI continues to
use llama.cpp. pb
discovers GGUF files below the pulled model's cache directory, including Hugging Face variant
subdirectories, and selects shard 1 for a split checkpoint; it does not mistake a larger later
shard or `mmproj` file for the model entry point.
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
Native DSML tool generation now preserves tokenizer control-token identity, validates ordered
parameters and the exposed schema while sampling, and uses the same controller-snapshot mutation
gate as Qwen. Mutation payloads are JSON strings inside DSML so their exact closing boundary is
unambiguous. Invalid supported syntax, stale or inexact canonical patches, incomplete DSML, and EOS
before a required tool are rejected before execution. This changes neither expert scheduling nor
the local-session state contract.
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
| `.pb/tasks.toml` | Task decomposition effort presets, aggregate ceilings, and coordination limits. |
| `.pb/policy.toml` | User/tool policy rules evaluated for this project. |
| `.pb/mcp.toml` | Project MCP servers and their declared capabilities. |

Not every project needs every file. `pb init` creates or preserves the environment, workspace,
strict-workflow, durable-Goal, and Task-decomposition foundations.

## Task decomposition policy

**Shipped policy; Build partitioning is on by default.** `.pb/tasks.toml` is a versioned, hashed ceiling document for
high-level Task-plan validation and controller-owned budget projection. It defines `small`,
`medium`, and `large` qualitative effort presets, an eight-Task aggregate ceiling by default, and a
two-attempt coordination allowance. Default partitions use bounded Build Tasks with the `small`
preset; executable numeric budgets are compiled from this document and included in the accepted
artifact digest.

The file cannot disable or enable Task planning, qualify automatic Goal selection, start a Goal,
expand workflow authority, or allow publication. Exact model/backend/template/protocol evidence is
required only for the separate automatic Goal-shaped Task promotion; that embedded catalog is
empty. Existing `.pb/workflow.toml` and `.pb/goal.toml` formats and limits are unchanged. Accepted
multi-Task Builds use these ceilings for the durable controller and read-only Tasks UI.

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
cmd = { regex = "(^|\\s)(git push|gh pr create)(\\s|$)" }

[[rules]]
name = "block recursive deletes"
outcome = "deny"
profiles = ["build"]
tools = ["run_command"]

[rules.params]
cmd = { regex = "rm\\s+-rf" }
```

Exact tool names and their argument paths are validated when the session starts. A misspelled tool
or argument makes the policy fail closed instead of silently leaving the call unmatched. Wildcard
tool rules remain available when one rule intentionally spans several tool schemas, but the
wildcard must match an exposed tool and every constrained argument must exist on at least one
matching schema.

Policy is evaluated in addition to workflow-stage capabilities. A policy can narrow an available
tool call; it cannot make a tool available in a stage where the harness forbids it.

Parameter matchers accept scalar literals or explicit `{ equals, contains, glob, regex, exists }`
tables. Unknown root, rule, or matcher fields fail configuration loading, so a misspelled matcher
cannot degrade into an inert literal object.

## MCP and LSP integrations

List configured integrations or discover marketplace entries:

```bash
pb integrations list
pb integrations list --marketplace
```

**Shipped package contract.** Marketplace LSP images publish a versioned typed manifest in the
`uk.unrtd.pb.integration.lsp-manifest` OCI annotation. pb validates its size, shape, language IDs,
arguments, initialization options, and cache IDs before writing global configuration. A package
manifest cannot select a host command, inject environment values, pin a container runtime, request
networking, or make the workspace writable. Marketplace LSP installation fails when that manifest
is absent or invalid; the web UI reports the missing contract instead of offering an unconfigured
install.

Marketplace configuration resolves the displayed tag to an immutable OCI manifest digest. pb
stores that digest as the executable image identity, keeps the original tag only as display/update
metadata, verifies the typed manifest again through the digest, and downloads that exact image
during installation when a container runtime is available. Task startup does not pull a missing
pinned image. Existing tag-only marketplace LSP entries appear as **Upgrade required** and cannot
run until they are explicitly upgraded or reinstalled; no compatibility switch permits mutable
execution. **Configure** always edits environment values against the installed digest and preserves
the saved source tag. **Upgrade** is a separate action that re-resolves that source, verifies the
new digest and package metadata, and then replaces the pinned identity. Project MCP marketplace
entries preserve the same pinned/source distinction. Schema, install, list, and removal failures
remain visible in the integration UI instead of closing the form or silently doing nothing.
Registry/repository case and a trailing registry dot are canonicalized before marketplace policy is
applied. Each selected platform manifest and image-config blob is hashed and matched to its parent
digest rather than trusting an optional registry response header. The web UI cancels superseded
schema lookups and refuses to install metadata that does not name the currently selected image.

The rust-analyzer package is published by the `crunchy-pb/lsp-rust-analyzer` marketplace repository.
Install it from the web integration store or from the CLI; pb reads and validates the embedded
manifest, so no local manifest or package checkout is required:

```bash
pb integrations add lsp ghcr.io/crunchy-pb/lsp-rust-analyzer:latest \
  --name rust-analyzer
```

The rust-analyzer profile mounts the workspace read-only, runs without egress, disables dependency
fetching, check-on-save, build scripts, procedural macros, cache priming, and automatic Cargo
reload, and leaves settled-state compilation and tests to pb's checks. Its packaged Rust toolchain
is a stable baseline, not a promise that projects pinned to another compiler receive exact semantic
parity.

Settled-transaction LSP semantic analysis is separately opt-in per server. This is distinct from the
automatic request-local native Rust layer described above: the LSP receives complete candidate
overlays and is not queried for token-time diagnostics on incomplete code. The field is accepted in global
`[lsp.servers.<name>]` entries and project `.pb/lsp.toml` entries:

```toml
[servers.rust-analyzer]
container_image = "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:<digest>"
language_ids = ["rust"]
semantic_enforcement = "advisory" # disabled | advisory | required
```

`disabled` is the default and disables only this LSP transaction gate; the syntax collar and any
qualified automatic native language layer remain active. `advisory`
opens exact controller-provided base/candidate overlays in a fresh bounded shadow workspace and
isolated provider session, and records semantic outcomes, but an error, timeout, stale workspace,
unpinned host command, incomplete baseline, or unsupported construct does not block generation or
publication. `required` permits a mutation-payload close and entry into the final publication step
only after a digest-pinned provider returns a complete full pull-diagnostic result for the exact
document versions and introduces no classified error. Required Rust evidence also needs a loaded
non-empty crate graph and analyzer-confirmed membership for every overlay document. Provider loss,
timeout, stale content, push-only diagnostics, unclassified errors, detached files, deletions whose
dependants cannot be proven, and other `Unknown` outcomes fail closed. A host-command LSP is never
treated as a pinned semantic authority even if its configuration contains a digest-shaped string.

The shipped semantic classifier covers scoped unresolved name/import/field/method, call, type,
privacy, ownership, and mutability errors reported by rust-analyzer, the TypeScript language
service, and Pyright. The guarantee is diagnostic- and project-profile-relative: JavaScript needs
configured type checking, and Python `Any`, dynamic imports, monkey-patching, descriptors, or other
runtime-only behavior can remain unknown. HTML and CSS retain syntax-only guarantees. Existing
baseline diagnostics are compared with candidate diagnostics so repairs remain possible; only an
exact complete baseline can authorize a required clean result. Provider dependency resolution uses
the digest-pinned sidecar's embedded offline sysroot/package/stub data. Declared external LSP cache
attachments currently make the result `Unknown`; a later dependency-fact/cache profile must bind
their exact content and mount it read-only before it can authorize required mode. This keeps
out-of-repository public-symbol resolution local without treating a mutable cache as evidence.

At generation time, Qwen JSON and DeepSeek DSML reject a provider-confirmed bad closing payload
token before it commits, leaving further source tokens reachable. The same full-vocabulary ordering
is used by FlashMoe and llama.cpp. The executor then rebuilds the exact virtual mutation, verifies
the live base, captures and revalidates the current workspace, reruns the configured provider, and
publishes only after that final gate. Invocation telemetry contains the active guarantee rung,
provider outcome counts and elapsed milliseconds, recovery capability, schema hashes, and
counts—not source, paths, diagnostic messages, or generated payload text. Generation-boundary and
final-executor decisions have separate content-free receipts; the former never substitutes for the
latter. Speculative multi-token rewind is not enabled; receipts report `candidate_probe_only`.

After configuration, pb uses the server automatically for changed Rust task paths. During partial
implementation it surfaces only syntax/parser errors; at a settled work boundary and before
handoff it surfaces current error diagnostics. The web transcript shows this as Trinity inspecting
language diagnostics, with the Actions drawer retaining the structured result. These passes are
bounded repair hints, not substitutes for configured checks, code review, or the managed commit.
An unavailable or timed-out server is reported and pb continues to the ordinary checks. pb never
applies language-server edits, commands, formatting, or code actions. A full pull-diagnostic
response can complete a server/file target. A push-only server can still surface a fresh bounded
snapshot, but even an empty snapshot remains explicitly incomplete and is never presented as proof
that the target is clean. One 12-second proactive deadline begins before bounded workspace
observation and covers server launch, stdin delivery, initialization, collection, any typed restart,
shutdown, and final workspace revalidation.

Language servers start lazily on their first manual query or matching proactive pass. A project does
not need a container-backed task environment: for local and otherwise unconfigured projects, pb
creates a service-only lease using the detected local container runtime, mounts the workspace
read-only, applies the package's network declaration, rejects cache IDs that are not backed by a
declared task-environment cache, and removes the owned sidecar at the session boundary. The
integration list distinguishes ready, unavailable, disabled, and legacy
unverified installations instead of claiming every configured server is ready.

The `--runtime` install argument is optional and normally should be omitted. An integration without
that field follows the runtime that owns each task session. Supplying it retains an exact
compatibility assertion; a task using another runtime will reject the service rather than create a
second cleanup domain.

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
read_only_tools = ["search_*", "get_document"]

[mcp.servers.example.capabilities.secret_env]
API_TOKEN = "EXAMPLE_API_TOKEN"
```

Workspace, network, cache, and secret capabilities default to none for a container service. Secret
values are resolved from the host environment at launch and are not written to project
configuration, command arguments, or the session ledger.

`read_only_tools` is a separate, operator-audited list of raw server tool names or `*` patterns.
It defaults to empty. Only matching tools are exposed to current agents; server annotations are
descriptive hints and cannot grant authority. pb currently has no external-mutation MCP workflow,
so a mutating tool cannot be enabled merely by granting its service read-write workspace access.
Unsupported dynamic schemas, duplicate normalized names, and failed discovery are exposed only as
explicit integration status failures. Empty, oversized, or unmatched `read_only_tools` patterns
also produce a status failure instead of silently broadening or pretending to expose a tool.

Host-command MCP servers are less isolated. Container-backed sessions require such integrations to
use a container image so pb can enforce workspace, cache, and network capabilities.
When a container integration retains an explicit `container_runtime` field, pb treats it as an exact
compatibility assertion against the runtime that owns the session. A mismatch fails startup rather
than silently launching with a different runtime; the field does not create an independent runtime
or cleanup domain for that one service.

The GitHub setup flow uses a loopback OAuth callback and writes its token below the user
configuration directory with owner-only mode on Unix:

```bash
pb mcp setup github
```

Setup establishes the server and credential path but deliberately leaves `read_only_tools` empty.
Inspect the raw tool names reported by the server, then add only the operations you have audited as
read-only. This prevents an updated external server from granting new authority through its own
annotations.

On macOS, Safari Technology Preview 247 or newer includes its own local MCP server. pb does not
ship a separate WebDriver browser-control layer. Configure the preview server for the current
project with:

```bash
pb mcp setup safari
```

The command points the project MCP entry at Safari Technology Preview's `safaridriver --mcp`.
It also leaves `read_only_tools` empty; browser navigation, clicks, typing, and page-side actions are
not read-only merely because they do not modify the repository. Add only genuinely observational
raw tools after auditing the installed Safari server.
Before starting an agent, enable **Developer > Enable remote automation and external agents** in
Safari Technology Preview. Use `--driver-path` if the application is installed somewhere other
than `/Applications`. Because this is a host-command MCP server, use it with a local execution
environment rather than a container-backed session.

Enabling any remote integration is also a privacy decision. Review [Your data and privacy](data-and-privacy.md)
before granting repository or egress access.
