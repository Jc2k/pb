# pb

A local coding agent CLI.

## Commands

- `pb self update` - self update from the latest GitHub release binary.
- `pb pull [model]` - pull model blobs with retry, batching, and parallel download.
- `pb agent <task> --model-path /absolute/path/to/model.gguf` - run a local in-process coding agent with streamed output, tool calls (read/search/edit/skill), and inline diffs.

### Agent usage

```bash
pb agent "fix failing test in src/lib.rs" \
  --model-path /absolute/path/to/model.gguf \
  --workdir /absolute/path/to/repo
```

The agent runs the model in-process (no LLM subprocess), streams token output to terminal, executes tool calls inside the workspace root, prints edit diffs, and supports `skill` lookups for `copilot`, `codex`, and `claude-code`.

## Build and test

```bash
cargo test
cargo build --release
```

## CI

GitHub Actions workflow (`.github/workflows/ci-release.yml`) builds with cache, runs unit tests, performs semantic release tagging, and then produces an optimized macOS arm64 binary asset.
