# pb

A local coding agent CLI.

## Commands

- `pb self update` - self update from the latest GitHub release binary.
- `pb pull [model]` - pull `qwen3-coder-next` (default) from the Ollama model registry with retry, batching, and parallel download.
- `pb agent <task>` - run a simple terminal-first agent workflow and stream progress to the console.

## Build and test

```bash
cargo test
cargo build --release
```

## CI

GitHub Actions workflow (`/home/runner/work/pb/pb/Jc2k/pb/.github/workflows/ci-release.yml`) builds with cache, runs unit tests, performs semantic release tagging, and then produces an optimized macOS arm64 binary asset.
