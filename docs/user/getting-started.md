# Getting started

pb is currently centered on macOS arm64. The release pipeline publishes a single
`pb-aarch64-apple-darwin` binary, and service and tray management use launchd.

## Install

Download the latest macOS arm64 binary from the
[GitHub releases page](https://github.com/Jc2k/pb/releases), make it executable, and run it from a
terminal. `pb self install` moves the current binary to `~/.local/bin/pb`, offers to install the
launchd agents for `pb serve` and the menu bar item, and starts them after confirmation.

```bash
chmod +x pb-aarch64-apple-darwin
./pb-aarch64-apple-darwin self install
```

The release can be ad-hoc or self-signed rather than Apple-notarized, so Gatekeeper may ask you to
confirm the first launch.

## Pull a local model

`pb pull` downloads the configured default model. It also accepts an Ollama-compatible model name
or a Hugging Face URI.

```bash
pb pull
pb pull qwen3-coder-next
pb pull hf://owner/repository/model.gguf
```

Model files normally live below the pb data directory. Use `--out-dir` or set `model.model_dir` if
you want to manage their location yourself.

## Prepare a project

From a Git repository, let pb inspect the project and write its per-project environment
configuration:

```bash
cd /path/to/project
pb init
```

Apple containers are the default execution backend. For projects that require the macOS host—such
as Xcode or other platform-only toolchains—select local execution explicitly:

```bash
pb init --backend local
```

Inspect the result before running work:

```bash
pb env status
pb env start
```

`pb init` preserves existing project configuration rather than silently replacing it.

## Start pb

If you did not install the launchd service, start the local server in a terminal:

```bash
pb serve
```

The web interface listens on `127.0.0.1:8311` by default. Open that address in a browser. The same
process also creates a local Unix socket used by `pb queue`.

## Run your first task

`pb queue` is an explicit delivery request. Give it a task and a project directory:

```bash
pb queue "Add validation for empty project names" --workdir /path/to/project
```

The terminal streams the same typed events shown in the web interface. You can list or reattach to
daemon sessions without starting a second run:

```bash
pb queue --list
pb queue --session SESSION_ID
```

Continue with [Working with pb](working-with-pb.md) to choose between conversation and delivery and
to understand what a Ready result means.

## Remove pb

`pb self uninstall` stops the launchd agents, removes their configuration, and removes the installed
binary after confirmation. Local models, configuration, and history are preserved unless you also
pass `--delete-data`.

```bash
pb self uninstall
pb self uninstall --delete-data
```
