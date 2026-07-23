# rust-analyzer-lsp

An offline, read-only rust-analyzer sidecar packaged for pb. This directory is deliberately
repo-shaped: move its contents into `crunchy-pb/rust-analyzer-lsp`, give the repository the `lsp`
topic, and its workflow will publish `ghcr.io/crunchy-pb/rust-analyzer-lsp` for pb's marketplace.

## Safety profile

The embedded typed manifest gives pb the language and initialization defaults instead of relying
on an LLM to configure the server. Runtime behavior is deliberately narrower than a typical IDE:

- the project workspace is mounted read-only;
- the sidecar receives an internal network with no egress;
- Cargo dependency fetching is offline;
- cargo check, build scripts, procedural macros, cache priming, and automatic reload are disabled;
- pb's own settled-state checks remain authoritative.

The image contains Rust 1.96.0, rust-analyzer from that toolchain, and `rust-src`. Projects pinned
to another compiler may receive incomplete semantic information; syntax diagnostics remain useful.
The package never claims toolchain parity with the task container.

## Build and test locally

Apple container is the default local runtime:

```bash
./scripts/validate-package.py
./scripts/build-local.sh
CONTAINER_RUNTIME=container ./scripts/smoke.py
```

For Docker:

```bash
CONTAINER_RUNTIME=docker ./scripts/build-local.sh
CONTAINER_RUNTIME=docker ./scripts/smoke.py
```

Install the local image without pinning a runtime, so pb launches it in the runtime that owns each
task session:

```bash
pb integrations add lsp pb/rust-analyzer-lsp:dev \
  --name rust-analyzer \
  --manifest ./pb-lsp.json
```

Published marketplace installs read the same manifest from the
`uk.unrtd.pb.integration.lsp-manifest` OCI annotation.

## Publishing

The workflow validates metadata, performs a real initialize/open/diagnostics/shutdown LSP exchange,
then publishes Linux amd64 and arm64 images with SBOM and build provenance. Tag releases as `vX.Y.Z`;
the default branch also publishes `latest`.

The container redistributes components from the Rust toolchain and rust-analyzer. Their upstream
license files are included within the copied toolchain and remain governed by their respective
licenses.
