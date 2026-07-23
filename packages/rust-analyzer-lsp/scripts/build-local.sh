#!/bin/sh
set -eu

package_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime=${CONTAINER_RUNTIME:-container}
image=${IMAGE:-pb/rust-analyzer-lsp:dev}
manifest=$(tr -d '\n' < "$package_root/pb-lsp.json")
config_schema=$(tr -d '\n' < "$package_root/config-schema.json")

"$runtime" build \
  -t "$image" \
  -f "$package_root/Containerfile" \
  --build-arg "PB_LSP_MANIFEST=$manifest" \
  --build-arg "PB_CONFIG_SCHEMA=$config_schema" \
  "$package_root"
