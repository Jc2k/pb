#!/usr/bin/env bash
set -euo pipefail

# Install the Deno runtime used by this repo's web UI tasks.
#
# Usage:
#   scripts/install-deno.sh                  # install latest stable Deno
#   DENO_VERSION=2.1.4 scripts/install-deno.sh
#   DENO_INSTALL="$HOME/.deno" scripts/install-deno.sh
#
# After installation, add "$DENO_INSTALL/bin" to PATH in your shell or CI job.

DENO_VERSION="${DENO_VERSION:-latest}"
DENO_INSTALL="${DENO_INSTALL:-$HOME/.deno}"
DENO_BIN="$DENO_INSTALL/bin/deno"

if [[ -x "$DENO_BIN" && "$DENO_VERSION" != "latest" ]]; then
  installed_version="$($DENO_BIN --version | awk 'NR == 1 { print $2 }')"
  if [[ "$installed_version" == "$DENO_VERSION" ]]; then
    echo "Deno $DENO_VERSION is already installed at $DENO_BIN"
    echo "Add to PATH: export PATH=\"$DENO_INSTALL/bin:\$PATH\""
    exit 0
  fi
  echo "Replacing Deno $installed_version with Deno $DENO_VERSION"
elif [[ -x "$DENO_BIN" ]]; then
  echo "Deno is already installed at $DENO_BIN"
  "$DENO_BIN" --version
  echo "Set DENO_VERSION=<version> to force a specific version."
  echo "Add to PATH: export PATH=\"$DENO_INSTALL/bin:\$PATH\""
  exit 0
fi

mkdir -p "$DENO_INSTALL"
install_args=()
if [[ "$DENO_VERSION" != "latest" ]]; then
  install_args=("v$DENO_VERSION")
fi

if command -v curl >/dev/null 2>&1; then
  curl -fsSL https://deno.land/install.sh | DENO_INSTALL="$DENO_INSTALL" sh -s -- "${install_args[@]}"
elif command -v wget >/dev/null 2>&1; then
  wget -qO- https://deno.land/install.sh | DENO_INSTALL="$DENO_INSTALL" sh -s -- "${install_args[@]}"
else
  echo "error: curl or wget is required to install Deno" >&2
  exit 1
fi

"$DENO_BIN" --version
cat <<MSG

Deno was installed to: $DENO_BIN
Add this to your shell profile or CI environment before running repo tasks:
  export PATH="$DENO_INSTALL/bin:\$PATH"

Then verify the web UI tests with:
  deno task test:web
MSG
