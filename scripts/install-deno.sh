#!/usr/bin/env bash
set -euo pipefail

# Install the Deno runtime used by this repo's web UI tasks.
#
# Usage:
#   scripts/install-deno.sh                  # install latest stable Deno
#   DENO_VERSION=2.1.4 scripts/install-deno.sh
#   DENO_INSTALL="$HOME/.deno" scripts/install-deno.sh
#   DENO_DOWNLOAD_TIMEOUT=120 scripts/install-deno.sh
#
# After installation, add "$DENO_INSTALL/bin" to PATH in your shell or CI job.

DENO_VERSION="${DENO_VERSION:-latest}"
DENO_INSTALL="${DENO_INSTALL:-$HOME/.deno}"
DENO_DOWNLOAD_TIMEOUT="${DENO_DOWNLOAD_TIMEOUT:-120}"
DENO_BIN="$DENO_INSTALL/bin/deno"

if [[ -x "$DENO_BIN" && "$DENO_VERSION" != "latest" ]]; then
  installed_version="$("$DENO_BIN" --version | awk 'NR == 1 { print $2 }')"
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

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64 | Linux-amd64)
    deno_target="x86_64-unknown-linux-gnu"
    ;;
  Linux-aarch64 | Linux-arm64)
    deno_target="aarch64-unknown-linux-gnu"
    ;;
  Darwin-x86_64 | Darwin-amd64)
    deno_target="x86_64-apple-darwin"
    ;;
  Darwin-aarch64 | Darwin-arm64)
    deno_target="aarch64-apple-darwin"
    ;;
  *)
    echo "error: unsupported platform $(uname -s)-$(uname -m)" >&2
    exit 1
    ;;
esac

if [[ "$DENO_VERSION" == "latest" ]]; then
  deno_urls=(
    "https://dl.deno.land/release/latest/deno-${deno_target}.zip"
    "https://github.com/denoland/deno/releases/latest/download/deno-${deno_target}.zip"
  )
else
  deno_urls=(
    "https://dl.deno.land/release/v${DENO_VERSION}/deno-${deno_target}.zip"
    "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-${deno_target}.zip"
  )
fi

mkdir -p "$DENO_INSTALL/bin"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
archive="$tmp_dir/deno.zip"

download_with_curl() {
  curl --fail --location --show-error --silent \
    --connect-timeout 20 \
    --max-time "$DENO_DOWNLOAD_TIMEOUT" \
    --retry 3 \
    --retry-delay 2 \
    --retry-max-time $((DENO_DOWNLOAD_TIMEOUT * 2)) \
    --output "$archive" \
    "$deno_url"
}

download_with_wget() {
  wget --quiet \
    --timeout=20 \
    --tries=3 \
    --output-document "$archive" \
    "$deno_url"
}

downloaded=0
for deno_url in "${deno_urls[@]}"; do
  echo "Downloading Deno from $deno_url"
  if command -v curl >/dev/null 2>&1; then
    if download_with_curl; then
      downloaded=1
      break
    fi
  elif command -v wget >/dev/null 2>&1; then
    if download_with_wget; then
      downloaded=1
      break
    fi
  else
    echo "error: curl or wget is required to install Deno" >&2
    exit 1
  fi

  echo "warning: failed to download from $deno_url; trying next mirror" >&2
done

if [[ "$downloaded" != "1" ]]; then
  echo "error: failed to download Deno after trying all mirrors" >&2
  exit 1
fi

if command -v unzip >/dev/null 2>&1; then
  unzip -q -o "$archive" -d "$tmp_dir"
else
  echo "error: unzip is required to extract Deno" >&2
  exit 1
fi

install -m 0755 "$tmp_dir/deno" "$DENO_BIN"

"$DENO_BIN" --version
cat <<MSG

Deno was installed to: $DENO_BIN
Add this to your shell profile or CI environment before running repo tasks:
  export PATH="$DENO_INSTALL/bin:\$PATH"

Then verify the web UI tests with:
  deno task test:web
MSG
