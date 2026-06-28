#!/usr/bin/env bash
set -euo pipefail

# Install the Deno runtime used by this repo's web UI tasks.
#
# Usage:
#   scripts/install-deno.sh                  # install latest stable Deno
#   DENO_VERSION=2.1.4 scripts/install-deno.sh
#   DENO_INSTALL="$HOME/.deno" scripts/install-deno.sh
#   DENO_DOWNLOAD_TIMEOUT=120 scripts/install-deno.sh
#   DENO_INSTALL_PROJECT_DEPS=0 scripts/install-deno.sh  # only install Deno
#
# After installation, add "$DENO_INSTALL/bin" to PATH in your shell or CI job.

DENO_VERSION="${DENO_VERSION:-latest}"
DENO_VERSION="${DENO_VERSION#v}"
DENO_INSTALL="${DENO_INSTALL:-$HOME/.deno}"
DENO_DOWNLOAD_TIMEOUT="${DENO_DOWNLOAD_TIMEOUT:-120}"
DENO_INSTALL_PROJECT_DEPS="${DENO_INSTALL_PROJECT_DEPS:-1}"
DENO_BIN="$DENO_INSTALL/bin/deno"

if ! [[ "$DENO_DOWNLOAD_TIMEOUT" =~ ^[0-9]+$ ]] || (( DENO_DOWNLOAD_TIMEOUT <= 0 )); then
  echo "error: DENO_DOWNLOAD_TIMEOUT must be a positive integer number of seconds" >&2
  exit 1
fi

install_project_dependencies() {
  if [[ "$DENO_INSTALL_PROJECT_DEPS" == "0" ]]; then
    return
  fi

  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd -- "$script_dir/.." && pwd)"

  if [[ -f "$repo_root/deno.json" ]]; then
    echo "Installing web UI dependencies"
    (cd "$repo_root" && "$DENO_BIN" install)
  fi

  if [[ -f "$repo_root/Cargo.toml" ]]; then
    if command -v cargo >/dev/null 2>&1; then
      echo "Installing Rust dependencies"
      # Limit prefetching to the current Rust host target. An unrestricted
      # `cargo fetch` resolves every target-specific dependency in Cargo.lock,
      # including macOS-only crates such as objc2 on Linux CI runners. The
      # install script only needs dependencies for the local verification
      # commands it prints below, so fetching the host target keeps dependency
      # installation aligned with `cargo test --all-targets`.
      rust_host="$(rustc -vV | awk '/^host:/ { print $2 }')"
      if [[ -n "$rust_host" ]]; then
        (cd "$repo_root" && cargo fetch --locked --target "$rust_host")
      else
        (cd "$repo_root" && cargo fetch --locked)
      fi
    else
      echo "warning: cargo is not installed; skipping Rust dependency installation" >&2
    fi
  fi
}

if [[ -x "$DENO_BIN" && "$DENO_VERSION" != "latest" ]]; then
  installed_version="$("$DENO_BIN" --version | awk 'NR == 1 { print $2 }')"
  if [[ "$installed_version" == "$DENO_VERSION" ]]; then
    echo "Deno $DENO_VERSION is already installed at $DENO_BIN"
    install_project_dependencies
    echo "Add to PATH: export PATH=\"$DENO_INSTALL/bin:\$PATH\""
    exit 0
  fi
  echo "Replacing Deno $installed_version with Deno $DENO_VERSION"
elif [[ -x "$DENO_BIN" ]]; then
  echo "Deno is already installed at $DENO_BIN"
  "$DENO_BIN" --version
  install_project_dependencies
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

deno_urls=()
if [[ "$DENO_VERSION" == "latest" ]]; then
  resolved_deno_version=""
  if command -v curl >/dev/null 2>&1; then
    resolved_deno_version="$(curl --fail --location --show-error --silent \
      --connect-timeout 20 \
      --max-time "$DENO_DOWNLOAD_TIMEOUT" \
      https://dl.deno.land/release/latest.txt || true)"
  elif command -v wget >/dev/null 2>&1; then
    resolved_deno_version="$(wget --quiet \
      --timeout=20 \
      --tries=3 \
      --output-document - \
      https://dl.deno.land/release/latest.txt || true)"
  else
    echo "error: curl or wget is required to install Deno" >&2
    exit 1
  fi

  if [[ "$resolved_deno_version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    deno_urls+=("https://dl.deno.land/release/${resolved_deno_version}/deno-${deno_target}.zip")
  fi
  deno_urls+=("https://github.com/denoland/deno/releases/latest/download/deno-${deno_target}.zip")
else
  resolved_deno_version="v${DENO_VERSION}"
  deno_urls=(
    "https://dl.deno.land/release/${resolved_deno_version}/deno-${deno_target}.zip"
    "https://github.com/denoland/deno/releases/download/${resolved_deno_version}/deno-${deno_target}.zip"
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
  if command -v curl >/dev/null 2>&1 && download_with_curl; then
    downloaded=1
    break
  fi

  if command -v wget >/dev/null 2>&1 && download_with_wget; then
    downloaded=1
    break
  fi

  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
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
install_project_dependencies
cat <<MSG

Deno was installed to: $DENO_BIN
Add this to your shell profile or CI environment before running repo tasks:
  export PATH="$DENO_INSTALL/bin:\$PATH"

Then verify the installed dependencies with:
  deno task test:web
  cargo test --all-targets
MSG
