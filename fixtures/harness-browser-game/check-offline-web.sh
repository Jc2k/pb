#!/bin/sh
set -eu

matches="$(
  find . -type f \( -name '*.html' -o -name '*.css' -o -name '*.js' -o -name '*.mjs' \) \
    -exec grep -nE -H '(https?://|npm:|jsr:|[[:space:]](src|href)[[:space:]]*=[[:space:]]*["'"']//|url\([[:space:]]*["'"']?//)' {} + \
    2>/dev/null || true
)"

if [ -n "$matches" ]; then
  printf '%s\n' 'external browser dependency detected:' "$matches" >&2
  exit 1
fi

printf '%s\n' 'offline web dependency check passed'
