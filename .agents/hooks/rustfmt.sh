#!/usr/bin/env bash
# afterFileEdit: format edited .rs files with nightly rustfmt (fail open).
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
  exit 0
fi

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.file_path // empty')"

if [[ -z "$file_path" || "$file_path" != *.rs ]]; then
  exit 0
fi

if [[ ! -f "$file_path" ]]; then
  exit 0
fi

# Prefer nightly so unstable rustfmt.toml options apply; fail open always.
if cargo +nightly fmt -- --version >/dev/null 2>&1; then
  cargo +nightly fmt -- "$file_path" || true
elif command -v rustfmt >/dev/null 2>&1; then
  rustfmt --edition 2021 "$file_path" || true
fi

exit 0
