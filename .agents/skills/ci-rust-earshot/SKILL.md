---
name: ci-rust-earshot
description: >-
  Local and GitHub Actions CI for earshot. Use when editing workflows, agent
  verify/lint/test commands, rustfmt nightly options, or diagnosing cmake /
  libclang / CUDA build failures.
trigger: >-
  CI, GitHub Actions, cargo test, clippy, rustfmt, nightly fmt, cargo audit,
  libclang, cmake, CUDA feature, Cargo.lock, verify command
---

# CI for earshot

## Workflows

| Workflow | Role |
|----------|------|
| `.github/workflows/lint.yml` | nightly `fmt --check` + stable clippy |
| `.github/workflows/test.yml` | stable `cargo test` |
| `.github/workflows/audit.yml` | `cargo audit` (path/schedule) |

Agent command mirrors live under `.agents/commands/` only.

## Default feature policy

- Do **not** use `--all-features` in default CI or `/verify` / `/lint` / `/test`.
- The only non-default feature is `cuda`, which needs a CUDA toolkit.

## Formatting

- `rustfmt.toml` enables unstable options → use `cargo +nightly fmt`.
- After agent edits, `.agents/hooks/rustfmt.sh` formats `.rs` files (fail open).

## Native build deps

whisper-rs-sys needs:

- CMake
- C/C++ compiler
- libclang (bindgen)

FFmpeg is not required. `Cargo.lock` is gitignored; generate in CI when missing.

## When fixing CI

1. Align `.agents/commands/*` with the failing workflow.
2. Keep action pins and `permissions: contents: read`.
3. Install the same apt packages as the workflows on Linux runners.
