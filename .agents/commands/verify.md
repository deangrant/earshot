# Verify

Run the local CI checklist for this crate (lint + test + optional audit).
Report pass/fail for each step. Fix failures only if the user asks.

## Build requirements

Compiling `whisper-rs` needs CMake, a C/C++ compiler, and `libclang`
(for example `libclang-dev` on Debian/Ubuntu). FFmpeg is not required.

## Steps

From the repository root:

```bash
cargo +nightly fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Optional (security workflow parity):

```bash
# if Cargo.lock is missing:
cargo generate-lockfile
cargo audit
```

Do **not** pass `--all-features` by default: the `cuda` feature needs a CUDA
toolkit and fails on plain CI runners.

## Output

Summarize each command as pass or fail with the first failing error line if any.
Do not commit.
