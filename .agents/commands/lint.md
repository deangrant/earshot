# Lint

Run formatting and Clippy checks only (parity with `.github/workflows/lint.yml`).
Report pass/fail for each step. Fix failures only if the user asks.

## Steps

From the repository root:

```bash
cargo +nightly fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

Nightly is required for `rustfmt` because `rustfmt.toml` enables unstable
options (`imports_granularity`, `group_imports`). Clippy runs on the default
(stable) toolchain.

Do **not** pass `--all-features` (CUDA is not available on default runners).

## Output

Summarize each command as pass or fail with the first failing error line if any.
Do not commit.
