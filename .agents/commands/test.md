# Test

Run the unit/doc test suite only (parity with `.github/workflows/test.yml`).
Report pass/fail. Fix failures only if the user asks.

## Steps

From the repository root:

```bash
# if Cargo.lock is missing:
cargo generate-lockfile

cargo test
```

Build dependencies (cmake, C/C++ compiler, libclang) must be available.
Do **not** pass `--all-features` (CUDA is not available on default runners).

## Output

Summarize as pass or fail with the first failing error line if any.
Do not commit.
