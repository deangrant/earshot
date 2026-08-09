# Audit

Run `cargo audit` (parity with `.github/workflows/audit.yml`).
Report findings. Fix advisories only if the user asks.

## Steps

From the repository root:

```bash
# if Cargo.lock is missing:
cargo generate-lockfile

# install once if needed:
cargo install cargo-audit --locked

cargo audit
```

## Output

Summarize as pass (no vulnerabilities) or list advisory IDs and crates.
Do not commit.
