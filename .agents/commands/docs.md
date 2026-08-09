# Docs

Build API docs and surface missing-docs or broken intra-doc links.
This crate uses `#![deny(missing_docs)]`.

## Steps

From the repository root:

```bash
cargo doc --no-deps
```

Optionally open docs locally (do not require a browser in CI-like runs):

```bash
cargo doc --no-deps --open
```

## Output

Report pass/fail. On failure, list the first missing-doc or link errors.
Fix only if the user asks. Do not commit.
