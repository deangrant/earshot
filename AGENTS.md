# Agent and contributor guidance

Structured conventions for AI agents and humans working in this repository. For
fuller context, see [README.md](README.md).

## Docs

- [`.agents/docs/ARCHITECTURE.md`](.agents/docs/ARCHITECTURE.md) — high-level system architecture and diagrams
- [DeepWiki](https://deepwiki.com/deangrant/earshot) — indexed project wiki (architecture, API, pipeline)

## Rules

- [`.agents/rules/`](.agents/rules/) (symlinked from [`.cursor/rules`](.cursor/rules))
- [`.agents/rules/earshot-core.mdc`](.agents/rules/earshot-core.mdc) — always-on crate policy (modules, CUDA/FFmpeg, skills)

## Skills

- [`.agents/skills/`](.agents/skills/)
- [`.agents/skills/whisper-earshot/`](.agents/skills/whisper-earshot/) — Whisper STT domain contracts
- [`.agents/skills/symphonia-decode/`](.agents/skills/symphonia-decode/) — Symphonia decode and resample pipeline
- [`.agents/skills/ci-rust-earshot/`](.agents/skills/ci-rust-earshot/) — CI and local verify parity
- [`.agents/skills/solid-rust/`](.agents/skills/solid-rust/) — SOLID design in Rust
- [`.agents/skills/style-guide-rust/`](.agents/skills/style-guide-rust/) — Rust style and documentation

## Commands

- [`.agents/commands/`](.agents/commands/) (symlinked from [`.cursor/commands`](.cursor/commands))
- `/verify` — local CI checklist (nightly fmt, clippy, test)
- `/lint` — fmt + clippy only
- `/test` — `cargo test` only
- Also: `/audit`, `/docs`, `/smoke-transcribe`, `/code-review-api`

## Hooks

- Config: [`.cursor/hooks.json`](.cursor/hooks.json)
- `sessionStart` → [`.agents/hooks/session-context.sh`](.agents/hooks/session-context.sh)
- `afterFileEdit` → [`.agents/hooks/rustfmt.sh`](.agents/hooks/rustfmt.sh) formats edited `*.rs` files
- `beforeShellExecution` → [`.agents/hooks/shell-safety.sh`](.agents/hooks/shell-safety.sh)
- `beforeReadFile` → [`.agents/hooks/warn-binaries.sh`](.agents/hooks/warn-binaries.sh)
- `stop` → [`.agents/hooks/suggest-lint.sh`](.agents/hooks/suggest-lint.sh)
