# Code review API

Review the public API and module boundaries for contract and docs quality.
Do not modify code unless the user asks; describe findings only.

## Skills to apply

- `.agents/skills/solid-rust` — DIP/LSP/ISP for traits (`AudioDecoder`, `Transcriber`)
- `.agents/skills/style-guide-rust` — docs, naming, file/function size
- `.agents/skills/whisper-earshot` — domain contracts (PCM rate, language, trust)

## Scope

Focus on:

- `src/lib.rs` re-exports and crate docs
- `src/engine.rs`, `src/audio/`, `src/model.rs`, `src/config.rs`, `src/types.rs`, `src/error.rs`
- Public examples under `examples/`

## Checklist

1. Trait contracts documented and honored (non-empty decode; empty PCM → error).
2. Concrete decoders wired at composition roots, not inside trait defaults.
3. Canonical doc sections: `# Errors`, `# Abort`, `# Panics` where applicable.
4. `#[doc(inline)]` on crate `pub use` of local items.
5. No speculative traits or over-abstraction.

## Output

Markdown findings by file, tagged High / Medium / Low. No code rewrites.
