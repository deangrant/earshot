# Smoke transcribe

Run the `transcribe` example against a real GGML model and audio file.
Skip with a clear message if paths are missing. Do not invent CUDA or FFmpeg steps.

## Inputs

Require two paths (from the user, env, or obvious local files):

1. `MODEL` — GGML Whisper `.bin` (trusted source only; see model-trust rule)
2. `AUDIO` — MP3, WAV, FLAC, or AAC/M4A

If either path is missing or not a file, stop and report what is needed.
Do not download arbitrary models.

## Steps

From the repository root (CPU default):

```bash
cargo run --example transcribe -- "$MODEL" "$AUDIO"
```

Only add `--features cuda` when the user explicitly asks and a CUDA toolkit is
available.

## Output

Report success with detected language and a short transcript sample, or the
first error line on failure. Do not commit.
