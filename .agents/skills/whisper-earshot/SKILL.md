---
name: whisper-earshot
description: >-
  Domain playbook for the earshot Whisper STT crate. Use when editing
  transcription, audio decode, model loading, language detection, ComputeType,
  or examples involving GGML / Symphonia.
trigger: >-
  Whisper, transcription, earshot, GGML, Symphonia, resample, 16 kHz,
  detect_language, ComputeType, model_ftype, AudioDecoder, Transcriber,
  transcribe_file, MAX_AUDIO_DURATION
---

# Whisper / earshot domain

## Audio contract

- Whisper expects **mono `f32` PCM at 16 kHz** (`WHISPER_SAMPLE_RATE`).
- `AudioDecoder::decode_file` must return a **non-empty** buffer on success.
- Default decode duration cap is `MAX_AUDIO_DURATION_SECS` (2 hours); override
  via `SymphoniaDecoder::max_duration_secs`.
- Enabled Symphonia formats: MP3, WAV, FLAC, AAC/MP4/M4A (see `Cargo.toml`
  features). Do not claim unrestricted Symphonia format support.

## Transcription

- Prefer language `"auto"` (or `None` → `"auto"`) for detect-then-transcribe.
- Do **not** set whisper.cpp `detect_language(true)` if full transcription is
  required; that flag can mean detect-and-exit.
- Reject language strings containing null bytes before calling whisper-rs.
- Empty PCM → `Error::InvalidConfig`.
- Inject decoders with `transcribe_file_with`; default Symphonia wiring belongs
  on `WhisperModel::transcribe_file`, not on the `Transcriber` trait.

## Models and compute

- `ComputeType` must match the GGML file’s `model_ftype` at load time.
- Int8 ≈ quantized (`q8_0` / `q5_0` / …); Float16/Float32 match F16/F32 weights.
- Model paths are a **native trust boundary** (`# Abort` on load).
- CUDA is feature-gated; CPU is the default path.

## Skills to combine

- SOLID / traits: `.agents/skills/solid-rust`
- Style / docs: `.agents/skills/style-guide-rust`
- Decode pipeline details: `.agents/skills/symphonia-decode`
