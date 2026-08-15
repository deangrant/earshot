---
name: symphonia-decode
description: >-
  Symphonia decode pipeline guidance for earshot. Use when editing
  src/audio/symphonia.rs, resampling, channel downmix, packet loops, or
  duration limits.
trigger: >-
  Symphonia, SampleBuffer, resample, rubato, to_mono, decode_file,
  ResetRequired, AudioTooLong, channel count, Fft, process_all
---

# Symphonia decode pipeline

## Structure

Prefer small helpers over a monolithic `decode_file`:

1. Open/probe format
2. Select audio track + copy codec params
3. Decode packets to interleaved PCM
4. Downmix to mono
5. Resample to `WHISPER_SAMPLE_RATE` with rubato `Fft::process_all`

## Correctness

- Grow or recreate `SampleBuffer` when a later packet needs more capacity.
- Do not silently skip decode errors; surface `Error::AudioDecode` (or typed
  variants like `UnsupportedBitstreamReset`, `AudioTooLong`).
- Resolve channel count from metadata and decoded frames; mismatch is an error.
- Format-level `ResetRequired` is unsupported; decoder-level reset may clear
  local PCM state and continue.
- Enforce `max_duration_secs` while decoding to bound memory.

## Resampling

- Identity path when rates match.
- Otherwise use rubato `Fft` with `FixedSync::Both` and `process_all` on a
  mono `InterleavedSlice` so chunking and startup-delay trim are handled
  internally.

## Testing ideas

- Same-rate identity, 44.1→16 kHz length, growing packet capacities, duration
  cap, invalid files.
