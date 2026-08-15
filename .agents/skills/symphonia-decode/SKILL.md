---
name: symphonia-decode
description: >-
  Symphonia decode pipeline guidance for earshot. Use when editing
  src/audio/symphonia.rs, resampling, channel downmix, packet loops, or
  duration limits.
trigger: >-
  Symphonia, GenericAudioBufferRef, resample, rubato, to_mono, decode_file,
  ResetRequired, AudioTooLong, channel count, Probe::probe, AudioCodecParameters,
  Fft, process_all
---

# Symphonia decode pipeline

## Structure

Prefer small helpers over a monolithic `decode_file`:

1. Open/probe format (`Probe::probe`)
2. Select audio track (`default_track(TrackType::Audio)`) + copy
   `AudioCodecParameters`
3. Decode packets to interleaved PCM
4. Downmix to mono
5. Resample to `WHISPER_SAMPLE_RATE` with rubato `Fft::process_all`

## Correctness

- Append decoded frames with `GenericAudioBufferRef::copy_to_slice_interleaved`
  into a growing `Vec` (do not use a removed `SampleBuffer`).
- Do not silently skip decode errors; surface `Error::AudioDecode` (or typed
  variants like `UnsupportedBitstreamReset`, `AudioTooLong`).
- Resolve channel count from metadata and decoded frames; mismatch is an error.
- Format-level `ResetRequired` is unsupported; decoder-level reset may clear
  local PCM state and continue.
- `next_packet` returns `Ok(None)` at end of stream; unexpected IO EOF is an
  error.
- Enforce `max_duration_secs` while decoding to bound memory.

## Resampling

- Identity path when rates match.
- Otherwise use rubato `Fft` with `FixedSync::Both` and `process_all` on a
  mono `InterleavedSlice` so chunking and startup-delay trim are handled
  internally.

## Testing ideas

- Same-rate identity, 44.1→16 kHz length, successive packet appends, duration
  cap, invalid files.
