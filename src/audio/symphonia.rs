//! Symphonia-backed decoding for MP3, WAV, FLAC, and AAC (MP4/M4A).

use std::fs::File;
use std::path::Path;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, Packet};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::audio::decoder::{AudioDecoder, MAX_AUDIO_DURATION_SECS, WHISPER_SAMPLE_RATE};
use crate::error::{Error, Result};

/// Decodes MP3, WAV, FLAC, and AAC (MP4/M4A) to 16 kHz mono PCM.
#[derive(Debug, Clone, Copy)]
pub struct SymphoniaDecoder {
    /// Maximum decoded duration in seconds before failing.
    pub max_duration_secs: u64,
}

impl Default for SymphoniaDecoder {
    fn default() -> Self {
        Self {
            max_duration_secs: MAX_AUDIO_DURATION_SECS,
        }
    }
}

/// Metadata for the selected audio track used during packet decode.
struct TrackInfo {
    track_id: u32,
    sample_rate: u32,
    metadata_channels: Option<usize>,
    codec_params: CodecParameters,
}

impl AudioDecoder for SymphoniaDecoder {
    fn decode_file(&self, path: &Path) -> Result<Vec<f32>> {
        let mut format = open_format(path)?;
        let track = select_audio_track(format.as_ref(), path)?;
        let (interleaved, channels) =
            decode_interleaved(format.as_mut(), &track, self.max_duration_secs)?;
        let mono = to_mono(&interleaved, channels);
        resample_mono(mono, track.sample_rate, WHISPER_SAMPLE_RATE)
    }
}

/// Opens and probes `path` into a Symphonia format reader.
fn open_format(path: &Path) -> Result<Box<dyn FormatReader>> {
    let file = File::open(path).map_err(|source| Error::AudioIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::AudioDecode(e.to_string()))?;
    Ok(probed.format)
}

/// Selects the first non-null codec track and copies its decode parameters.
fn select_audio_track(format: &dyn FormatReader, path: &Path) -> Result<TrackInfo> {
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Error::NoAudioTrack {
            path: path.to_path_buf(),
        })?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| Error::AudioDecode("missing sample rate".into()))?;
    let metadata_channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .filter(|&n| n > 0);

    Ok(TrackInfo {
        track_id: track.id,
        sample_rate,
        metadata_channels,
        codec_params: track.codec_params.clone(),
    })
}

/// Decodes all packets for `track` into interleaved PCM and a channel count.
fn decode_interleaved(
    format: &mut dyn FormatReader,
    track: &TrackInfo,
    max_duration_secs: u64,
) -> Result<(Vec<f32>, usize)> {
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::AudioDecode(e.to_string()))?;

    let mut interleaved = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut channels = track.metadata_channels;

    while let Some(packet) = next_track_packet(format, track.track_id)? {
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let decoded_channels = decoded.spec().channels.count();
                let channel_count = resolve_channels(channels, decoded_channels)?;
                channels = Some(channel_count);
                append_decoded(&decoded, &mut sample_buf, &mut interleaved);
                let frames = interleaved.len() / channel_count;
                if duration_exceeded(frames, track.sample_rate, max_duration_secs) {
                    return Err(Error::AudioTooLong {
                        max_secs: max_duration_secs,
                    });
                }
            }
            // Clear local PCM state and continue after a decoder-level reset.
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                sample_buf = None;
                channels = track.metadata_channels;
            }
            Err(e) => return Err(Error::AudioDecode(e.to_string())),
        }
    }

    if interleaved.is_empty() {
        return Err(Error::AudioDecode("no audio samples decoded".into()));
    }

    let channels = channels.ok_or_else(|| Error::AudioDecode("missing channel count".into()))?;
    Ok((interleaved, channels))
}

/// Reads the next packet for `track_id`, or `None` at end of stream.
fn next_track_packet(format: &mut dyn FormatReader, track_id: u32) -> Result<Option<Packet>> {
    loop {
        match format.next_packet() {
            Ok(packet) if packet.track_id() == track_id => return Ok(Some(packet)),
            Ok(_) => continue,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            // Format-level reset needs track re-probe; that path is unsupported.
            Err(SymphoniaError::ResetRequired) => {
                return Err(Error::UnsupportedBitstreamReset);
            }
            Err(e) => return Err(Error::AudioDecode(e.to_string())),
        }
    }
}

/// Resolves channel count from optional metadata and a decoded packet.
///
/// When `known` is missing, uses `decoded`. When both are present they must
/// match. A decoded count of zero is always an error.
fn resolve_channels(known: Option<usize>, decoded: usize) -> Result<usize> {
    if decoded == 0 {
        return Err(Error::AudioDecode("missing channel count".into()));
    }
    match known {
        None => Ok(decoded),
        Some(n) if n == decoded => Ok(n),
        Some(n) => Err(Error::AudioDecode(format!(
            "channel count mismatch: metadata={n}, decoded={decoded}"
        ))),
    }
}

/// Returns true when decoded `frames` exceed `max_secs` at `sample_rate`.
fn duration_exceeded(frames: usize, sample_rate: u32, max_secs: u64) -> bool {
    if sample_rate == 0 {
        return true;
    }
    let max_frames = max_secs.saturating_mul(u64::from(sample_rate));
    frames as u64 > max_frames
}

fn append_decoded(
    decoded: &AudioBufferRef<'_>,
    sample_buf: &mut Option<SampleBuffer<f32>>,
    out: &mut Vec<f32>,
) {
    let spec = *decoded.spec();
    let frames = decoded.capacity() as u64;
    let needed = decoded.capacity() * spec.channels.count();
    let needs_new = sample_buf
        .as_ref()
        .map(|buf| buf.capacity() < needed)
        .unwrap_or(true);
    if needs_new {
        *sample_buf = Some(SampleBuffer::<f32>::new(frames, spec));
    }
    if let Some(buf) = sample_buf.as_mut() {
        buf.copy_interleaved_ref(decoded.clone());
        out.extend_from_slice(buf.samples());
    }
}

fn to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_mono(samples: Vec<f32>, from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate || samples.is_empty() {
        return Ok(samples);
    }

    let expected_len =
        (samples.len() as f64 * f64::from(to_rate) / f64::from(from_rate)).round() as usize;
    let mut resampler = build_resampler(from_rate, to_rate, samples.len())?;
    process_and_trim(&mut resampler, samples, expected_len)
}

/// Builds a fixed-input sinc resampler for a mono buffer of `input_len` samples.
fn build_resampler(from_rate: u32, to_rate: u32, input_len: usize) -> Result<SincFixedIn<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    SincFixedIn::<f32>::new(
        f64::from(to_rate) / f64::from(from_rate),
        2.0,
        params,
        input_len,
        1,
    )
    .map_err(|e| Error::AudioDecode(format!("resampler init failed: {e}")))
}

/// Processes `samples` through `resampler`, flushes delay, and trims to length.
fn process_and_trim(
    resampler: &mut SincFixedIn<f32>,
    samples: Vec<f32>,
    expected_len: usize,
) -> Result<Vec<f32>> {
    let delay = resampler.output_delay();
    let waves_in = vec![samples];
    let mut waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| Error::AudioDecode(format!("resample failed: {e}")))?;

    let mut out = waves_out.pop().unwrap_or_default();
    let target = expected_len.saturating_add(delay);
    while out.len() < target {
        let before = out.len();
        let flushed = resampler
            .process_partial::<Vec<f32>>(None, None)
            .map_err(|e| Error::AudioDecode(format!("resample flush failed: {e}")))?;
        let chunk = flushed.into_iter().next().unwrap_or_default();
        if chunk.is_empty() {
            break;
        }
        out.extend_from_slice(&chunk);
        if out.len() == before {
            break;
        }
    }

    let start = delay.min(out.len());
    let end = (start + expected_len).min(out.len());
    Ok(out[start..end].to_vec())
}

#[cfg(test)]
mod tests {
    use symphonia::core::audio::{AsAudioBufferRef, AudioBuffer, Layout, Signal, SignalSpec};

    use super::*;

    #[test]
    fn mono_averages_channels() {
        let stereo = vec![1.0, 3.0, 2.0, 4.0];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono, vec![2.0, 3.0]);
    }

    #[test]
    fn resolve_channels_uses_decoded_when_metadata_missing() {
        assert_eq!(resolve_channels(None, 2).unwrap(), 2);
    }

    #[test]
    fn resolve_channels_accepts_matching_metadata() {
        assert_eq!(resolve_channels(Some(2), 2).unwrap(), 2);
    }

    #[test]
    fn resolve_channels_rejects_metadata_mismatch() {
        let err = resolve_channels(Some(1), 2).unwrap_err();
        assert!(matches!(err, Error::AudioDecode(_)));
        assert!(err.to_string().contains("channel count mismatch"));
    }

    #[test]
    fn resolve_channels_rejects_zero_decoded() {
        let err = resolve_channels(None, 0).unwrap_err();
        assert!(matches!(err, Error::AudioDecode(_)));
        assert!(err.to_string().contains("missing channel count"));
    }

    #[test]
    fn unsupported_bitstream_reset_is_typed_error() {
        let err = Error::UnsupportedBitstreamReset;
        assert_eq!(err.to_string(), "bitstream reset is not supported");
        assert!(matches!(err, Error::UnsupportedBitstreamReset));
    }

    #[test]
    fn resample_same_rate_is_identity() {
        let samples = vec![0.1, 0.2, 0.3];
        let out = resample_mono(samples.clone(), 16_000, 16_000).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn resample_44100_to_16000_preserves_expected_length() {
        let from_rate = 44_100u32;
        let to_rate = 16_000u32;
        let samples = vec![0.1_f32; from_rate as usize]; // one second
        let expected =
            (samples.len() as f64 * f64::from(to_rate) / f64::from(from_rate)).round() as usize;

        let out = resample_mono(samples, from_rate, to_rate).unwrap();
        assert!(
            out.len().abs_diff(expected) <= 1,
            "got {} samples, expected about {}",
            out.len(),
            expected
        );
        assert_eq!(out.len(), to_rate as usize);
    }

    #[test]
    fn append_decoded_grows_sample_buffer_for_larger_packets() {
        let small = filled_mono_buffer(8, 0.25);
        let large = filled_mono_buffer(64, 0.5);
        let mut sample_buf = None;
        let mut out = Vec::new();

        append_decoded(&small.as_audio_buffer_ref(), &mut sample_buf, &mut out);
        assert_eq!(out.len(), 8);

        append_decoded(&large.as_audio_buffer_ref(), &mut sample_buf, &mut out);
        assert_eq!(out.len(), 8 + 64);
        assert!(sample_buf.unwrap().capacity() >= 64);
    }

    fn filled_mono_buffer(frames: u64, value: f32) -> AudioBuffer<f32> {
        let spec = SignalSpec::new_with_layout(16_000, Layout::Mono);
        let mut buf = AudioBuffer::<f32>::new(frames, spec);
        buf.render_reserved(Some(frames as usize));
        for sample in buf.chan_mut(0) {
            *sample = value;
        }
        buf
    }

    #[test]
    fn duration_exceeded_respects_limit() {
        assert!(!duration_exceeded(15_999, 16_000, 1));
        assert!(!duration_exceeded(16_000, 16_000, 1));
        assert!(duration_exceeded(16_001, 16_000, 1));
        assert!(duration_exceeded(1, 16_000, 0));
    }

    #[test]
    fn decode_file_rejects_invalid_audio() {
        use std::io::Write;

        let path = std::env::temp_dir().join("earshot-invalid-audio.bin");
        {
            let mut file = File::create(&path).unwrap();
            file.write_all(b"not a real audio bitstream").unwrap();
        }

        let result = SymphoniaDecoder::default().decode_file(&path);
        let _ = std::fs::remove_file(&path);

        assert!(matches!(
            result,
            Err(Error::AudioDecode(_)) | Err(Error::NoAudioTrack { .. })
        ));
    }

    #[test]
    fn decode_file_rejects_audio_over_max_duration() {
        let path = std::env::temp_dir().join("earshot-duration-cap.wav");
        write_minimal_wav(&path, 16_000, &[0_i16; 100]);

        let decoder = SymphoniaDecoder {
            max_duration_secs: 0,
        };
        let result = decoder.decode_file(&path);
        let _ = std::fs::remove_file(&path);

        assert!(matches!(result, Err(Error::AudioTooLong { max_secs: 0 })));
    }

    fn write_minimal_wav(path: &Path, sample_rate: u32, samples: &[i16]) {
        use std::io::Write;

        let data_bytes = samples.len() * 2;
        let mut bytes = Vec::with_capacity(44 + data_bytes);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let mut file = File::create(path).unwrap();
        file.write_all(&bytes).unwrap();
    }
}
