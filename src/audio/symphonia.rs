//! Symphonia-backed audio decoding without an external FFmpeg install.

use std::fs::File;
use std::path::Path;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::audio::decoder::{AudioDecoder, WHISPER_SAMPLE_RATE};
use crate::error::{Error, Result};

/// Decodes common audio formats via Symphonia and resamples to 16 kHz mono.
#[derive(Debug, Default, Clone, Copy)]
pub struct SymphoniaDecoder;

impl AudioDecoder for SymphoniaDecoder {
    fn decode_file(&self, path: &Path) -> Result<Vec<f32>> {
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

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| Error::NoAudioTrack {
                path: path.to_path_buf(),
            })?;

        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| Error::AudioDecode("missing sample rate".into()))?;
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count())
            .unwrap_or(1)
            .max(1);

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| Error::AudioDecode(e.to_string()))?;

        let mut interleaved = Vec::new();
        let mut sample_buf: Option<SampleBuffer<f32>> = None;

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(SymphoniaError::ResetRequired) => {
                    return Err(Error::AudioDecode(
                        "bitstream reset is not supported".into(),
                    ));
                }
                Err(e) => return Err(Error::AudioDecode(e.to_string())),
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    append_decoded(&decoded, &mut sample_buf, &mut interleaved);
                }
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(SymphoniaError::IoError(_)) => continue,
                Err(e) => return Err(Error::AudioDecode(e.to_string())),
            }
        }

        let mono = to_mono(&interleaved, channels);
        resample_mono(mono, sample_rate, WHISPER_SAMPLE_RATE)
    }
}

fn append_decoded(
    decoded: &AudioBufferRef<'_>,
    sample_buf: &mut Option<SampleBuffer<f32>>,
    out: &mut Vec<f32>,
) {
    let spec = *decoded.spec();
    let duration = decoded.capacity() as u64;
    if sample_buf.is_none() {
        *sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
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

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        f64::from(to_rate) / f64::from(from_rate),
        2.0,
        params,
        samples.len(),
        1,
    )
    .map_err(|e| Error::AudioDecode(format!("resampler init failed: {e}")))?;

    let waves_in = vec![samples];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| Error::AudioDecode(format!("resample failed: {e}")))?;

    Ok(waves_out.into_iter().next().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_averages_channels() {
        let stereo = vec![1.0, 3.0, 2.0, 4.0];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono, vec![2.0, 3.0]);
    }

    #[test]
    fn resample_same_rate_is_identity() {
        let samples = vec![0.1, 0.2, 0.3];
        let out = resample_mono(samples.clone(), 16_000, 16_000).unwrap();
        assert_eq!(out, samples);
    }
}
