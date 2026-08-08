//! Transcription orchestration over a loaded Whisper model.

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy};

use crate::audio::{AudioDecoder, SymphoniaDecoder};
use crate::config::TranscribeConfig;
use crate::error::{Error, Result};
use crate::model::WhisperModel;
use crate::types::{Segment, TranscriptionResult};

/// Runs speech-to-text over PCM or audio files.
///
/// This trait is object-safe and can be used as [`dyn Transcriber`].
pub trait Transcriber {
    /// Transcribes mono 16 kHz `f32` PCM samples.
    ///
    /// # Errors
    ///
    /// Returns an error when the Whisper backend fails, language metadata
    /// cannot be read, or [`TranscribeConfig::language`] contains a null byte
    /// (which would panic inside whisper-rs).
    fn transcribe_samples(
        &self,
        samples: &[f32],
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult>;

    /// Decodes `path` with `decoder` and transcribes the PCM.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails, including
    /// [`Error::AudioTooLong`] when decoded audio exceeds the decoder limit.
    fn transcribe_file_with(
        &self,
        decoder: &dyn AudioDecoder,
        path: &Path,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        let samples = decoder.decode_file(path)?;
        self.transcribe_samples(&samples, config)
    }

    /// Decodes `path` with the default Symphonia decoder and transcribes it.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails, including
    /// [`Error::AudioTooLong`] when decoded audio exceeds the decoder limit.
    fn transcribe_file(
        &self,
        path: &Path,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        self.transcribe_file_with(&SymphoniaDecoder::default(), path, config)
    }
}

impl WhisperModel {
    /// Decodes an audio file and transcribes it with the default decoder.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails, including
    /// [`Error::AudioTooLong`] when decoded audio exceeds the default
    /// maximum duration ([`crate::MAX_AUDIO_DURATION_SECS`]).
    pub fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        Transcriber::transcribe_file(self, path.as_ref(), config)
    }

    /// Decodes an audio file with `decoder` and transcribes it.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails.
    pub fn transcribe_file_with(
        &self,
        decoder: &impl AudioDecoder,
        path: impl AsRef<Path>,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        Transcriber::transcribe_file_with(self, decoder, path.as_ref(), config)
    }

    /// Transcribes pre-decoded mono 16 kHz PCM samples.
    ///
    /// # Errors
    ///
    /// Returns an error when the Whisper backend fails, or when
    /// [`TranscribeConfig::language`] contains a null byte.
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        Transcriber::transcribe_samples(self, samples, config)
    }
}

impl Transcriber for WhisperModel {
    fn transcribe_samples(
        &self,
        samples: &[f32],
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        if samples.is_empty() {
            return Err(Error::InvalidConfig(
                "audio samples must not be empty".into(),
            ));
        }

        let mut state = self
            .context()
            .create_state()
            .map_err(|e| Error::Transcription(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(config.n_threads.max(1));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Use language "auto" for detection-then-transcribe. Do not set
        // detect_language(true): in whisper.cpp that flag means detect and exit.
        // whisper-rs panics on null bytes inside set_language; reject them here.
        let language_owned = config.language.clone();
        let language = language_param(language_owned.as_deref())?;
        params.set_language(Some(language));

        state
            .full(params, samples)
            .map_err(|e| Error::Transcription(e.to_string()))?;

        let language = detected_language(&state)?;
        let segments = collect_segments(&state)?;
        Ok(TranscriptionResult::from_segments(language, segments))
    }
}

/// Returns the language string for whisper-rs, or an error if it is unsafe.
///
/// whisper-rs converts the value with `CString::new(...).expect(...)`, so
/// embedded null bytes would panic; reject them as invalid configuration.
fn language_param(language: Option<&str>) -> Result<&str> {
    match language {
        None => Ok("auto"),
        Some(lang) if lang.contains('\0') => Err(Error::InvalidConfig(
            "language must not contain null bytes".into(),
        )),
        Some(lang) => Ok(lang),
    }
}

fn detected_language(state: &whisper_rs::WhisperState) -> Result<String> {
    let id = state.full_lang_id_from_state();
    if id < 0 {
        return Ok("unknown".into());
    }
    whisper_rs::get_lang_str(id)
        .map(str::to_owned)
        .ok_or_else(|| Error::Transcription(format!("unknown language id {id}")))
}

fn collect_segments(state: &whisper_rs::WhisperState) -> Result<Vec<Segment>> {
    let n = state.full_n_segments();
    if n < 0 {
        return Err(Error::Transcription(
            "negative segment count from backend".into(),
        ));
    }

    let mut segments = Vec::with_capacity(n as usize);
    for i in 0..n {
        let seg = state
            .get_segment(i)
            .ok_or_else(|| Error::Transcription(format!("missing segment at index {i}")))?;
        let text = seg
            .to_str()
            .map_err(|e| Error::Transcription(e.to_string()))?
            .to_owned();
        segments.push(Segment::from_centiseconds(
            seg.start_timestamp(),
            seg.end_timestamp(),
            text,
        ));
    }
    Ok(segments)
}

const _: Option<&dyn Transcriber> = None;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_param_defaults_to_auto() {
        assert_eq!(language_param(None).unwrap(), "auto");
    }

    #[test]
    fn language_param_passes_through_valid_codes() {
        assert_eq!(language_param(Some("en")).unwrap(), "en");
    }

    #[test]
    fn language_param_rejects_null_bytes() {
        let err = language_param(Some("en\0")).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
        assert!(err.to_string().contains("null bytes"));
    }

    struct StubTranscriber;

    impl Transcriber for StubTranscriber {
        fn transcribe_samples(
            &self,
            samples: &[f32],
            _config: TranscribeConfig,
        ) -> Result<TranscriptionResult> {
            if samples.is_empty() {
                return Err(Error::InvalidConfig(
                    "audio samples must not be empty".into(),
                ));
            }
            Ok(TranscriptionResult::from_segments("en", vec![]))
        }
    }

    struct EmptyDecoder;

    impl AudioDecoder for EmptyDecoder {
        fn decode_file(&self, _path: &Path) -> Result<Vec<f32>> {
            Ok(Vec::new())
        }
    }

    struct FixedDecoder;

    impl AudioDecoder for FixedDecoder {
        fn decode_file(&self, _path: &Path) -> Result<Vec<f32>> {
            Ok(vec![0.0; 16])
        }
    }

    #[test]
    fn transcribe_file_with_uses_injected_decoder() {
        let err = StubTranscriber
            .transcribe_file_with(
                &EmptyDecoder,
                Path::new("unused.wav"),
                TranscribeConfig::default(),
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    #[test]
    fn transcribe_file_with_forwards_decoded_samples() {
        let result = StubTranscriber
            .transcribe_file_with(
                &FixedDecoder,
                Path::new("unused.wav"),
                TranscribeConfig::default(),
            )
            .unwrap();
        assert_eq!(result.language, "en");
        assert!(result.segments.is_empty());
    }
}
