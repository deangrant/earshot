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
    /// Returns [`Error::InvalidConfig`] when `samples` is empty, when
    /// [`TranscribeConfig::n_threads`] is less than 1, or when
    /// [`TranscribeConfig::language`] contains a null byte (which would panic
    /// inside whisper-rs). Returns an error when the Whisper backend fails,
    /// language metadata cannot be read, or segment timestamps are
    /// inconsistent.
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
    /// [`Error::AudioTooLong`] when decoded audio exceeds the decoder limit,
    /// and the empty-PCM / config / timestamp failures described by
    /// [`Self::transcribe_samples`].
    fn transcribe_file_with(
        &self,
        decoder: &dyn AudioDecoder,
        path: &Path,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        let samples = decoder.decode_file(path)?;
        self.transcribe_samples(&samples, config)
    }
}

impl WhisperModel {
    /// Decodes an audio file and transcribes it with the default decoder.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails, including
    /// [`Error::AudioTooLong`] when decoded audio exceeds the default
    /// maximum duration ([`crate::MAX_AUDIO_DURATION_SECS`]), and the
    /// empty-PCM / config / timestamp failures described by
    /// [`Self::transcribe_samples`].
    pub fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        self.transcribe_file_with(&SymphoniaDecoder::default(), path, config)
    }

    /// Decodes an audio file with `decoder` and transcribes it.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails, including
    /// [`Error::AudioTooLong`] when decoded audio exceeds the decoder limit,
    /// decode failures from `decoder`, and the empty-PCM / config / timestamp
    /// failures described by [`Self::transcribe_samples`].
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
    /// Returns [`Error::InvalidConfig`] when `samples` is empty, when
    /// [`TranscribeConfig::n_threads`] is less than 1, or when
    /// [`TranscribeConfig::language`] contains a null byte. Returns an error
    /// when the Whisper backend fails or segment timestamps are inconsistent.
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
        validate_n_threads(config.n_threads)?;

        let mut state = self
            .context()
            .create_state()
            .map_err(|e| Error::Transcription(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(config.n_threads);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Use language "auto" for detection-then-transcribe. Do not set
        // detect_language(true): in whisper.cpp that flag means detect and
        // exit. Whisper-rs panics on null bytes inside set_language; reject
        // them here.
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

/// Returns the language string for whisper-rs, rejecting null bytes.
///
/// Whisper-rs converts the value with `CString::new(...).expect(...)`, so
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

fn validate_n_threads(n_threads: i32) -> Result<()> {
    if n_threads < 1 {
        return Err(Error::InvalidConfig(format!(
            "n_threads must be >= 1, got {n_threads}"
        )));
    }
    Ok(())
}

/// Checks segment centisecond times for end-before-start and non-monotonic starts.
fn validate_segment_times(index: i32, start: i64, end: i64, prev_start: Option<i64>) -> Result<()> {
    if end < start {
        return Err(Error::Transcription(format!(
            "segment {index} has end before start: start={start}, end={end}"
        )));
    }
    if let Some(prev) = prev_start {
        if start < prev {
            return Err(Error::Transcription(format!(
                "segment {index} starts before previous segment: start={start}, prev_start={prev}"
            )));
        }
    }
    Ok(())
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
    let mut prev_start: Option<i64> = None;
    for i in 0..n {
        let seg = state
            .get_segment(i)
            .ok_or_else(|| Error::Transcription(format!("missing segment at index {i}")))?;
        let start = seg.start_timestamp();
        let end = seg.end_timestamp();
        validate_segment_times(i, start, end, prev_start)?;
        prev_start = Some(start);
        let text = seg
            .to_str()
            .map_err(|e| Error::Transcription(e.to_string()))?
            .to_owned();
        segments.push(Segment::from_centiseconds(start, end, text));
    }
    Ok(segments)
}

// Ensures `Transcriber` stays object-safe (`dyn Transcriber`).
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

    #[test]
    fn validate_n_threads_accepts_positive() {
        assert!(validate_n_threads(1).is_ok());
        assert!(validate_n_threads(8).is_ok());
    }

    #[test]
    fn validate_n_threads_rejects_zero_and_negative() {
        for n in [0, -1] {
            let err = validate_n_threads(n).unwrap_err();
            assert!(matches!(err, Error::InvalidConfig(_)));
            assert!(err.to_string().contains("n_threads"));
        }
    }

    #[test]
    fn validate_segment_times_accepts_ordered() {
        assert!(validate_segment_times(0, 0, 100, None).is_ok());
        assert!(validate_segment_times(1, 100, 200, Some(0)).is_ok());
        assert!(validate_segment_times(2, 100, 100, Some(100)).is_ok());
    }

    #[test]
    fn validate_segment_times_rejects_end_before_start() {
        let err = validate_segment_times(0, 200, 100, None).unwrap_err();
        assert!(matches!(err, Error::Transcription(_)));
        assert!(err.to_string().contains("end before start"));
    }

    #[test]
    fn validate_segment_times_rejects_non_monotonic_starts() {
        let err = validate_segment_times(1, 50, 100, Some(100)).unwrap_err();
        assert!(matches!(err, Error::Transcription(_)));
        assert!(err.to_string().contains("starts before previous"));
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

    struct FailingDecoder;

    impl AudioDecoder for FailingDecoder {
        fn decode_file(&self, _path: &Path) -> Result<Vec<f32>> {
            Err(Error::AudioDecode("no audio samples decoded".into()))
        }
    }

    struct FixedDecoder;

    impl AudioDecoder for FixedDecoder {
        fn decode_file(&self, _path: &Path) -> Result<Vec<f32>> {
            Ok(vec![0.0; 16])
        }
    }

    #[test]
    fn transcribe_samples_rejects_empty_pcm() {
        let err = StubTranscriber
            .transcribe_samples(&[], TranscribeConfig::default())
            .unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn transcribe_file_with_propagates_decode_errors() {
        let err = StubTranscriber
            .transcribe_file_with(
                &FailingDecoder,
                Path::new("unused.wav"),
                TranscribeConfig::default(),
            )
            .unwrap_err();
        assert!(matches!(err, Error::AudioDecode(_)));
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
