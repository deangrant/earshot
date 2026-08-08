//! Transcription orchestration over a loaded Whisper model.

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy};

use crate::audio::{AudioDecoder, SymphoniaDecoder};
use crate::config::TranscribeConfig;
use crate::error::{Error, Result};
use crate::model::WhisperModel;
use crate::types::{Segment, TranscriptionResult};

/// Runs speech-to-text over PCM or audio files.
pub trait Transcriber {
    /// Transcribes mono 16 kHz `f32` PCM samples.
    ///
    /// # Errors
    ///
    /// Returns an error when the Whisper backend fails or language metadata
    /// cannot be read.
    fn transcribe_samples(
        &self,
        samples: &[f32],
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult>;

    /// Decodes `path` and transcribes it.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails.
    fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult>;
}

impl WhisperModel {
    /// Decodes an audio file and transcribes it with the default decoder.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or transcription fails.
    pub fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        Transcriber::transcribe_file(self, path, config)
    }

    /// Transcribes pre-decoded mono 16 kHz PCM samples.
    ///
    /// # Errors
    ///
    /// Returns an error when the Whisper backend fails.
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
        let language_owned = config.language.clone();
        match language_owned.as_deref() {
            Some(lang) => params.set_language(Some(lang)),
            None => params.set_language(Some("auto")),
        }

        state
            .full(params, samples)
            .map_err(|e| Error::Transcription(e.to_string()))?;

        let language = detected_language(&state)?;
        let segments = collect_segments(&state)?;
        Ok(TranscriptionResult::from_segments(language, segments))
    }

    fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        config: TranscribeConfig,
    ) -> Result<TranscriptionResult> {
        let samples = SymphoniaDecoder.decode_file(path.as_ref())?;
        self.transcribe_samples(&samples, config)
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
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
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
