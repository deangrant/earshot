//! Trait for decoding audio files into Whisper-ready PCM.

use std::path::Path;

use crate::error::Result;

/// Target sample rate expected by Whisper models.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decodes audio files into mono `f32` PCM at [`WHISPER_SAMPLE_RATE`].
pub trait AudioDecoder {
    /// Decodes `path` into mono f32 samples at 16 kHz.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened, contains no audio
    /// track, or fails to decode.
    fn decode_file(&self, path: &Path) -> Result<Vec<f32>>;
}
