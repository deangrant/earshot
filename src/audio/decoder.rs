//! Trait for decoding audio files into Whisper-ready PCM.

use std::path::Path;

use crate::error::Result;

/// Target sample rate expected by Whisper models.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Default maximum decoded audio duration in seconds (2 hours).
pub const MAX_AUDIO_DURATION_SECS: u64 = 7_200;

/// Decodes audio files into mono `f32` PCM at [`WHISPER_SAMPLE_RATE`].
pub trait AudioDecoder {
    /// Decodes `path` into mono f32 samples at 16 kHz.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened, contains no audio
    /// track, fails to decode, exceeds the decoder's maximum duration
    /// ([`Error::AudioTooLong`](crate::Error::AudioTooLong)), or requires an
    /// unsupported format-level bitstream reset
    /// ([`Error::UnsupportedBitstreamReset`](crate::Error::UnsupportedBitstreamReset)).
    fn decode_file(&self, path: &Path) -> Result<Vec<f32>>;
}
