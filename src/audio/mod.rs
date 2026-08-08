//! Audio decoding into Whisper-ready PCM samples.

mod decoder;
mod symphonia;

#[doc(inline)]
pub use decoder::{AudioDecoder, MAX_AUDIO_DURATION_SECS, WHISPER_SAMPLE_RATE};
#[doc(inline)]
pub use symphonia::SymphoniaDecoder;
