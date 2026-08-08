//! Audio decoding into Whisper-ready PCM samples.

mod decoder;
mod symphonia;

#[doc(inline)]
pub use decoder::{AudioDecoder, WHISPER_SAMPLE_RATE};
#[doc(inline)]
pub use symphonia::SymphoniaDecoder;
