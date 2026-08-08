//! Fast Whisper speech-to-text with timed segments and language detection.
//!
//! Earshot loads GGML Whisper models via whisper.cpp, decodes audio with a
//! pure-Rust pipeline (no FFmpeg install), and returns timed transcript
//! segments plus the detected spoken language.
//!
//! Model files are a native-code trust boundary: only load GGML weights from
//! sources you trust. See [`WhisperModel::load`] for details.
//!
//! # Examples
//!
//! ```no_run
//! use earshot::{ComputeType, Device, ModelConfig, TranscribeConfig, WhisperModel};
//!
//! let model = WhisperModel::load(
//!     "models/ggml-base.en-q8_0.bin",
//!     ModelConfig::default()
//!         .device(Device::Cpu)
//!         .compute_type(ComputeType::Int8),
//! )?;
//!
//! let result = model.transcribe_file("audio.mp3", TranscribeConfig::default())?;
//! println!("language: {}", result.language);
//! for segment in &result.segments {
//!     println!("[{:.2}-{:.2}] {}", segment.start, segment.end, segment.text);
//! }
//! # Ok::<(), earshot::Error>(())
//! ```

#![deny(missing_docs)]

pub mod audio;
pub mod config;
pub mod engine;
pub mod error;
pub mod model;
pub mod types;

#[doc(inline)]
pub use audio::{AudioDecoder, SymphoniaDecoder, MAX_AUDIO_DURATION_SECS, WHISPER_SAMPLE_RATE};
#[doc(inline)]
pub use config::{ComputeType, Device, ModelConfig, TranscribeConfig};
#[doc(inline)]
pub use engine::Transcriber;
#[doc(inline)]
pub use error::{Error, Result};
#[doc(inline)]
pub use model::WhisperModel;
#[doc(inline)]
pub use types::{Segment, TranscriptionResult};
