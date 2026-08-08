//! Error types returned by earshot operations.

use std::path::PathBuf;

use crate::config::ComputeType;

/// Errors that can occur while loading models or transcribing audio.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The audio file could not be opened or read.
    #[error("failed to open audio file `{path}`: {source}")]
    AudioIo {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// The audio container or codec is unsupported or corrupt.
    #[error("failed to decode audio: {0}")]
    AudioDecode(String),

    /// The demuxer requested a bitstream reset that this decoder does not support.
    ///
    /// Symphonia can emit `ResetRequired` after seeks or discontinuities. Earshot
    /// recovers from decoder-level resets, but format-level resets (re-probe track
    /// list / recreate demuxer state) are rejected as hard errors.
    #[error("bitstream reset is not supported")]
    UnsupportedBitstreamReset,

    /// The decoded audio exceeds the configured maximum duration.
    #[error("audio longer than {max_secs} seconds is not supported")]
    AudioTooLong {
        /// Maximum allowed duration in seconds.
        max_secs: u64,
    },

    /// No decodable audio track was found in the file.
    #[error("no audio track found in `{path}`")]
    NoAudioTrack {
        /// Path that lacked an audio track.
        path: PathBuf,
    },

    /// The Whisper model file could not be loaded.
    #[error("failed to load model `{path}`: {message}")]
    ModelLoad {
        /// Path to the model file.
        path: PathBuf,
        /// Backend error message.
        message: String,
    },

    /// The loaded model weights do not match [`ModelConfig::compute_type`](crate::ModelConfig).
    #[error("model compute type mismatch: config={expected:?}, model_ftype={actual_ftype}")]
    ComputeTypeMismatch {
        /// Compute type requested in configuration.
        expected: ComputeType,
        /// GGML `model_ftype` reported by the loaded model.
        actual_ftype: i32,
    },

    /// Transcription failed inside the Whisper backend.
    #[error("transcription failed: {0}")]
    Transcription(String),

    /// CUDA was requested but the crate was built without the `cuda` feature.
    #[error("CUDA support requires building with the `cuda` feature")]
    CudaUnavailable,

    /// A configuration value was invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Convenient result alias for earshot APIs.
pub type Result<T> = std::result::Result<T, Error>;
