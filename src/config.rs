//! Configuration for model loading and transcription.

/// Hardware device used for Whisper inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Run on the CPU (default).
    #[default]
    Cpu,
    /// Run on an NVIDIA GPU via CUDA.
    ///
    /// Requires building with the `cuda` feature.
    Cuda {
        /// Zero-based CUDA device index.
        device_id: i32,
    },
}

/// Expected GGML weight format validated when the model is loaded.
///
/// Whisper.cpp stores precision/quantization in the model file. Earshot checks
/// the loaded `model_ftype` against this value. Choose a matching GGML file
/// (for example a `q8_0` model for [`ComputeType::Int8`]).
///
/// [`ComputeType::Int8`] means a quantized GGML weight file (Q4/Q5/Q8/etc.),
/// not a runtime cast of an F16/F32 model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeType {
    /// Full 32-bit floating point weights (`model_ftype` 0).
    Float32,
    /// 16-bit floating point weights (`model_ftype` 1).
    Float16,
    /// Quantized GGML weights such as Q4/Q5/Q8 (`model_ftype` >= 2).
    #[default]
    Int8,
}

/// Options applied when loading a Whisper model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    /// Device used for inference.
    pub device: Device,
    /// Expected model precision / quantization.
    pub compute_type: ComputeType,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            compute_type: ComputeType::Int8,
        }
    }
}

impl ModelConfig {
    /// Sets the inference device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Sets the expected compute / quantization type.
    pub fn compute_type(mut self, compute_type: ComputeType) -> Self {
        self.compute_type = compute_type;
        self
    }
}

/// Options applied to a single transcription request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscribeConfig {
    /// Forced language code, or `None` to auto-detect then transcribe.
    pub language: Option<String>,
    /// Number of threads used for CPU inference.
    pub n_threads: i32,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            language: None,
            n_threads: num_cpus_hint(),
        }
    }
}

impl TranscribeConfig {
    /// Forces transcription in the given language code (for example `"en"`).
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Enables automatic language detection before transcription.
    pub fn detect_language(mut self) -> Self {
        self.language = None;
        self
    }

    /// Sets the CPU thread count used during inference.
    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }
}

fn num_cpus_hint() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .clamp(1, 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_builder() {
        let cfg = ModelConfig::default()
            .device(Device::Cuda { device_id: 0 })
            .compute_type(ComputeType::Float16);
        assert_eq!(cfg.device, Device::Cuda { device_id: 0 });
        assert_eq!(cfg.compute_type, ComputeType::Float16);
    }

    #[test]
    fn transcribe_config_defaults_to_auto_language() {
        let cfg = TranscribeConfig::default();
        assert!(cfg.language.is_none());
        assert!(cfg.n_threads >= 1);
    }
}
