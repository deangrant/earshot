//! Whisper model loading and context ownership.

use std::path::{Path, PathBuf};

use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::config::{ComputeType, Device, ModelConfig};
use crate::error::{Error, Result};

/// A loaded Whisper model ready for transcription.
pub struct WhisperModel {
    ctx: WhisperContext,
    path: PathBuf,
    config: ModelConfig,
}

impl WhisperModel {
    /// Loads a GGML Whisper model from `path` using `config`.
    ///
    /// Prefer a quantized GGML file (for example `*-q8_0.bin`) when
    /// [`ComputeType::Int8`] is selected.
    ///
    /// # Security
    ///
    /// Model files are loaded and parsed by native whisper.cpp code. Treat
    /// `path` as a trust boundary: only load models from sources you trust
    /// (for example known publishers), not arbitrary user uploads. Corrupt or
    /// adversarial files may crash the process or worse; this is not a
    /// sandboxed loader.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CudaUnavailable`] when [`Device::Cuda`] is requested
    /// without the `cuda` feature. Returns [`Error::ModelLoad`] when the
    /// backend fails to open the model.
    pub fn load(path: impl AsRef<Path>, config: ModelConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let params = context_params(&config)?;
        let ctx = WhisperContext::new_with_params(path.to_string_lossy().as_ref(), params)
            .map_err(|e| Error::ModelLoad {
                path: path.clone(),
                message: e.to_string(),
            })?;

        Ok(Self { ctx, path, config })
    }

    /// Returns the path used to load this model.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the configuration used when loading this model.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub(crate) fn context(&self) -> &WhisperContext {
        &self.ctx
    }
}

impl std::fmt::Debug for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperModel")
            .field("path", &self.path)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn context_params(config: &ModelConfig) -> Result<WhisperContextParameters<'static>> {
    let mut params = WhisperContextParameters::default();
    match config.device {
        Device::Cpu => {
            params.use_gpu = false;
            params.gpu_device = 0;
        }
        Device::Cuda { device_id } => {
            if !cfg!(feature = "cuda") {
                return Err(Error::CudaUnavailable);
            }
            params.use_gpu = true;
            params.gpu_device = device_id;
        }
    }

    // Float16 benefits from flash attention when available on GPU builds.
    params.flash_attn = matches!(config.compute_type, ComputeType::Float16)
        && matches!(config.device, Device::Cuda { .. });

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_context_params_disable_gpu() {
        let cfg = ModelConfig::default().device(Device::Cpu);
        let params = context_params(&cfg).unwrap();
        assert!(!params.use_gpu);
    }

    #[test]
    fn cuda_without_feature_errors() {
        let cfg = ModelConfig::default().device(Device::Cuda { device_id: 0 });
        let result = context_params(&cfg);
        if cfg!(feature = "cuda") {
            assert!(result.is_ok());
        } else {
            assert!(matches!(result, Err(Error::CudaUnavailable)));
        }
    }
}
