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
    /// [`ComputeType::Int8`] is selected. After open, the model's
    /// `model_ftype` is checked against [`ModelConfig::compute_type`].
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
    /// backend fails to open the model or rejects the path (including
    /// non-UTF-8 paths on platforms where the backend requires UTF-8).
    /// Returns [`Error::ComputeTypeMismatch`] when the loaded weights do not
    /// match [`ModelConfig::compute_type`].
    pub fn load(path: impl AsRef<Path>, config: ModelConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let params = context_params(&config)?;
        let ctx = WhisperContext::new_with_params(&path, params).map_err(|e| Error::ModelLoad {
            path: path.clone(),
            message: e.to_string(),
        })?;

        let actual_ftype = ctx.model_ftype();
        if !compute_type_matches(config.compute_type, actual_ftype) {
            return Err(Error::ComputeTypeMismatch {
                expected: config.compute_type,
                actual_ftype,
            });
        }

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
            params.flash_attn = false;
        }
        Device::Cuda { device_id } => {
            if !cfg!(feature = "cuda") {
                return Err(Error::CudaUnavailable);
            }
            params.use_gpu = true;
            params.gpu_device = device_id;
            params.flash_attn = true;
        }
    }

    Ok(params)
}

/// Returns whether `ftype` from whisper.cpp matches `compute`.
fn compute_type_matches(compute: ComputeType, ftype: i32) -> bool {
    match compute {
        ComputeType::Float32 => ftype == 0,
        ComputeType::Float16 => ftype == 1,
        ComputeType::Int8 => ftype >= 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_context_params_disable_gpu() {
        let cfg = ModelConfig::default().device(Device::Cpu);
        let params = context_params(&cfg).unwrap();
        assert!(!params.use_gpu);
        assert!(!params.flash_attn);
    }

    #[test]
    fn cuda_without_feature_errors() {
        let cfg = ModelConfig::default().device(Device::Cuda { device_id: 0 });
        let result = context_params(&cfg);
        if cfg!(feature = "cuda") {
            let params = result.unwrap();
            assert!(params.use_gpu);
            assert!(params.flash_attn);
        } else {
            assert!(matches!(result, Err(Error::CudaUnavailable)));
        }
    }

    #[test]
    fn cuda_enables_flash_attn_for_any_compute_type() {
        if !cfg!(feature = "cuda") {
            return;
        }
        for compute in [
            ComputeType::Float32,
            ComputeType::Float16,
            ComputeType::Int8,
        ] {
            let cfg = ModelConfig::default()
                .device(Device::Cuda { device_id: 0 })
                .compute_type(compute);
            let params = context_params(&cfg).unwrap();
            assert!(params.flash_attn, "compute={compute:?}");
        }
    }

    #[test]
    fn compute_type_matches_ftype_buckets() {
        assert!(compute_type_matches(ComputeType::Float32, 0));
        assert!(!compute_type_matches(ComputeType::Float32, 1));
        assert!(!compute_type_matches(ComputeType::Float32, 7));

        assert!(compute_type_matches(ComputeType::Float16, 1));
        assert!(!compute_type_matches(ComputeType::Float16, 0));
        assert!(!compute_type_matches(ComputeType::Float16, 7));

        assert!(compute_type_matches(ComputeType::Int8, 2));
        assert!(compute_type_matches(ComputeType::Int8, 7));
        assert!(!compute_type_matches(ComputeType::Int8, 0));
        assert!(!compute_type_matches(ComputeType::Int8, 1));
        assert!(!compute_type_matches(ComputeType::Int8, -1));
    }

    #[cfg(unix)]
    #[test]
    fn load_preserves_non_utf8_path_in_model_load_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let path = PathBuf::from(OsStr::from_bytes(b"model-\xFF.bin"));
        let err = WhisperModel::load(&path, ModelConfig::default()).unwrap_err();
        match err {
            Error::ModelLoad { path: err_path, .. } => assert_eq!(err_path, path),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }
}
