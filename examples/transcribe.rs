//! Load a Whisper model and transcribe an audio file.
//!
//! Usage:
//! ```text
//! cargo run --example transcribe -- path/to/model.bin path/to/audio.mp3
//! ```
//!
//! Optional GPU build:
//! ```text
//! cargo run --example transcribe --features cuda -- model.bin audio.mp3
//! ```

use std::env;
use std::process::ExitCode;

use earshot::{ComputeType, Device, ModelConfig, TranscribeConfig, WhisperModel};

fn main() -> ExitCode {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: transcribe <MODEL> <AUDIO>"))?;
    let audio_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: transcribe <MODEL> <AUDIO>"))?;

    let device = if cfg!(feature = "cuda") {
        Device::Cuda { device_id: 0 }
    } else {
        Device::Cpu
    };

    let model = WhisperModel::load(
        &model_path,
        ModelConfig::default()
            .device(device)
            .compute_type(ComputeType::Int8),
    )?;

    let result = model.transcribe_file(&audio_path, TranscribeConfig::default())?;

    println!("language: {}", result.language);
    println!("text: {}", result.text);
    println!();
    for segment in &result.segments {
        println!(
            "[{:7.2} -> {:7.2}] {}",
            segment.start, segment.end, segment.text
        );
    }

    Ok(())
}
