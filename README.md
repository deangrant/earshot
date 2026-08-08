# earshot

Fast Whisper speech-to-text for Rust. Earshot wraps [whisper.cpp](https://github.com/ggerganov/whisper.cpp) through [`whisper-rs`](https://crates.io/crates/whisper-rs) for efficient CPU (and optional NVIDIA CUDA) inference, returns timed transcript segments, and detects the spoken language.

Audio is decoded with [Symphonia](https://crates.io/crates/symphonia)—no separate FFmpeg install is required.

## Features

- Transcribe MP3, WAV, FLAC, and AAC (MP4/M4A)
- Timed segments (`start` / `end` in seconds) plus full transcript text
- Automatic language detection (or force a language code)
- CPU by default; enable the `cuda` feature for NVIDIA GPUs
- Int8 / quantized GGML models for lower memory use (`ComputeType::Int8`)
- Default 2-hour decode cap (`MAX_AUDIO_DURATION_SECS`) to bound memory; override via `SymphoniaDecoder::max_duration_secs`

## Install

### Build requirements

Compiling `whisper-rs` (and therefore earshot) needs:

- A Rust toolchain (edition 2021)
- CMake
- A C/C++ compiler (`cc` / `c++`)
- `libclang` (for bindgen), e.g. `libclang-dev` on Debian/Ubuntu

FFmpeg is **not** required at build or runtime.

### Dependency

Add the crate to your project:

```toml
[dependencies]
earshot = { path = "." }
```

For NVIDIA GPU acceleration:

```toml
[dependencies]
earshot = { path = ".", features = ["cuda"] }
```

Building with `cuda` requires a working CUDA toolkit on the machine that compiles the crate.

## Models

Download a GGML Whisper model (prefer a quantized file such as `q8_0` for `ComputeType::Int8`), for example from the [whisper.cpp model collection](https://huggingface.co/ggerganov/whisper.cpp).

Place the `.bin` file somewhere on disk and pass its path to `WhisperModel::load`.

### Security

GGML model files are loaded by native whisper.cpp code and are a **trust boundary**. Only use models from origins you trust. Do not load arbitrary remote or user-supplied `.bin` files without your own validation and trust policy—corrupt or adversarial files may crash the process or worse.

## Example

```bash
cargo run --example transcribe -- path/to/ggml-base.en-q8_0.bin path/to/audio.mp3
```

With CUDA:

```bash
cargo run --example transcribe --features cuda -- path/to/model.bin path/to/audio.mp3
```

## Library usage

```rust
use earshot::{ComputeType, Device, ModelConfig, TranscribeConfig, WhisperModel};

let model = WhisperModel::load(
    "models/ggml-base.en-q8_0.bin",
    ModelConfig::default()
        .device(Device::Cpu)
        .compute_type(ComputeType::Int8),
)?;

let result = model.transcribe_file("audio.mp3", TranscribeConfig::default())?;
println!("Detected language: {}", result.language);
for segment in &result.segments {
    println!("[{:.2}-{:.2}] {}", segment.start, segment.end, segment.text);
}
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
