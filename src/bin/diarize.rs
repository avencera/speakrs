use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;

use speakrs::clustering::plda::PldaTransform;
use speakrs::inference::ExecutionMode;
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::{FAST_SEGMENTATION_STEP_SECONDS, SEGMENTATION_STEP_SECONDS, diarize};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, wav_path) = parse_args(&args);
    if wav_path.is_empty() {
        std::process::exit(1);
    }

    if let CliMode::PyannoteDevice(device) = mode {
        let output = run_pyannote_sidecar(device, wav_path).expect("pyannote sidecar failed");
        print!("{output}");
        return;
    }

    let CliMode::Native(mode) = mode else {
        unreachable!();
    };
    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");

    let step = match mode {
        ExecutionMode::CoreMl | ExecutionMode::MiniCoreMl => FAST_SEGMENTATION_STEP_SECONDS,
        _ => SEGMENTATION_STEP_SECONDS,
    };
    let mut seg_model = SegmentationModel::with_mode(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        step as f32,
        mode,
    )
    .expect("failed to load segmentation model");
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
        mode,
    )
    .expect("failed to load embedding model");
    let plda = PldaTransform::from_dir(&models_dir).expect("failed to load PLDA parameters");

    let (samples, sr) = load_wav_samples(wav_path);
    assert_eq!(sr, 16000, "expected 16kHz WAV, got {sr}Hz");

    let result = diarize(&mut seg_model, &mut emb_model, &plda, &samples, "file1")
        .expect("diarization failed");
    print!("{}", result.rttm);
}

enum CliMode {
    Native(ExecutionMode),
    PyannoteDevice(&'static str),
}

fn parse_args(args: &[String]) -> (CliMode, &str) {
    match args {
        [_, wav_path] => (CliMode::Native(ExecutionMode::ExactCpu), wav_path),
        [_, flag, mode, wav_path] if flag == "--mode" => {
            let parsed = match mode.as_str() {
                "exact" => CliMode::Native(ExecutionMode::ExactCpu),
                "coreml" => CliMode::Native(ExecutionMode::CoreMl),
                "mini-coreml" => CliMode::Native(ExecutionMode::MiniCoreMl),
                "cuda" => CliMode::Native(ExecutionMode::Cuda),
                "pyannote-cpu" => CliMode::PyannoteDevice("cpu"),
                "pyannote-mps" => CliMode::PyannoteDevice("mps"),
                "pyannote-cuda" => CliMode::PyannoteDevice("cuda"),
                _ => {
                    eprintln!("Unknown mode: {mode}");
                    eprintln!(
                        "Usage: diarize [--mode exact|coreml|mini-coreml|cuda|pyannote-cpu|pyannote-mps|pyannote-cuda] <path/to/audio.wav>"
                    );
                    return (CliMode::Native(ExecutionMode::ExactCpu), "");
                }
            };
            (parsed, wav_path)
        }
        _ => {
            eprintln!(
                "Usage: diarize [--mode exact|coreml|mini-coreml|cuda|pyannote-cpu|pyannote-mps|pyannote-cuda] <path/to/audio.wav>"
            );
            (CliMode::Native(ExecutionMode::ExactCpu), "")
        }
    }
}

fn run_pyannote_sidecar(
    device: &'static str,
    wav_path: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/diarize_pyannote.py");
    let output = Command::new("uv")
        .arg("run")
        .arg(script_path)
        .arg("--device")
        .arg(device)
        .arg(wav_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pyannote sidecar exited with {}: {stderr}", output.status).into());
    }

    Ok(String::from_utf8(output.stdout)?)
}

fn load_wav_samples(path: &str) -> (Vec<f32>, u32) {
    let file = File::open(path).expect("failed to open WAV file");
    let mut reader = BufReader::new(file);
    let mut riff_header = [0u8; 12];
    reader
        .read_exact(&mut riff_header)
        .expect("failed to read WAV header");
    assert_eq!(&riff_header[0..4], b"RIFF", "expected RIFF WAV");
    assert_eq!(&riff_header[8..12], b"WAVE", "expected WAVE file");

    let mut sample_rate = None;
    let mut channels = None;
    let mut bits_per_sample = None;

    loop {
        let mut chunk_header = [0u8; 8];
        if reader.read_exact(&mut chunk_header).is_err() {
            break;
        }

        let chunk_id = &chunk_header[0..4];
        let chunk_size = u32::from_le_bytes(chunk_header[4..8].try_into().unwrap()) as usize;

        match chunk_id {
            b"fmt " => {
                let mut fmt = vec![0u8; chunk_size];
                reader
                    .read_exact(&mut fmt)
                    .expect("failed to read fmt chunk");
                let audio_format = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
                let chunk_channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
                let chunk_sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
                let chunk_bits_per_sample = u16::from_le_bytes(fmt[14..16].try_into().unwrap());

                assert_eq!(audio_format, 1, "expected PCM WAV");
                channels = Some(chunk_channels);
                sample_rate = Some(chunk_sample_rate);
                bits_per_sample = Some(chunk_bits_per_sample);
            }
            b"data" => {
                let sample_rate = sample_rate.expect("fmt chunk must appear before data chunk");
                let channels = channels.expect("missing channel count");
                let bits_per_sample = bits_per_sample.expect("missing bits per sample");
                assert_eq!(channels, 1, "expected mono WAV");
                assert_eq!(bits_per_sample, 16, "expected 16-bit PCM WAV");

                let mut samples = Vec::with_capacity(chunk_size / 2);
                let mut remaining = chunk_size;
                let mut buffer = [0u8; 8192];

                while remaining > 0 {
                    let to_read = remaining.min(buffer.len());
                    reader
                        .read_exact(&mut buffer[..to_read])
                        .expect("failed to read WAV samples");
                    for bytes in buffer[..to_read].chunks_exact(2) {
                        samples.push(i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0);
                    }
                    remaining -= to_read;
                }

                if chunk_size % 2 == 1 {
                    reader
                        .seek(SeekFrom::Current(1))
                        .expect("failed to skip WAV padding");
                }

                return (samples, sample_rate);
            }
            _ => {
                reader
                    .seek(SeekFrom::Current(chunk_size as i64))
                    .expect("failed to skip WAV chunk");
            }
        }

        if chunk_size % 2 == 1 {
            reader
                .seek(SeekFrom::Current(1))
                .expect("failed to skip WAV padding");
        }
    }

    panic!("no data chunk found in WAV");
}
