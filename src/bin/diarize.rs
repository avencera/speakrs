use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use speakrs::clustering::plda::PldaTransform;
use speakrs::inference::ExecutionMode;
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
#[cfg(feature = "online")]
use speakrs::models::ModelManager;
use speakrs::pipeline::{FAST_SEGMENTATION_STEP_SECONDS, SEGMENTATION_STEP_SECONDS, diarize};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let (mode, wav_paths) = parse_args(&args);
    if wav_paths.is_empty() {
        std::process::exit(1);
    }

    if let CliMode::PyannoteDevice(device) = mode {
        let output = run_pyannote_sidecar(device, wav_paths[0]).expect("pyannote sidecar failed");
        print!("{output}");
        return;
    }

    let CliMode::Native(mode, mode_name) = mode else {
        unreachable!();
    };

    let models_dir = resolve_models_dir(mode);

    let step = match mode_name {
        "coreml-fast" => FAST_SEGMENTATION_STEP_SECONDS,
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

    for wav_path in &wav_paths {
        let file_id = Path::new(wav_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file1".to_string());

        let (samples, sr) = load_wav_samples(wav_path);
        assert_eq!(sr, 16000, "expected 16kHz WAV, got {sr}Hz");

        let start = Instant::now();
        let result = diarize(&mut seg_model, &mut emb_model, &plda, &samples, &file_id)
            .expect("diarization failed");
        let elapsed = start.elapsed();

        if wav_paths.len() > 1 {
            eprintln!("{file_id}: {:.3}s", elapsed.as_secs_f64());
        }
        print!("{}", result.rttm);
    }
}

fn resolve_models_dir(mode: ExecutionMode) -> PathBuf {
    if let Ok(dir) = std::env::var("SPEAKRS_MODELS_DIR") {
        return PathBuf::from(dir);
    }

    #[cfg(feature = "online")]
    {
        let manager = ModelManager::new().expect("failed to initialize model manager");
        let hf_mode = match mode {
            ExecutionMode::Cpu => speakrs::models::Mode::Cpu,
            ExecutionMode::CoreMl => speakrs::models::Mode::CoreMl,
            ExecutionMode::CoreMlFast => speakrs::models::Mode::CoreMlFast,
            ExecutionMode::Cuda => speakrs::models::Mode::Cuda,
        };
        manager.ensure(hf_mode).expect("failed to download models")
    }

    #[cfg(not(feature = "online"))]
    {
        let _ = mode;
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/models")
            .to_path_buf()
    }
}

enum CliMode {
    Native(ExecutionMode, &'static str),
    PyannoteDevice(&'static str),
}

fn parse_args(args: &[String]) -> (CliMode, Vec<&str>) {
    const USAGE: &str = "Usage: diarize [--mode cpu|coreml|coreml-fast|cuda|pyannote-cpu|pyannote-mps|pyannote-cuda] <wav_files...>";

    fn parse_mode(mode: &str) -> Option<CliMode> {
        match mode {
            "cpu" => Some(CliMode::Native(ExecutionMode::Cpu, "cpu")),
            "coreml" => Some(CliMode::Native(ExecutionMode::CoreMl, "coreml")),
            "coreml-fast" => Some(CliMode::Native(ExecutionMode::CoreMlFast, "coreml-fast")),
            "cuda" => Some(CliMode::Native(ExecutionMode::Cuda, "cuda")),
            "pyannote-cpu" => Some(CliMode::PyannoteDevice("cpu")),
            "pyannote-mps" => Some(CliMode::PyannoteDevice("mps")),
            "pyannote-cuda" => Some(CliMode::PyannoteDevice("cuda")),
            _ => None,
        }
    }

    if args.len() < 2 {
        eprintln!("{USAGE}");
        return (CliMode::Native(ExecutionMode::Cpu, "cpu"), vec![]);
    }

    // diarize --mode <mode> file1.wav file2.wav ...
    if args.len() >= 4 && args[1] == "--mode" {
        let Some(parsed) = parse_mode(&args[2]) else {
            eprintln!("Unknown mode: {}", args[2]);
            eprintln!("{USAGE}");
            return (CliMode::Native(ExecutionMode::Cpu, "cpu"), vec![]);
        };
        let paths: Vec<&str> = args[3..].iter().map(|s| s.as_str()).collect();
        return (parsed, paths);
    }

    // diarize file1.wav file2.wav ...
    let paths: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
    (CliMode::Native(ExecutionMode::Cpu, "cpu"), paths)
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
