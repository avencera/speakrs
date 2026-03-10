use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;

use ndarray::{Array2, Array3, ArrayView2, s};
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::SEGMENTATION_STEP_SECONDS;
use speakrs::powerset::PowersetMapping;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        eprintln!(
            "Usage: profile_stages <seg-only|embed-stream|embed-store|embed-repeat> <path/to/audio.wav> [iterations_or_log_every] [log_every]"
        );
        std::process::exit(1);
    }

    let mode = &args[1];
    let wav_path = &args[2];
    let log_every = if mode == "embed-repeat" {
        args.get(4)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(200)
    } else {
        args.get(3)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(100)
    };
    let iterations = if mode == "embed-repeat" {
        args.get(3)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5000)
    } else {
        0
    };

    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/models");
    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .expect("failed to load segmentation model");
    let mut emb_model = EmbeddingModel::new(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
    )
    .expect("failed to load embedding model");
    let powerset = PowersetMapping::new(3, 2);

    let (samples, sample_rate) = load_wav_samples(wav_path);
    assert_eq!(
        sample_rate, 16_000,
        "expected 16kHz WAV, got {sample_rate}Hz"
    );

    eprintln!("start rss_mb={:.1}", rss_mb());
    let raw_windows = seg_model.run(&samples).expect("segmentation failed");
    eprintln!(
        "after_segmentation num_windows={} rss_mb={:.1}",
        raw_windows.len(),
        rss_mb()
    );

    if mode == "seg-only" {
        return;
    }

    let segmentations = decode_windows(raw_windows, &powerset);
    eprintln!(
        "after_decode shape=({}, {}, {}) rss_mb={:.1}",
        segmentations.shape()[0],
        segmentations.shape()[1],
        segmentations.shape()[2],
        rss_mb()
    );

    match mode.as_str() {
        "embed-stream" => run_embedding_stream(
            &seg_model,
            &mut emb_model,
            &samples,
            &segmentations,
            log_every,
        ),
        "embed-store" => run_embedding_store(
            &seg_model,
            &mut emb_model,
            &samples,
            &segmentations,
            log_every,
        ),
        "embed-repeat" => run_embedding_repeat(
            &seg_model,
            &mut emb_model,
            &samples,
            &segmentations,
            iterations,
            log_every,
        ),
        _ => {
            eprintln!("Unknown mode: {mode}");
            std::process::exit(1);
        }
    }
}

fn run_embedding_stream(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
    log_every: usize,
) {
    let num_chunks = segmentations.shape()[0];
    let num_speakers = segmentations.shape()[2];
    for chunk_idx in 0..num_chunks {
        let chunk_audio = chunk_audio(audio, seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean_masks = clean_masks(&chunk_segmentations);

        for speaker_idx in 0..num_speakers {
            let mask = chunk_segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let _ = emb_model
                .embed_masked(
                    chunk_audio,
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                )
                .expect("embedding failed");
        }

        if (chunk_idx + 1) % log_every == 0 || chunk_idx + 1 == num_chunks {
            eprintln!(
                "embed_stream chunk={}/{} rss_mb={:.1}",
                chunk_idx + 1,
                num_chunks,
                rss_mb()
            );
        }
    }
}

fn run_embedding_store(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
    log_every: usize,
) {
    let num_chunks = segmentations.shape()[0];
    let num_speakers = segmentations.shape()[2];
    let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

    for chunk_idx in 0..num_chunks {
        let chunk_audio = chunk_audio(audio, seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean_masks = clean_masks(&chunk_segmentations);

        for speaker_idx in 0..num_speakers {
            let mask = chunk_segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let embedding = emb_model
                .embed_masked(
                    chunk_audio,
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                )
                .expect("embedding failed");
            embeddings
                .slice_mut(s![chunk_idx, speaker_idx, ..])
                .assign(&embedding);
        }

        if (chunk_idx + 1) % log_every == 0 || chunk_idx + 1 == num_chunks {
            eprintln!(
                "embed_store chunk={}/{} rss_mb={:.1}",
                chunk_idx + 1,
                num_chunks,
                rss_mb()
            );
        }
    }

    eprintln!(
        "after_embed_store shape=({}, {}, {}) rss_mb={:.1}",
        embeddings.shape()[0],
        embeddings.shape()[1],
        embeddings.shape()[2],
        rss_mb()
    );
    let finite_count = embeddings.iter().filter(|value| value.is_finite()).count();
    let total_count = embeddings.len();
    eprintln!("embed_store finite={finite_count}/{total_count}");
}

fn run_embedding_repeat(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
    iterations: usize,
    log_every: usize,
) {
    let chunk_audio = chunk_audio(audio, seg_model, 0);
    let chunk_segmentations = segmentations.slice(s![0, .., ..]);
    let clean_masks = clean_masks(&chunk_segmentations);
    let local_idx = (0..chunk_segmentations.ncols())
        .find(|&speaker_idx| chunk_segmentations.column(speaker_idx).sum() > 0.0)
        .unwrap_or(0);
    let mask = chunk_segmentations.column(local_idx).to_owned();
    let clean_mask = clean_masks.column(local_idx).to_owned();

    for iteration in 0..iterations {
        let _ = emb_model
            .embed_masked(
                chunk_audio,
                mask.as_slice().unwrap(),
                Some(clean_mask.as_slice().unwrap()),
            )
            .expect("embedding failed");

        if (iteration + 1) % log_every == 0 || iteration + 1 == iterations {
            eprintln!(
                "embed_repeat iter={}/{} rss_mb={:.1}",
                iteration + 1,
                iterations,
                rss_mb()
            );
        }
    }
}

fn decode_windows(raw_windows: Vec<Array2<f32>>, powerset: &PowersetMapping) -> Array3<f32> {
    let num_windows = raw_windows.len();
    let mut windows = raw_windows.into_iter();
    let first = powerset.hard_decode(&windows.next().unwrap());
    let mut stacked = Array3::<f32>::zeros((num_windows, first.nrows(), first.ncols()));
    stacked.slice_mut(s![0, .., ..]).assign(&first);

    for (window_idx, window) in windows.enumerate() {
        let decoded = powerset.hard_decode(&window);
        stacked
            .slice_mut(s![window_idx + 1, .., ..])
            .assign(&decoded);
    }

    stacked
}

fn clean_masks(segmentations: &ArrayView2<f32>) -> Array2<f32> {
    let single_active: Vec<bool> = segmentations
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().sum::<f32>() < 2.0)
        .collect();
    let mut clean = Array2::<f32>::zeros(segmentations.raw_dim());
    for (frame_idx, is_single_active) in single_active.iter().enumerate() {
        if !*is_single_active {
            continue;
        }

        clean
            .slice_mut(s![frame_idx, ..])
            .assign(&segmentations.slice(s![frame_idx, ..]));
    }
    clean
}

fn chunk_audio<'a>(audio: &'a [f32], seg_model: &SegmentationModel, chunk_idx: usize) -> &'a [f32] {
    let step_samples = (SEGMENTATION_STEP_SECONDS * seg_model.sample_rate() as f64) as usize;
    let start = chunk_idx * step_samples;
    let end = (start + seg_model.window_samples()).min(audio.len());
    if start < audio.len() {
        &audio[start..end]
    } else {
        &[]
    }
}

fn rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .expect("failed to read rss from ps");
    let rss_kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0);
    rss_kb / 1024.0
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
