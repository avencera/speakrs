use std::path::Path;
use std::process::Command;

use color_eyre::eyre::{Result, bail, ensure};
use ndarray::{Array2, Array3, ArrayView2, s};
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::SEGMENTATION_STEP_SECONDS;
use speakrs::powerset::PowersetMapping;

use crate::wav;

pub fn run(mode: &str, wav_path: &str, iterations: usize, log_every: usize) -> Result<()> {
    let log_every = if log_every == 0 {
        if mode == "embed-repeat" { 200 } else { 100 }
    } else {
        log_every
    };
    let iterations = if mode == "embed-repeat" && iterations == 0 {
        5000
    } else {
        iterations
    };

    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/models");
    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )?;
    let mut emb_model = EmbeddingModel::new(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
    )?;
    let powerset = PowersetMapping::new(3, 2);

    let (samples, sample_rate) = wav::load_wav_samples(wav_path)?;
    ensure!(
        sample_rate == 16_000,
        "expected 16kHz WAV, got {sample_rate}Hz"
    );

    eprintln!("start rss_mb={:.1}", rss_mb());
    let raw_windows = seg_model.run(&samples)?;
    eprintln!(
        "after_segmentation num_windows={} rss_mb={:.1}",
        raw_windows.len(),
        rss_mb()
    );

    if mode == "seg-only" {
        return Ok(());
    }

    let segmentations = decode_windows(raw_windows, &powerset);
    eprintln!(
        "after_decode shape=({}, {}, {}) rss_mb={:.1}",
        segmentations.shape()[0],
        segmentations.shape()[1],
        segmentations.shape()[2],
        rss_mb()
    );

    match mode {
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
        _ => bail!("unknown mode: {mode}"),
    }

    Ok(())
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
