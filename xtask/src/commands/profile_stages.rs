use color_eyre::eyre::{Result, bail, ensure};
use ndarray::{Array3, s};
use speakrs::PowersetMapping;
use speakrs::inference::{EmbeddingModel, SegmentationModel};
use speakrs::pipeline::SEGMENTATION_STEP_SECONDS;

use crate::commands::profile_support;
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

    let models_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fixtures/models");
    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx"),
        SEGMENTATION_STEP_SECONDS as f32,
    )?;
    let mut emb_model = EmbeddingModel::new(models_dir.join("wespeaker-voxceleb-resnet34.onnx"))?;
    let powerset = PowersetMapping::new(3, 2);

    let (samples, sample_rate) = wav::load_wav_samples(wav_path)?;
    ensure!(
        sample_rate == 16_000,
        "expected 16kHz WAV, got {sample_rate}Hz"
    );

    eprintln!("start rss_mb={:.1}", profile_support::rss_mb()?);
    let raw_windows = seg_model.run(&samples)?;
    eprintln!(
        "after_segmentation num_windows={} rss_mb={:.1}",
        raw_windows.len(),
        profile_support::rss_mb()?
    );

    if mode == "seg-only" {
        return Ok(());
    }

    let segmentations = profile_support::decode_windows(raw_windows, &powerset);
    eprintln!(
        "after_decode shape=({}, {}, {}) rss_mb={:.1}",
        segmentations.shape()[0],
        segmentations.shape()[1],
        segmentations.shape()[2],
        profile_support::rss_mb()?
    );

    match mode {
        "embed-stream" => run_embedding_stream(
            &seg_model,
            &mut emb_model,
            &samples,
            &segmentations,
            log_every,
        )?,
        "embed-store" => run_embedding_store(
            &seg_model,
            &mut emb_model,
            &samples,
            &segmentations,
            log_every,
        )?,
        "embed-repeat" => run_embedding_repeat(
            &seg_model,
            &mut emb_model,
            &samples,
            &segmentations,
            iterations,
            log_every,
        )?,
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
) -> Result<()> {
    let num_chunks = segmentations.shape()[0];
    let num_speakers = segmentations.shape()[2];
    for chunk_idx in 0..num_chunks {
        let chunk_audio = profile_support::chunk_audio(audio, seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean_masks = profile_support::clean_masks(&chunk_segmentations);

        for speaker_idx in 0..num_speakers {
            let mask = chunk_segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let _ = emb_model
                .embed_masked(
                    chunk_audio,
                    profile_support::array1_slice(&mask, "profile stages stream mask")?,
                    Some(profile_support::array1_slice(
                        &clean_mask,
                        "profile stages stream clean mask",
                    )?),
                )
                .map_err(|error| color_eyre::eyre::eyre!("embedding failed: {error}"))?;
        }

        if (chunk_idx + 1) % log_every == 0 || chunk_idx + 1 == num_chunks {
            eprintln!(
                "embed_stream chunk={}/{} rss_mb={:.1}",
                chunk_idx + 1,
                num_chunks,
                profile_support::rss_mb()?
            );
        }
    }

    Ok(())
}

fn run_embedding_store(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
    log_every: usize,
) -> Result<()> {
    let num_chunks = segmentations.shape()[0];
    let num_speakers = segmentations.shape()[2];
    let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

    for chunk_idx in 0..num_chunks {
        let chunk_audio = profile_support::chunk_audio(audio, seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean_masks = profile_support::clean_masks(&chunk_segmentations);

        for speaker_idx in 0..num_speakers {
            let mask = chunk_segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let embedding = emb_model
                .embed_masked(
                    chunk_audio,
                    profile_support::array1_slice(&mask, "profile stages store mask")?,
                    Some(profile_support::array1_slice(
                        &clean_mask,
                        "profile stages store clean mask",
                    )?),
                )
                .map_err(|error| color_eyre::eyre::eyre!("embedding failed: {error}"))?;
            embeddings
                .slice_mut(s![chunk_idx, speaker_idx, ..])
                .assign(&embedding);
        }

        if (chunk_idx + 1) % log_every == 0 || chunk_idx + 1 == num_chunks {
            eprintln!(
                "embed_store chunk={}/{} rss_mb={:.1}",
                chunk_idx + 1,
                num_chunks,
                profile_support::rss_mb()?
            );
        }
    }

    eprintln!(
        "after_embed_store shape=({}, {}, {}) rss_mb={:.1}",
        embeddings.shape()[0],
        embeddings.shape()[1],
        embeddings.shape()[2],
        profile_support::rss_mb()?
    );
    let finite_count = embeddings.iter().filter(|value| value.is_finite()).count();
    let total_count = embeddings.len();
    eprintln!("embed_store finite={finite_count}/{total_count}");
    Ok(())
}

fn run_embedding_repeat(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
    iterations: usize,
    log_every: usize,
) -> Result<()> {
    let chunk_audio = profile_support::chunk_audio(audio, seg_model, 0);
    let chunk_segmentations = segmentations.slice(s![0, .., ..]);
    let clean_masks = profile_support::clean_masks(&chunk_segmentations);
    let local_idx = (0..chunk_segmentations.ncols())
        .find(|&speaker_idx| chunk_segmentations.column(speaker_idx).sum() > 0.0)
        .unwrap_or(0);
    let mask = chunk_segmentations.column(local_idx).to_owned();
    let clean_mask = clean_masks.column(local_idx).to_owned();

    for iteration in 0..iterations {
        let _ = emb_model
            .embed_masked(
                chunk_audio,
                profile_support::array1_slice(&mask, "profile stages repeat mask")?,
                Some(profile_support::array1_slice(
                    &clean_mask,
                    "profile stages repeat clean mask",
                )?),
            )
            .map_err(|error| color_eyre::eyre::eyre!("embedding failed: {error}"))?;

        if (iteration + 1) % log_every == 0 || iteration + 1 == iterations {
            eprintln!(
                "embed_repeat iter={}/{} rss_mb={:.1}",
                iteration + 1,
                iterations,
                profile_support::rss_mb()?
            );
        }
    }
    Ok(())
}
