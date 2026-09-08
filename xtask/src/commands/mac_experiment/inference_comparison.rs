use std::{collections::BTreeSet, path::Path, time::Instant};

use chrono::{SecondsFormat, Utc};
use color_eyre::eyre::{Context, ContextCompat, Result, ensure};
use serde::Serialize;
use speakrs::{
    ExecutionMode, PipelineBuilder, RuntimeConfig,
    pipeline::{
        CoreMlChunkLayout, CoreMlFbankNormalizationScope, DiarizationResult,
        ExperimentInferenceConfig, InferenceArtifacts,
    },
};

use super::{
    InferenceComparisonStep,
    identity::{HostIdentity, digest_paths, digest_paths_cached},
    store::atomic_write,
};
use crate::wav::load_wav_samples;

const SAMPLE_RATE: u32 = 16_000;

#[derive(Serialize)]
struct ComparisonReport {
    schema_version: u32,
    generated_at: String,
    input: InputIdentity,
    host: HostIdentity,
    chunk_layout: &'static str,
    per_window_layout: &'static str,
    chunk_inference_seconds: f64,
    per_window_inference_seconds: f64,
    chunk_count: usize,
    per_window_count: usize,
    matching_embedding_pairs: usize,
    availability_mismatches: usize,
    cosine_similarity: CosineDistribution,
    predicted_speakers: PredictedSpeakerStability,
}

#[derive(Serialize)]
struct InputIdentity {
    audio_path: String,
    audio_sha256: String,
    sample_rate: u32,
    sample_count: usize,
    models_sha256: String,
}

#[derive(Serialize)]
struct CosineDistribution {
    minimum: f64,
    p05: f64,
    median: f64,
    p95: f64,
    maximum: f64,
    mean: f64,
}

#[derive(Serialize)]
struct PredictedSpeakerStability {
    chunk_layout_count: usize,
    per_window_layout_count: usize,
    count_matches: bool,
}

struct LayoutRun {
    artifacts: InferenceArtifacts,
    result: DiarizationResult,
    inference_seconds: f64,
}

pub(super) fn run(
    models_dir: &Path,
    audio: &Path,
    step: InferenceComparisonStep,
    output: &Path,
) -> Result<()> {
    let root = std::env::current_dir().context("failed to find the repository directory")?;
    let models_dir = models_dir
        .canonicalize()
        .wrap_err_with(|| format!("failed to resolve {}", models_dir.display()))?;
    let audio = audio
        .canonicalize()
        .wrap_err_with(|| format!("failed to resolve {}", audio.display()))?;
    let (samples, sample_rate) = load_wav_samples(&audio.to_string_lossy())?;
    ensure!(
        sample_rate == SAMPLE_RATE,
        "audio sample rate must be {SAMPLE_RATE} Hz, got {sample_rate} Hz"
    );
    let (
        chunk_layout,
        chunk_normalization,
        per_window_layout,
        per_window_normalization,
        chunk_name,
        per_window_name,
    ) = layouts(step);

    let chunk = run_layout(&models_dir, &samples, chunk_layout, chunk_normalization)?;
    let per_window = run_layout(
        &models_dir,
        &samples,
        per_window_layout,
        per_window_normalization,
    )?;
    let (cosines, availability_mismatches) =
        matching_cosines(&chunk.artifacts, &per_window.artifacts)?;
    let chunk_speakers = predicted_speaker_count(&chunk.result);
    let per_window_speakers = predicted_speaker_count(&per_window.result);
    let report = ComparisonReport {
        schema_version: 1,
        generated_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        input: InputIdentity {
            audio_path: audio.display().to_string(),
            audio_sha256: digest_paths(&root, std::slice::from_ref(&audio))?,
            sample_rate,
            sample_count: samples.len(),
            models_sha256: digest_paths_cached(
                &root,
                std::slice::from_ref(&models_dir),
                &root.join("_benchmarks/macos/digest-cache"),
            )?,
        },
        host: HostIdentity::collect(&root)?,
        chunk_layout: chunk_name,
        per_window_layout: per_window_name,
        chunk_inference_seconds: chunk.inference_seconds,
        per_window_inference_seconds: per_window.inference_seconds,
        chunk_count: chunk.artifacts.embeddings().shape()[0],
        per_window_count: per_window.artifacts.embeddings().shape()[0],
        matching_embedding_pairs: cosines.len(),
        availability_mismatches,
        cosine_similarity: cosine_distribution(&cosines)?,
        predicted_speakers: PredictedSpeakerStability {
            chunk_layout_count: chunk_speakers,
            per_window_layout_count: per_window_speakers,
            count_matches: chunk_speakers == per_window_speakers,
        },
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    atomic_write(output, &bytes)?;

    println!(
        "Inference comparison: pairs={}, median cosine={:.8}, speaker count match={}, output={}",
        report.matching_embedding_pairs,
        report.cosine_similarity.median,
        report.predicted_speakers.count_matches,
        output.display()
    );
    Ok(())
}

fn layouts(
    step: InferenceComparisonStep,
) -> (
    CoreMlChunkLayout,
    CoreMlFbankNormalizationScope,
    CoreMlChunkLayout,
    CoreMlFbankNormalizationScope,
    &'static str,
    &'static str,
) {
    match step {
        InferenceComparisonStep::OneSecond => (
            CoreMlChunkLayout::OneSecondPhased,
            CoreMlFbankNormalizationScope::Chunk,
            CoreMlChunkLayout::PerWindow,
            CoreMlFbankNormalizationScope::Chunk,
            "one_second_phased",
            "per_window_1s",
        ),
        InferenceComparisonStep::NormalizationScope => (
            CoreMlChunkLayout::OneSecondPhased,
            CoreMlFbankNormalizationScope::Chunk,
            CoreMlChunkLayout::OneSecondPhased,
            CoreMlFbankNormalizationScope::TenSecondSegments,
            "chunk_normalized",
            "ten_second_segment_normalized",
        ),
    }
}

fn run_layout(
    models_dir: &Path,
    samples: &[f32],
    layout: CoreMlChunkLayout,
    normalization_scope: CoreMlFbankNormalizationScope,
) -> Result<LayoutRun> {
    let experiment =
        ExperimentInferenceConfig::new(layout).with_fbank_normalization_scope(normalization_scope);
    let runtime = RuntimeConfig::default().with_experiment(experiment);
    let mut pipeline = PipelineBuilder::from_dir(models_dir, ExecutionMode::CoreMl)
        .runtime(runtime)
        .build()?;
    let start = Instant::now();
    let artifacts = pipeline.run_inference_only(samples)?;
    let inference_seconds = start.elapsed().as_secs_f64();
    let result = pipeline.finish_post_inference(artifacts.clone(), &pipeline.pipeline_config())?;

    Ok(LayoutRun {
        artifacts,
        result,
        inference_seconds,
    })
}

fn matching_cosines(
    chunk: &InferenceArtifacts,
    per_window: &InferenceArtifacts,
) -> Result<(Vec<f64>, usize)> {
    let chunk = chunk.embeddings();
    let per_window = per_window.embeddings();
    ensure!(
        chunk.shape() == per_window.shape(),
        "embedding shapes differ: {:?} and {:?}",
        chunk.shape(),
        per_window.shape()
    );

    let mut cosines = Vec::new();
    let mut availability_mismatches = 0;
    for window_index in 0..chunk.shape()[0] {
        for speaker_index in 0..chunk.shape()[1] {
            let chunk_embedding = chunk.slice(ndarray::s![window_index, speaker_index, ..]);
            let per_window_embedding =
                per_window.slice(ndarray::s![window_index, speaker_index, ..]);
            let chunk_available = chunk_embedding.iter().all(|value| value.is_finite());
            let per_window_available = per_window_embedding.iter().all(|value| value.is_finite());
            match (chunk_available, per_window_available) {
                (true, true) => cosines.push(cosine_similarity(
                    chunk_embedding
                        .as_slice()
                        .context("chunk embedding is not contiguous")?,
                    per_window_embedding
                        .as_slice()
                        .context("per-window embedding is not contiguous")?,
                )?),
                (false, false) => {}
                _ => availability_mismatches += 1,
            }
        }
    }
    ensure!(
        !cosines.is_empty(),
        "no matching finite embeddings were found"
    );

    Ok((cosines, availability_mismatches))
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f64> {
    ensure!(left.len() == right.len(), "embedding sizes differ");
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    ensure!(
        left_norm > 0.0 && right_norm > 0.0,
        "cosine similarity requires non-zero embeddings"
    );

    Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

fn cosine_distribution(values: &[f64]) -> Result<CosineDistribution> {
    ensure!(!values.is_empty(), "cosine distribution is empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "cosine distribution contains a non-finite value"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    Ok(CosineDistribution {
        minimum: sorted[0],
        p05: percentile(&sorted, 0.05),
        median: percentile(&sorted, 0.5),
        p95: percentile(&sorted, 0.95),
        maximum: *sorted.last().expect("non-empty values"),
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn predicted_speaker_count(result: &DiarizationResult) -> usize {
    result
        .hard_clusters
        .iter()
        .copied()
        .filter(|cluster| *cluster >= 0)
        .collect::<BTreeSet<_>>()
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_distribution_is_order_independent() {
        let distribution = cosine_distribution(&[0.8, 1.0, 0.9]).unwrap();

        assert_eq!(distribution.minimum, 0.8);
        assert_eq!(distribution.median, 0.9);
        assert_eq!(distribution.maximum, 1.0);
    }

    #[test]
    fn cosine_similarity_accepts_scaled_vectors() {
        let similarity = cosine_similarity(&[1.0, -2.0], &[2.0, -4.0]).unwrap();

        assert!((similarity - 1.0).abs() < 1e-12);
    }
}
