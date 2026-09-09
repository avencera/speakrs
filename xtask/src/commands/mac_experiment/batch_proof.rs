use std::{path::Path, time::Instant};

use chrono::{SecondsFormat, Utc};
use color_eyre::eyre::{Context, ContextCompat, Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use speakrs::{BatchInput, DiarizationResult, ExecutionMode, OwnedDiarizationPipeline};

use super::{
    identity::{HostIdentity, digest_paths, digest_paths_cached},
    statistics::{median, median_absolute_deviation, speed_improvement_percent},
    store::atomic_write,
};
use crate::cmd::project_root;
use crate::wav::load_wav_samples;

const SAMPLE_RATE: u32 = 16_000;
const MAX_FILES: usize = 32;
const MAX_REPETITIONS: usize = 100;

#[derive(Serialize)]
struct BatchProofReport {
    schema_version: u32,
    generated_at: String,
    host: HostIdentity,
    model_sha256: String,
    files: Vec<InputIdentity>,
    duration_seconds: f64,
    repetitions_per_condition: usize,
    order: Vec<Condition>,
    runs: Vec<RunResult>,
    summary: BatchSummary,
}

#[derive(Serialize)]
struct InputIdentity {
    id: String,
    path: String,
    sha256: String,
    sample_count: usize,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Condition {
    RepeatedSingles,
    Batch,
}

#[derive(Serialize)]
struct RunResult {
    index: usize,
    condition: Condition,
    total_seconds: f64,
    single_file_seconds: Vec<f64>,
    output_sha256: Vec<String>,
    matches_reference: bool,
}

#[derive(Serialize)]
struct BatchSummary {
    repeated_singles_seconds: Distribution,
    batch_seconds: Distribution,
    batch_speed_improvement_percent: f64,
    batch_completion_latency_seconds: f64,
    repeated_single_file_latency_seconds: f64,
    single_file_latency_penalty_percent: f64,
    output_parity_passed: bool,
}

#[derive(Serialize)]
struct Distribution {
    median: f64,
    median_absolute_deviation: f64,
    minimum: f64,
    maximum: f64,
}

struct InputAudio {
    id: String,
    path: std::path::PathBuf,
    samples: Vec<f32>,
}

pub(super) fn run(
    models_dir: &Path,
    audio_paths: &[std::path::PathBuf],
    duration_seconds: f64,
    repetitions: usize,
    output: &Path,
) -> Result<()> {
    ensure!(
        (2..=MAX_FILES).contains(&audio_paths.len()),
        "batch proof requires between 2 and {MAX_FILES} audio files"
    );
    ensure!(
        duration_seconds.is_finite() && duration_seconds >= 10.0,
        "duration seconds must be finite and at least 10"
    );
    ensure!(
        (6..=MAX_REPETITIONS).contains(&repetitions) && repetitions.is_multiple_of(2),
        "repetitions must be even and between 6 and {MAX_REPETITIONS}"
    );

    let root = project_root();
    let models_dir = models_dir
        .canonicalize()
        .wrap_err_with(|| format!("failed to resolve {}", models_dir.display()))?;
    let sample_count = seconds_to_samples(duration_seconds)?;
    let inputs = load_inputs(audio_paths, sample_count)?;
    let identities = inputs
        .iter()
        .map(|input| {
            Ok(InputIdentity {
                id: input.id.clone(),
                path: input.path.display().to_string(),
                sha256: digest_paths(&root, std::slice::from_ref(&input.path))?,
                sample_count: input.samples.len(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let model_sha256 = digest_paths_cached(
        &root,
        std::slice::from_ref(&models_dir),
        &root.join("_benchmarks/macos/digest-cache"),
    )?;

    let mut singles_pipeline =
        OwnedDiarizationPipeline::from_dir(&models_dir, ExecutionMode::CoreMl)?;
    let mut batch_pipeline =
        OwnedDiarizationPipeline::from_dir(&models_dir, ExecutionMode::CoreMl)?;
    let reference = run_singles(&mut singles_pipeline, &inputs)?.fingerprints;
    let warm_batch = run_batch(&mut batch_pipeline, &inputs)?;
    ensure!(
        warm_batch.fingerprints == reference,
        "batch warmup output does not match repeated singles"
    );

    let order = abba_order(repetitions);
    let mut runs = Vec::with_capacity(order.len());
    for (index, condition) in order.iter().copied().enumerate() {
        let measured = match condition {
            Condition::RepeatedSingles => run_singles(&mut singles_pipeline, &inputs)?,
            Condition::Batch => run_batch(&mut batch_pipeline, &inputs)?,
        };
        let matches_reference = measured.fingerprints == reference;
        runs.push(RunResult {
            index,
            condition,
            total_seconds: measured.total_seconds,
            single_file_seconds: measured.single_file_seconds,
            output_sha256: measured.fingerprints,
            matches_reference,
        });
    }

    let summary = summarize(&runs)?;
    let report = BatchProofReport {
        schema_version: 1,
        generated_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        host: HostIdentity::collect(&root)?,
        model_sha256,
        files: identities,
        duration_seconds,
        repetitions_per_condition: repetitions,
        order,
        runs,
        summary,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    atomic_write(output, &bytes)?;

    println!(
        "Batch proof: speed={:.2}%, latency penalty={:.2}%, parity={}, output={}",
        report.summary.batch_speed_improvement_percent,
        report.summary.single_file_latency_penalty_percent,
        report.summary.output_parity_passed,
        output.display()
    );
    Ok(())
}

struct MeasuredRun {
    total_seconds: f64,
    single_file_seconds: Vec<f64>,
    fingerprints: Vec<String>,
}

fn run_singles(
    pipeline: &mut OwnedDiarizationPipeline,
    inputs: &[InputAudio],
) -> Result<MeasuredRun> {
    let total_start = Instant::now();
    let mut single_file_seconds = Vec::with_capacity(inputs.len());
    let mut fingerprints = Vec::with_capacity(inputs.len());
    for input in inputs {
        let start = Instant::now();
        let result = pipeline.run_with_file_id(&input.samples, &input.id)?;
        single_file_seconds.push(start.elapsed().as_secs_f64());
        fingerprints.push(result_fingerprint(&result));
    }

    Ok(MeasuredRun {
        total_seconds: total_start.elapsed().as_secs_f64(),
        single_file_seconds,
        fingerprints,
    })
}

fn run_batch(
    pipeline: &mut OwnedDiarizationPipeline,
    inputs: &[InputAudio],
) -> Result<MeasuredRun> {
    let batch_inputs: Vec<_> = inputs
        .iter()
        .map(|input| BatchInput {
            audio: &input.samples,
            file_id: &input.id,
        })
        .collect();
    let start = Instant::now();
    let results = pipeline.run_batch(&batch_inputs)?;
    let total_seconds = start.elapsed().as_secs_f64();
    ensure!(
        results.len() == inputs.len(),
        "batch returned {} results for {} inputs",
        results.len(),
        inputs.len()
    );

    Ok(MeasuredRun {
        total_seconds,
        // The batch API returns all results together, so each file has the batch completion latency
        single_file_seconds: vec![total_seconds; inputs.len()],
        fingerprints: results.iter().map(result_fingerprint).collect(),
    })
}

fn summarize(runs: &[RunResult]) -> Result<BatchSummary> {
    let singles: Vec<_> = runs
        .iter()
        .filter(|run| matches!(run.condition, Condition::RepeatedSingles))
        .map(|run| run.total_seconds)
        .collect();
    let batches: Vec<_> = runs
        .iter()
        .filter(|run| matches!(run.condition, Condition::Batch))
        .map(|run| run.total_seconds)
        .collect();
    let single_file_latencies: Vec<_> = runs
        .iter()
        .filter(|run| matches!(run.condition, Condition::RepeatedSingles))
        .flat_map(|run| run.single_file_seconds.iter().copied())
        .collect();
    let singles_distribution = distribution(&singles)?;
    let batch_distribution = distribution(&batches)?;
    let repeated_single_file_latency_seconds = median(&single_file_latencies)?;
    let batch_completion_latency_seconds = batch_distribution.median;
    let single_file_latency_penalty_percent =
        (batch_completion_latency_seconds / repeated_single_file_latency_seconds - 1.0) * 100.0;

    Ok(BatchSummary {
        batch_speed_improvement_percent: speed_improvement_percent(
            singles_distribution.median,
            batch_distribution.median,
        )?,
        repeated_singles_seconds: singles_distribution,
        batch_seconds: batch_distribution,
        batch_completion_latency_seconds,
        repeated_single_file_latency_seconds,
        single_file_latency_penalty_percent,
        output_parity_passed: runs.iter().all(|run| run.matches_reference),
    })
}

fn distribution(values: &[f64]) -> Result<Distribution> {
    Ok(Distribution {
        median: median(values)?,
        median_absolute_deviation: median_absolute_deviation(values)?,
        minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
        maximum: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

fn abba_order(repetitions: usize) -> Vec<Condition> {
    (0..repetitions / 2)
        .flat_map(|_| {
            [
                Condition::RepeatedSingles,
                Condition::Batch,
                Condition::Batch,
                Condition::RepeatedSingles,
            ]
        })
        .collect()
}

fn load_inputs(paths: &[std::path::PathBuf], sample_count: usize) -> Result<Vec<InputAudio>> {
    paths
        .iter()
        .map(|path| {
            let path = path
                .canonicalize()
                .wrap_err_with(|| format!("failed to resolve {}", path.display()))?;
            let (mut samples, sample_rate) = load_wav_samples(&path.to_string_lossy())?;
            ensure!(
                sample_rate == SAMPLE_RATE,
                "audio sample rate must be {SAMPLE_RATE} Hz, got {sample_rate} Hz"
            );
            ensure!(
                samples.len() >= sample_count,
                "audio {} has {} samples but {sample_count} are required",
                path.display(),
                samples.len()
            );
            samples.truncate(sample_count);
            let id = path
                .file_stem()
                .context("audio path has no file stem")?
                .to_string_lossy()
                .into_owned();

            Ok(InputAudio { id, path, samples })
        })
        .collect()
}

fn seconds_to_samples(seconds: f64) -> Result<usize> {
    let exact = seconds * f64::from(SAMPLE_RATE);
    let samples = exact.round() as usize;
    ensure!(
        (exact - samples as f64).abs() <= 1e-6,
        "duration seconds must identify an exact 16 kHz sample"
    );

    Ok(samples)
}

fn result_fingerprint(result: &DiarizationResult) -> String {
    let mut digest = Sha256::new();
    for dimension in result.discrete_diarization.shape() {
        digest.update(dimension.to_le_bytes());
    }
    for value in result.discrete_diarization.iter() {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abba_order_has_equal_condition_counts() {
        let order = abba_order(6);

        assert_eq!(order.len(), 12);
        assert!(matches!(order[0], Condition::RepeatedSingles));
        assert!(matches!(order[1], Condition::Batch));
        assert!(matches!(order[2], Condition::Batch));
        assert!(matches!(order[3], Condition::RepeatedSingles));
    }

    #[test]
    fn seconds_to_samples_rejects_subsample_duration() {
        assert!(seconds_to_samples(10.000_01).is_err());
        assert_eq!(seconds_to_samples(10.0).unwrap(), 160_000);
    }
}
