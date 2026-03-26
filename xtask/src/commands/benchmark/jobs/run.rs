#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::time::Duration;

#[cfg(feature = "cuda")]
use color_eyre::eyre::ensure;
use color_eyre::eyre::{Result, bail};

#[cfg(feature = "cuda")]
use super::super::{
    BatchCommandRunner, BenchmarkMetadata, DerAccumulation, DerImplResult, DerResultsWriter,
    ImplType, discover_files, format_eta, now_stamp,
};
use super::gpu::resolve_gpu_impls;
use super::preflight::preflight;
use super::{BenchmarkJobConfig, BenchmarkJobResult, GpuBenchmarkSuiteConfig, ProgressUpdate};
use crate::commands::benchmark::runner::BatchRunOutput;
#[cfg(feature = "cuda")]
use crate::path::file_stem_string;

pub fn run_gpu_benchmark_suite(config: &GpuBenchmarkSuiteConfig) -> Result<()> {
    let datasets_list: Vec<crate::datasets::Dataset> = if config.dataset == "all" {
        crate::datasets::all_datasets()
    } else {
        vec![
            crate::datasets::find_dataset(&config.dataset).ok_or_else(|| {
                color_eyre::eyre::eyre!(
                    "unknown dataset: {}. Use --dataset list to see available datasets",
                    config.dataset
                )
            })?,
        ]
    };

    let implementations = resolve_gpu_impls(&config.impls);
    let multi_dataset = datasets_list.len() > 1;

    if !config.no_preflight {
        preflight(
            &datasets_list,
            &config.datasets_dir,
            &implementations,
            &config.models_dir,
            &config.root,
            config.pyannote_batch_sizes,
        )?;
    }

    for dataset in datasets_list {
        let job_config = BenchmarkJobConfig {
            models_dir: config.models_dir.clone(),
            datasets_dir: config.datasets_dir.clone(),
            root: config.root.clone(),
            results_dir: config.results_dir.clone(),
            dataset,
            implementations: implementations.clone(),
            max_files: config.max_files,
            max_minutes: config.max_minutes,
            description: config.description.clone(),
            multi_dataset,
            pyannote_batch_sizes: config.pyannote_batch_sizes,
        };

        run_benchmark_job(&job_config, None)?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
pub fn run_benchmark_job(
    config: &BenchmarkJobConfig,
    progress_cb: Option<&(dyn Fn(&ProgressUpdate) + Send + Sync)>,
) -> Result<BenchmarkJobResult> {
    use crate::cmd::wav_duration_seconds;

    let metadata = BenchmarkMetadata::collect();

    println!();
    println!("========== {} ==========", config.dataset.display_name);

    config.dataset.ensure(&config.datasets_dir)?;
    let dataset_dir = config.dataset.dataset_dir(&config.datasets_dir);

    let files = discover_files(&dataset_dir, config.max_files, config.max_minutes as f64)?;
    if files.is_empty() {
        bail!(
            "No paired wav+rttm files found in {}",
            dataset_dir.display()
        );
    }

    let total_audio_seconds: f64 = files
        .iter()
        .map(|(wav, _)| wav_duration_seconds(wav).unwrap_or(0.0))
        .sum();
    let total_audio_minutes = total_audio_seconds / 60.0;

    let run_id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let run_dir = if config.multi_dataset {
        config.results_dir.join(&run_id).join(&config.dataset.id)
    } else {
        config.results_dir.join(&run_id)
    };
    fs::create_dir_all(&run_dir)?;

    if let Some(ref description) = config.description {
        fs::write(run_dir.join("README.md"), format!("{description}\n"))?;
    }

    println!(
        "Found {} files, {total_audio_minutes:.1} min total audio",
        files.len()
    );
    println!("Run ID: {run_id}");
    println!();

    let batch_timeout = Duration::from_secs_f64((total_audio_seconds * 5.0).max(120.0));
    let wav_paths: Vec<&Path> = files.iter().map(|(wav, _)| wav.as_path()).collect();
    let mut all_results: HashMap<String, DerImplResult> = HashMap::new();

    for (impl_name, impl_type) in &config.implementations {
        println!("Running {impl_name}...");

        let benchmark_result = match impl_type {
            ImplType::Speakrs(mode) => {
                run_speakrs_gpu(&config.models_dir, &files, mode, progress_cb)
            }
            ImplType::Pyannote(device) => BatchCommandRunner::pyannote(
                &config.root,
                device,
                &wav_paths,
                config.pyannote_batch_sizes,
            )
            .run_with_retries(batch_timeout),
            _ => {
                println!("  → skipped: not a GPU implementation");
                println!();
                let result = DerImplResult::skipped("not a GPU implementation".to_string());
                super::der::run::write_impl_result(
                    &run_dir,
                    impl_name,
                    &result,
                    total_audio_seconds,
                );
                all_results.insert(impl_name.to_string(), result);
                continue;
            }
        };

        let benchmark_output = match benchmark_result {
            Ok(result) => result,
            Err(err) => {
                println!("  → failed: {err}");
                println!();
                let result = DerImplResult::failed(err.to_string());
                super::der::run::write_impl_result(
                    &run_dir,
                    impl_name,
                    &result,
                    total_audio_seconds,
                );
                all_results.insert(impl_name.to_string(), result);
                continue;
            }
        };

        let acc = DerAccumulation::compute(&files, &benchmark_output.per_file_rttm)?;
        let (der_pct, miss_pct, fa_pct, conf_pct) = acc.der_percentages();

        let rtfx = total_audio_seconds / benchmark_output.total_seconds;
        if let Some(der) = der_pct {
            println!(
                "  → DER: {der:.1}%, Missed: {:.1}%, FA: {:.1}%, Confusion: {:.1}%, Time: {:.1}s, RTFx: {rtfx:.1}x",
                miss_pct.unwrap_or(0.0),
                fa_pct.unwrap_or(0.0),
                conf_pct.unwrap_or(0.0),
                benchmark_output.total_seconds,
            );
        } else {
            println!(
                "  → N/A, Time: {:.1}s, RTFx: {rtfx:.1}x",
                benchmark_output.total_seconds
            );
        }
        println!();

        let result = DerImplResult::completed(
            der_pct,
            miss_pct,
            fa_pct,
            conf_pct,
            benchmark_output.total_seconds,
            acc.file_count,
        );
        super::der::run::write_impl_result(&run_dir, impl_name, &result, total_audio_seconds);
        all_results.insert(impl_name.to_string(), result);
    }

    DerResultsWriter {
        run_dir: &run_dir,
        dataset_name: &config.dataset.display_name,
        implementations: &config.implementations,
        results: &all_results,
        files: &files,
        total_audio_minutes,
        collar: 0.0,
        description: config.description.as_deref(),
        max_files: config.max_files,
        max_minutes: config.max_minutes,
        metadata: &metadata,
        pyannote_batch_sizes: config.pyannote_batch_sizes,
    }
    .write()?;

    Ok(BenchmarkJobResult {
        run_id,
        run_dir,
        total_audio_minutes,
        results: all_results,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn run_benchmark_job(
    _config: &BenchmarkJobConfig,
    _progress_cb: Option<&(dyn Fn(&ProgressUpdate) + Send + Sync)>,
) -> Result<BenchmarkJobResult> {
    bail!("xtask benchmark jobs require the `cuda` feature")
}

#[cfg(feature = "cuda")]
pub fn run_speakrs_gpu(
    models_dir: &Path,
    files: &[(PathBuf, PathBuf)],
    mode: &str,
    progress_cb: Option<&(dyn Fn(&ProgressUpdate) + Send + Sync)>,
) -> Result<BatchRunOutput> {
    use speakrs::inference::{EmbeddingModel, ExecutionMode, SegmentationModel};
    use speakrs::pipeline::{
        CUDA_SEGMENTATION_STEP_SECONDS, DiarizationPipeline, FAST_SEGMENTATION_STEP_SECONDS,
    };

    use crate::wav;

    let execution_mode = match mode {
        "cuda-fast" => ExecutionMode::CudaFast,
        _ => ExecutionMode::Cuda,
    };
    let step = match execution_mode {
        ExecutionMode::CudaFast => FAST_SEGMENTATION_STEP_SECONDS,
        _ => CUDA_SEGMENTATION_STEP_SECONDS,
    };
    let mut seg_model = SegmentationModel::with_mode(
        models_dir.join("segmentation-3.0.onnx"),
        step as f32,
        execution_mode,
    )?;
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir.join("wespeaker-voxceleb-resnet34.onnx"),
        execution_mode,
    )?;
    let mut pipeline = DiarizationPipeline::new(&mut seg_model, &mut emb_model, models_dir)?;

    let mut per_file_rttm = HashMap::new();
    let total_files = files.len();
    let start = std::time::Instant::now();
    for (index, (wav_path, _)) in files.iter().enumerate() {
        let file_id = file_stem_string(wav_path).unwrap_or_else(|_| "file1".to_owned());

        let file_start = std::time::Instant::now();
        let load_start = std::time::Instant::now();
        let (samples, sample_rate) = wav::load_wav_samples(&wav_path.to_string_lossy())?;
        let load_elapsed = load_start.elapsed();
        ensure!(
            sample_rate == 16000,
            "expected 16kHz WAV, got {sample_rate}Hz"
        );

        let pipeline_start = std::time::Instant::now();
        let result = pipeline.run_with_file_id(&samples, &file_id)?;
        let file_elapsed = pipeline_start.elapsed().as_secs_f64();
        let total_elapsed_ms = file_start.elapsed().as_millis();

        per_file_rttm.insert(file_id.clone(), result.rttm(&file_id));

        let cumulative = start.elapsed().as_secs_f64();
        let average = cumulative / (index + 1) as f64;
        let remaining = (total_files - index - 1) as f64 * average;
        let eta = format_eta(remaining);
        let total_elapsed = format_eta(cumulative);
        eprintln!(
            "  [{}/{}] {file_id}: {file_elapsed:.1}s (elapsed {total_elapsed}, ETA {eta}) [{}]",
            index + 1,
            total_files,
            now_stamp()
        );
        tracing::trace!(
            %file_id,
            load_ms = load_elapsed.as_millis(),
            pipeline_ms = (file_elapsed * 1000.0).round() as u64,
            total_ms = total_elapsed_ms,
            "File timing",
        );

        if let Some(callback) = progress_cb {
            callback(&ProgressUpdate {
                impl_name: format!("speakrs {mode}"),
                file_index: index as u32,
                total_files: total_files as u32,
                file_id,
                elapsed_secs: file_elapsed,
            });
        }
    }

    Ok(BatchRunOutput {
        total_seconds: start.elapsed().as_secs_f64(),
        per_file_rttm,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn run_speakrs_gpu(
    _models_dir: &Path,
    _files: &[(PathBuf, PathBuf)],
    _mode: &str,
    _progress_cb: Option<&(dyn Fn(&ProgressUpdate) + Send + Sync)>,
) -> Result<BatchRunOutput> {
    bail!("xtask GPU benchmark path requires the `cuda` feature")
}
