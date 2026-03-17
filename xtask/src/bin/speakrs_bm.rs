use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use color_eyre::eyre::{Result, ensure};
use speakrs::inference::ExecutionMode;
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::{
    DiarizationPipeline, FAST_SEGMENTATION_STEP_SECONDS, SEGMENTATION_STEP_SECONDS,
};
use xtask::cmd::wav_duration_seconds;
use xtask::commands::benchmark::{
    BatchCommandRunner, BatchRunOutput, DerAccumulation, DerImplResult, DerResultsWriter, ImplType,
    discover_files, format_eta, now_stamp,
};
use xtask::datasets;
use xtask::wav;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const BM_IMPLS: &[(&str, &str, &str, ImplType)] = &[
    ("speakrs", "sg", "speakrs CUDA", ImplType::Speakrs("cuda")),
    (
        "speakrs-hybrid",
        "sgh",
        "speakrs CUDA Hybrid",
        ImplType::Speakrs("cuda-hybrid"),
    ),
    (
        "speakrs-fast",
        "sgf",
        "speakrs CUDA Fast",
        ImplType::Speakrs("cuda-fast"),
    ),
    (
        "pyannote",
        "pg",
        "pyannote CUDA",
        ImplType::Pyannote("cuda"),
    ),
];

fn resolve_bm_impl(name: &str) -> Option<usize> {
    BM_IMPLS
        .iter()
        .position(|(cli_id, alias, _, _)| *cli_id == name || *alias == name)
}

#[derive(Parser)]
#[command(name = "speakrs-bm", about = "GPU benchmark runner for speakrs")]
struct Cli {
    /// Dataset to evaluate (use "all" for all, "list" to show available)
    #[arg(long, default_value = "voxconverse-dev")]
    dataset: String,

    /// Implementations to run (use "list" to show available)
    #[arg(long, value_delimiter = ',', value_name = "IMPL")]
    impls: Vec<String>,

    #[arg(long)]
    max_files: Option<u32>,

    #[arg(long)]
    max_minutes: Option<u32>,

    /// Segmentation batch size (sets PYANNOTE_SEGMENTATION_BATCH_SIZE)
    #[arg(long)]
    seg_batch_size: Option<u32>,

    /// Embedding batch size (sets PYANNOTE_EMBEDDING_BATCH_SIZE)
    #[arg(long)]
    emb_batch_size: Option<u32>,

    /// Description of what this benchmark run is testing
    #[arg(long, short = 'd')]
    description: Option<String>,

    /// Skip the pre-flight smoke test
    #[arg(long)]
    no_preflight: bool,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    if let Some(seg) = cli.seg_batch_size {
        // SAFETY: single-threaded CLI, no other threads reading env vars
        unsafe { std::env::set_var("PYANNOTE_SEGMENTATION_BATCH_SIZE", seg.to_string()) };
    }
    if let Some(emb) = cli.emb_batch_size {
        unsafe { std::env::set_var("PYANNOTE_EMBEDDING_BATCH_SIZE", emb.to_string()) };
    }

    if cli.impls.len() == 1 && cli.impls[0] == "list" {
        println!("Available implementations:");
        for (cli_id, alias, display_name, _) in BM_IMPLS {
            println!("  {alias:<4} {cli_id:<15} {display_name}");
        }
        return Ok(());
    }

    if !cli.impls.is_empty() {
        for id in &cli.impls {
            if resolve_bm_impl(id).is_none() {
                let available: Vec<String> = BM_IMPLS
                    .iter()
                    .map(|(id, alias, _, _)| format!("{id} ({alias})"))
                    .collect();
                color_eyre::eyre::bail!(
                    "unknown implementation: {id}. Available: {}",
                    available.join(", ")
                );
            }
        }
    }

    if cli.dataset == "list" {
        println!("Available datasets:");
        for id in datasets::list_dataset_ids() {
            println!("  {id}");
        }
        println!("  all  (run all datasets)");
        return Ok(());
    }

    let datasets_list: Vec<datasets::Dataset> = if cli.dataset == "all" {
        datasets::all_datasets()
    } else {
        vec![datasets::find_dataset(&cli.dataset).ok_or_else(|| {
            color_eyre::eyre::eyre!(
                "unknown dataset: {}. Use --dataset list to see available datasets",
                cli.dataset
            )
        })?]
    };

    let models_dir = resolve_models_dir();
    let datasets_dir = resolve_datasets_dir();
    let root = resolve_root();
    let max_files = cli.max_files.unwrap_or(u32::MAX);
    let max_minutes = cli.max_minutes.unwrap_or(u32::MAX);

    let implementations: Vec<(&str, ImplType)> = if cli.impls.is_empty() {
        BM_IMPLS
            .iter()
            .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    } else {
        BM_IMPLS
            .iter()
            .filter(|(cli_id, alias, _, _)| cli.impls.iter().any(|i| i == cli_id || i == alias))
            .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    };

    // SAFETY: single-threaded CLI
    unsafe { std::env::set_var("SPEAKRS_MODELS_DIR", &models_dir) };

    for dataset in &datasets_list {
        println!();
        println!("========== {} ==========", dataset.display_name);

        dataset.ensure(&datasets_dir)?;
        let dataset_dir = dataset.dataset_dir(&datasets_dir);

        let files = discover_files(&dataset_dir, max_files, max_minutes as f64)?;
        if files.is_empty() {
            eprintln!(
                "No paired wav+rttm files found in {}",
                dataset_dir.display()
            );
            continue;
        }

        let total_audio_seconds: f64 = files
            .iter()
            .map(|(wav, _)| wav_duration_seconds(wav).unwrap_or(0.0))
            .sum();
        let total_audio_minutes = total_audio_seconds / 60.0;

        let run_id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let results_base = resolve_results_dir();
        let run_dir = if datasets_list.len() > 1 {
            results_base.join(&run_id).join(&dataset.id)
        } else {
            results_base.join(&run_id)
        };
        fs::create_dir_all(&run_dir)?;

        if let Some(ref desc) = cli.description {
            fs::write(run_dir.join("README.md"), format!("{desc}\n"))?;
        }

        println!(
            "Found {} files, {total_audio_minutes:.1} min total audio",
            files.len()
        );
        println!("Run ID: {run_id}");
        println!();

        let batch_timeout = Duration::from_secs_f64((total_audio_seconds * 5.0).max(120.0));
        let wav_paths: Vec<&Path> = files.iter().map(|(w, _)| w.as_path()).collect();
        let mut all_results: HashMap<String, DerImplResult> = HashMap::new();

        for (impl_name, impl_type) in &implementations {
            println!("Running {impl_name}...");

            let benchmark_result = match impl_type {
                ImplType::Speakrs(mode) => run_speakrs_gpu(&models_dir, &files, mode),
                ImplType::Pyannote(device) => {
                    BatchCommandRunner::pyannote(&root, device, &wav_paths)
                        .run_with_retries(batch_timeout)
                }
                _ => {
                    println!("  → skipped: not a GPU implementation");
                    println!();
                    all_results.insert(
                        impl_name.to_string(),
                        DerImplResult::skipped("not a GPU implementation".to_string()),
                    );
                    continue;
                }
            };

            let benchmark_output = match benchmark_result {
                Ok(result) => result,
                Err(err) => {
                    println!("  → failed: {err}");
                    println!();
                    all_results.insert(
                        impl_name.to_string(),
                        DerImplResult::failed(err.to_string()),
                    );
                    continue;
                }
            };

            let acc = DerAccumulation::compute(&files, &benchmark_output.per_file_rttm)?;
            let (der_pct, miss_pct, fa_pct, conf_pct) = acc.der_percentages();

            let rtfx = total_audio_seconds / benchmark_output.total_seconds;
            if let Some(d) = der_pct {
                println!(
                    "  → DER: {d:.1}%, Time: {:.1}s, RTFx: {rtfx:.1}x",
                    benchmark_output.total_seconds
                );
            } else {
                println!(
                    "  → N/A, Time: {:.1}s, RTFx: {rtfx:.1}x",
                    benchmark_output.total_seconds
                );
            }
            println!();

            all_results.insert(
                impl_name.to_string(),
                DerImplResult::completed(
                    der_pct,
                    miss_pct,
                    fa_pct,
                    conf_pct,
                    benchmark_output.total_seconds,
                    acc.file_count,
                ),
            );
        }

        DerResultsWriter {
            run_dir: &run_dir,
            dataset_name: &dataset.display_name,
            implementations: &implementations,
            results: &all_results,
            files: &files,
            total_audio_minutes,
            collar: 0.0,
            description: cli.description.as_deref(),
            max_files,
            max_minutes,
        }
        .write()?;
    }

    Ok(())
}

/// Run speakrs GPU in-process: load models once, diarize all files
fn run_speakrs_gpu(
    models_dir: &Path,
    files: &[(PathBuf, PathBuf)],
    mode: &str,
) -> Result<BatchRunOutput> {
    let execution_mode = match mode {
        "cuda-hybrid" => ExecutionMode::CudaHybrid,
        "cuda-fast" => ExecutionMode::CudaFast,
        _ => ExecutionMode::Cuda,
    };
    let step = match execution_mode {
        ExecutionMode::CudaFast => FAST_SEGMENTATION_STEP_SECONDS,
        _ => SEGMENTATION_STEP_SECONDS,
    };
    let mut seg_model = SegmentationModel::with_mode(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        step as f32,
        execution_mode,
    )?;
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
        execution_mode,
    )?;
    let mut pipeline = DiarizationPipeline::new(&mut seg_model, &mut emb_model, models_dir)?;

    let mut per_file_rttm = HashMap::new();
    let total_files = files.len();
    let start = Instant::now();

    for (i, (wav_path, _)) in files.iter().enumerate() {
        let file_id = wav_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file1".to_string());

        let (samples, sr) = wav::load_wav_samples(&wav_path.to_string_lossy())?;
        ensure!(sr == 16000, "expected 16kHz WAV, got {sr}Hz");

        let file_start = Instant::now();
        let result = pipeline.run_with_file_id(&samples, &file_id)?;
        let file_elapsed = file_start.elapsed().as_secs_f64();

        per_file_rttm.insert(file_id.clone(), result.rttm);

        let cumulative = start.elapsed().as_secs_f64();
        let avg = cumulative / (i + 1) as f64;
        let remaining = (total_files - i - 1) as f64 * avg;
        let eta = format_eta(remaining);
        let total_elapsed = format_eta(cumulative);
        eprintln!(
            "  [{}/{}] {file_id}: {file_elapsed:.1}s (elapsed {total_elapsed}, ETA {eta}) [{}]",
            i + 1,
            total_files,
            now_stamp()
        );
    }

    Ok(BatchRunOutput {
        total_seconds: start.elapsed().as_secs_f64(),
        per_file_rttm,
    })
}

fn resolve_models_dir() -> PathBuf {
    std::env::var("SPEAKRS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/workspace/models"))
}

fn resolve_datasets_dir() -> PathBuf {
    std::env::var("SPEAKRS_DATASETS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/workspace/datasets"))
}

fn resolve_root() -> PathBuf {
    std::env::var("SPEAKRS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/workspace"))
}

fn resolve_results_dir() -> PathBuf {
    std::env::var("SPEAKRS_RESULTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/workspace/_benchmarks"))
}
