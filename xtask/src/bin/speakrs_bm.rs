use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use color_eyre::eyre::{Result, ensure};
use xtask::cmd::wav_duration_seconds;
use xtask::commands::benchmark::{
    BatchCommandRunner, BenchmarkJobConfig, ImplType, PyannoteBatchSizes, discover_files,
    run_benchmark_job, run_speakrs_gpu,
};
use xtask::datasets;

const BM_IMPLS: &[(&str, &str, &str, ImplType)] = &[
    ("speakrs", "sg", "speakrs CUDA", ImplType::Speakrs("cuda")),
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

pub fn resolve_bm_impls(impls: &[String]) -> Vec<(&'static str, ImplType)> {
    if impls.is_empty() {
        BM_IMPLS
            .iter()
            .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    } else {
        BM_IMPLS
            .iter()
            .filter(|(cli_id, alias, _, _)| impls.iter().any(|i| i == cli_id || i == alias))
            .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    }
}

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

    /// Override pyannote segmentation batch size
    #[arg(long)]
    seg_batch_size: Option<u32>,

    /// Override pyannote embedding batch size
    #[arg(long)]
    emb_batch_size: Option<u32>,

    /// Description of what this benchmark run is testing
    #[arg(long, short = 'd')]
    description: Option<String>,

    /// Skip the pre-flight smoke test
    #[arg(long)]
    no_preflight: bool,

    /// Path to models directory
    #[arg(long, env = "SPEAKRS_MODELS_DIR", default_value = "/workspace/models")]
    models_dir: PathBuf,

    /// Path to datasets directory
    #[arg(
        long,
        env = "SPEAKRS_DATASETS_DIR",
        default_value = "/workspace/datasets"
    )]
    datasets_dir: PathBuf,

    /// Project root directory
    #[arg(long, env = "SPEAKRS_ROOT", default_value = "/workspace")]
    root: PathBuf,

    /// Path to results directory
    #[arg(
        long,
        env = "SPEAKRS_RESULTS_DIR",
        default_value = "/workspace/_benchmarks"
    )]
    results_dir: PathBuf,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let pyannote_batch_sizes =
        PyannoteBatchSizes::from_overrides(cli.seg_batch_size, cli.emb_batch_size);

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

    let max_files = cli.max_files.unwrap_or(u32::MAX);
    let max_minutes = cli.max_minutes.unwrap_or(u32::MAX);
    let implementations = resolve_bm_impls(&cli.impls);
    let multi_dataset = datasets_list.len() > 1;

    if !cli.no_preflight {
        preflight(
            &datasets_list,
            &cli.datasets_dir,
            &implementations,
            &cli.models_dir,
            &cli.root,
            pyannote_batch_sizes,
        )?;
    }

    for dataset in datasets_list {
        let config = BenchmarkJobConfig {
            models_dir: cli.models_dir.clone(),
            datasets_dir: cli.datasets_dir.clone(),
            root: cli.root.clone(),
            results_dir: cli.results_dir.clone(),
            dataset,
            implementations: implementations.clone(),
            max_files,
            max_minutes,
            description: cli.description.clone(),
            multi_dataset,
            pyannote_batch_sizes,
        };

        run_benchmark_job(&config, None)?;
    }

    Ok(())
}

fn preflight(
    datasets: &[datasets::Dataset],
    datasets_dir: &Path,
    implementations: &[(&str, ImplType)],
    models_dir: &Path,
    root: &Path,
    pyannote_batch_sizes: PyannoteBatchSizes,
) -> Result<()> {
    let first_dataset = &datasets[0];
    first_dataset.ensure(datasets_dir)?;
    let first_dir = first_dataset.dataset_dir(datasets_dir);
    let preflight_files = discover_files(&first_dir, 1, f64::MAX)?;

    if preflight_files.is_empty() {
        eprintln!("warning: no files for preflight, skipping");
        return Ok(());
    }

    let (wav_path, _) = &preflight_files[0];
    let stem = wav_path.file_stem().unwrap().to_string_lossy();
    let dur = wav_duration_seconds(wav_path).unwrap_or(0.0);
    println!();
    println!("=== Pre-flight check ({stem}, {dur:.0}s) ===");

    for (impl_name, impl_type) in implementations {
        let result = match impl_type {
            ImplType::Speakrs(mode) => run_speakrs_gpu(models_dir, &preflight_files, mode, None),
            ImplType::Pyannote(device) => BatchCommandRunner::pyannote(
                root,
                device,
                &[wav_path.as_path()],
                pyannote_batch_sizes,
            )
            .run_with_retries(Duration::from_secs(180)),
            _ => continue,
        };

        let output = result
            .map_err(|err| color_eyre::eyre::eyre!("preflight failed for {impl_name}: {err}"))?;

        ensure!(
            output
                .per_file_rttm
                .values()
                .any(|v: &String| !v.trim().is_empty()),
            "preflight failed for {impl_name}: empty RTTM output"
        );

        println!("  {impl_name:<22} ok ({:.1}s)", output.total_seconds);
    }

    println!();
    Ok(())
}
