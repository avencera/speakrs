use std::path::PathBuf;

use clap::Parser;
use color_eyre::eyre::Result;
use xtask::commands::benchmark::{
    GpuBenchmarkSuiteConfig, PyannoteBatchSizes, gpu_impls, run_gpu_benchmark_suite,
    validate_gpu_impls,
};
use xtask::datasets;

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
        for (cli_id, alias, display_name, _) in gpu_impls() {
            println!("  {alias:<4} {cli_id:<15} {display_name}");
        }
        return Ok(());
    }

    validate_gpu_impls(&cli.impls)?;

    if cli.dataset == "list" {
        println!("Available datasets:");
        for id in datasets::list_dataset_ids() {
            println!("  {id}");
        }
        println!("  all  (run all datasets)");
        return Ok(());
    }

    run_gpu_benchmark_suite(&GpuBenchmarkSuiteConfig {
        dataset: cli.dataset,
        impls: cli.impls,
        max_files: cli.max_files.unwrap_or(u32::MAX),
        max_minutes: cli.max_minutes.unwrap_or(u32::MAX),
        description: cli.description,
        no_preflight: cli.no_preflight,
        models_dir: cli.models_dir,
        datasets_dir: cli.datasets_dir,
        root: cli.root,
        results_dir: cli.results_dir,
        pyannote_batch_sizes,
    })
}
