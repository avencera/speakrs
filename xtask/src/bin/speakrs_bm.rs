use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use color_eyre::eyre::Result;
use speakrs::ModelManager;
use speakrs::inference::ExecutionMode;
use xtask::catalog::{ImplementationCatalog, ImplementationId, RunnerKind, SpeakrsMode};
use xtask::commands::benchmark::{
    GpuBenchmarkSuiteConfig, PyannoteBatchSizes, gpu_impls, run_gpu_benchmark_suite,
    validate_gpu_impls,
};
use xtask::datasets;

#[derive(Clone, Copy, ValueEnum)]
enum Models {
    Online,
    Local,
}

#[derive(Parser)]
#[command(name = "speakrs-bm", about = "GPU benchmark runner")]
struct Cli {
    /// Dataset to evaluate ("all" for all, "list" to show available)
    #[arg(long, default_value = "voxconverse-dev")]
    dataset: String,

    /// Implementations to run ("speakrs" selects native CUDA, "list" shows available)
    #[arg(long, value_delimiter = ',', value_name = "IMPL", value_parser = parse_implementation)]
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

    /// Short note for this benchmark run
    #[arg(long, short = 'd')]
    description: Option<String>,

    /// Skip the pre-flight smoke test
    #[arg(long)]
    no_preflight: bool,

    /// Fetch pinned public models, or use only the local directory
    #[arg(long, value_enum, default_value = "online")]
    models: Models,

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
        for (cli_id, alias, display_name) in gpu_impls() {
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

    if matches!(cli.models, Models::Online) {
        fetch_models(&cli.models_dir, &cli.impls)?;
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

fn fetch_models(destination: &Path, implementations: &[String]) -> Result<()> {
    let selected = if implementations.is_empty() {
        ImplementationCatalog::gpu().collect::<Vec<_>>()
    } else {
        ImplementationCatalog::resolve_many(implementations)?
    };
    let modes = selected.into_iter().filter_map(|spec| match spec.runner {
        RunnerKind::Speakrs(SpeakrsMode::Cuda) => Some(ExecutionMode::Cuda),
        RunnerKind::Speakrs(SpeakrsMode::CudaFast) => Some(ExecutionMode::CudaFast),
        _ => None,
    });

    fs::create_dir_all(destination)?;
    // the model repository is public, so downloads are anonymous and no token touches disk
    let cache_dir = destination.join(".cache/hub");

    for mode in modes {
        // keep the cache inside the workspace and let ModelManager own the asset catalog
        let manager = ModelManager::with_cache_dir(cache_dir.clone())?;
        let snapshot = manager.ensure(mode)?;
        copy_snapshot(&snapshot, destination)?;
    }
    Ok(())
}

fn copy_snapshot(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_snapshot(&entry.path(), &target)?;
            continue;
        }
        fs::copy(entry.path(), target)?;
    }
    Ok(())
}

fn parse_implementation(value: &str) -> std::result::Result<String, std::convert::Infallible> {
    let canonical = match value {
        "speakrs" => ImplementationId::SpeakrsCuda.as_str(),
        other => other,
    };
    Ok(canonical.to_owned())
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn speakrs_alias_selects_native_cuda() {
        let cli = Cli::try_parse_from(["speakrs-bm", "--impls", "speakrs"]).unwrap();
        assert_eq!(cli.impls, ["cuda"]);
    }

    #[test]
    fn aliases_and_list_remain_available() {
        let cli = Cli::try_parse_from(["speakrs-bm", "--impls", "sg,sgf"]).unwrap();
        assert_eq!(cli.impls, ["sg", "sgf"]);
        let cli = Cli::try_parse_from(["speakrs-bm", "--impls", "list"]).unwrap();
        assert_eq!(cli.impls, ["list"]);
    }
}
