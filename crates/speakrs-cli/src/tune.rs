//! `speakrs cuda tune`: measure CUDA kernel choices on this GPU and save a tune file

use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};
use eyre::WrapErr;
use speakrs::ExecutionMode;
use speakrs::inference::cuda::{CudaMath, CudaTuneOptions, CudaTuneReport, tune_cuda};
use speakrs_sdk::{ModelStore, PrepareModelsOptions};

#[derive(Subcommand)]
pub(crate) enum CudaCommand {
    Tune(TuneArgs),
}

/// Measure CUDA settings and save the selected settings
///
/// A tune file applies only to the speakrs version and CUDA feature that wrote it
#[derive(Args)]
pub(crate) struct TuneArgs {
    /// Model directory (default: the CUDA models in the download cache)
    #[arg(long, alias = "models", value_name = "PATH")]
    models_dir: Option<PathBuf>,

    /// Model download cache used without --models-dir (default: the Hugging Face hub cache)
    #[arg(long, value_name = "DIR", conflicts_with = "models_dir")]
    cache_dir: Option<PathBuf>,

    /// Output file
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// CUDA device number
    #[arg(long, value_name = "N", default_value_t = 0)]
    device: usize,

    /// Measure without writing a file
    #[arg(long)]
    dry_run: bool,

    /// Also time Library (excludes load and handle costs)
    #[arg(long)]
    include_library: bool,

    /// Segmentation arithmetic
    #[arg(long, value_enum, value_name = "MODE", default_value_t = Math::Fp32)]
    segmentation_math: Math,

    /// Embedding arithmetic
    #[arg(long, value_enum, value_name = "MODE", default_value_t = Math::Tf32)]
    embedding_math: Math,
}

#[derive(Clone, Copy, ValueEnum)]
enum Math {
    Fp32,
    Tf32,
}

impl From<Math> for CudaMath {
    fn from(math: Math) -> Self {
        match math {
            Math::Fp32 => CudaMath::Fp32,
            Math::Tf32 => CudaMath::Tf32,
        }
    }
}

pub(crate) fn run(command: CudaCommand) -> eyre::Result<()> {
    let CudaCommand::Tune(args) = command;
    let model_dir = match args.models_dir {
        Some(dir) => dir,
        None => {
            let store = args
                .cache_dir
                .map_or_else(ModelStore::default, ModelStore::new);
            store
                .prepare(PrepareModelsOptions::new(ExecutionMode::Cuda), &|file| {
                    eprintln!("downloading {file}")
                })
                .wrap_err("failed to prepare models")?
                .model_dir
        }
    };

    let options = CudaTuneOptions {
        model_dir,
        output_path: args.output,
        device: args.device,
        dry_run: args.dry_run,
        include_library: args.include_library,
        segmentation_math: args.segmentation_math.into(),
        embedding_math: args.embedding_math.into(),
    };
    let report = tune_cuda(&options)?;
    print_report(&report);
    Ok(())
}

fn print_report(report: &CudaTuneReport) {
    println!("Device: {}", report.device);
    println!("Output: {}", report.path.display());
    let boundary_width = report
        .rows
        .iter()
        .map(|row| row.boundary.len())
        .max()
        .unwrap_or(0)
        .max(8);
    let candidate_width = report
        .rows
        .iter()
        .flat_map(|row| &row.candidates)
        .map(|measurement| measurement.candidate.len())
        .max()
        .unwrap_or(0)
        .max(9);
    println!(
        "{:<boundary_width$} {:>5} {:<7} {:<candidate_width$} {:>12}  Selected",
        "Boundary", "Batch", "Math", "Candidate", "Median (ms)"
    );
    for row in &report.rows {
        let math = match row.math {
            CudaMath::Fp32 => "fp32",
            CudaMath::Tf32 => "tf32",
        };
        for measurement in &row.candidates {
            let selected = if measurement.candidate == row.selected {
                "*"
            } else {
                ""
            };
            println!(
                "{:<boundary_width$} {:>5} {:<7} {:<candidate_width$} {:>12.3}  {selected}",
                row.boundary, row.batch, math, measurement.candidate, measurement.median_ms
            );
        }
    }

    if report.written {
        println!("Wrote file {}", report.path.display());
    } else {
        println!("Dry run: no file was written");
    }
}
