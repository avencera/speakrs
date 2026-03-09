mod audio;
mod cargo;
mod cmd;
mod commands;
mod compare_rttm;
mod python;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;

#[derive(Parser)]
#[command(name = "xtask", about = "Development tasks for speakrs")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Model management (download, convert, deploy)
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Test fixture generation
    Fixtures {
        #[command(subcommand)]
        cmd: FixturesCmd,
    },
    /// Compare diarization outputs
    Compare {
        #[command(subcommand)]
        cmd: CompareCmd,
    },
    /// Benchmark diarization implementations
    Benchmark {
        #[command(subcommand)]
        cmd: BenchmarkCmd,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Download ONNX models and PLDA params, then build native CoreML bundles on macOS
    Download,
    /// Run CoreML model conversion only
    DownloadCoreml,
    /// Compare CoreML vs ONNX model outputs
    CompareCoreml,
    /// Upload models to HuggingFace Hub
    Deploy,
}

#[derive(Subcommand)]
enum FixturesCmd {
    /// Regenerate test fixtures via Python
    Generate,
}

#[derive(Subcommand)]
enum CompareCmd {
    /// Run speakrs and pyannote side-by-side on the same audio
    Run {
        source: String,
        #[arg(long, default_value = "cpu")]
        python_device: String,
        #[arg(long, default_value = "cpu")]
        rust_mode: String,
    },
    /// Compare two RTTM files
    Rttm { a: PathBuf, b: PathBuf },
    /// Compare speakrs, FluidAudio, and pyannote CPU on the same audio
    Accuracy {
        source: String,
        #[arg(long, default_value = "pyannote-mps")]
        rust_mode: String,
    },
}

#[derive(Subcommand)]
enum BenchmarkCmd {
    /// Benchmark speakrs vs pyannote on the same audio
    Run {
        source: String,
        #[arg(long, default_value = "auto")]
        python_device: String,
        #[arg(long, default_value_t = 1)]
        runs: u32,
        #[arg(long, default_value_t = 1)]
        warmups: u32,
        #[arg(long, default_value = "cpu")]
        rust_mode: String,
    },
    /// Multi-tool benchmark (speakrs CoreML, pyannote MPS, pyannote-rs, FluidAudio)
    Compare {
        source: String,
        #[arg(long, default_value_t = 1)]
        runs: u32,
        #[arg(long, default_value_t = 1)]
        warmups: u32,
    },
    /// DER evaluation on VoxConverse dev set
    Der {
        #[arg(long, default_value_t = 10)]
        max_files: u32,
        #[arg(long, default_value_t = 30)]
        max_minutes: u32,
    },
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    match cli.cmd {
        Command::Models { cmd } => match cmd {
            ModelsCmd::Download => commands::models::download(),
            ModelsCmd::DownloadCoreml => commands::models::download_coreml(),
            ModelsCmd::CompareCoreml => commands::models::compare_coreml(),
            ModelsCmd::Deploy => commands::models::deploy(),
        },
        Command::Fixtures { cmd } => match cmd {
            FixturesCmd::Generate => commands::fixtures::generate(),
        },
        Command::Compare { cmd } => match cmd {
            CompareCmd::Run {
                source,
                python_device,
                rust_mode,
            } => commands::compare::run(&source, &python_device, &rust_mode),
            CompareCmd::Rttm { a, b } => commands::compare::rttm(&a, &b),
            CompareCmd::Accuracy { source, rust_mode } => {
                commands::compare::accuracy(&source, &rust_mode)
            }
        },
        Command::Benchmark { cmd } => match cmd {
            BenchmarkCmd::Run {
                source,
                python_device,
                runs,
                warmups,
                rust_mode,
            } => commands::benchmark::run(&source, &python_device, runs, warmups, &rust_mode),
            BenchmarkCmd::Compare {
                source,
                runs,
                warmups,
            } => commands::benchmark::compare(&source, runs, warmups),
            BenchmarkCmd::Der {
                max_files,
                max_minutes,
            } => commands::benchmark::der(max_files, max_minutes),
        },
    }
}
