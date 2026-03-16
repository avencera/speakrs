use std::path::PathBuf;

use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use xtask::commands;
use xtask::commands::diarize::DiarizeMode;
use xtask::commands::gpu::{Backend, RUNPOD_GPU_FALLBACKS};

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
    /// Remote GPU benchmarking (RunPod, Vast.ai)
    Gpu {
        #[command(subcommand)]
        cmd: GpuCmd,
    },
    /// Run speaker diarization on WAV files
    Diarize {
        #[arg(long, default_value = "cpu", value_parser = clap::value_parser!(DiarizeMode))]
        mode: DiarizeMode,
        /// WAV files to diarize
        wav_files: Vec<PathBuf>,
    },
    /// Profile ORT embedding inference strategies
    ProfileOrtEmbedding {
        /// Mode: borrow, owned, prealloc, stream-borrow, stream-owned, stream-prealloc, stream-batched
        mode: String,
        /// Path to WAV file
        wav_path: PathBuf,
        #[arg(long, default_value_t = 100)]
        iterations: usize,
        #[arg(long, default_value_t = 100)]
        log_every: usize,
    },
    /// Profile pipeline stages
    ProfileStages {
        /// Mode: seg-only, embed-stream, embed-store, embed-repeat
        mode: String,
        /// Path to WAV file
        wav_path: PathBuf,
        #[arg(long, default_value_t = 100)]
        iterations: usize,
        #[arg(long, default_value_t = 100)]
        log_every: usize,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Export ONNX models and PLDA params, then build native CoreML bundles on macOS
    Export,
    /// Run CoreML model conversion only
    ExportCoreml,
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
    /// DER evaluation on benchmark datasets
    Der {
        /// Dataset to evaluate (use "all" for all datasets, "list" to show available)
        #[arg(long, default_value = "voxconverse-dev")]
        dataset: String,
        #[arg(long)]
        max_files: Option<u32>,
        #[arg(long)]
        max_minutes: Option<u32>,
        /// Description of what this benchmark run is testing
        #[arg(long, short = 'd')]
        description: Option<String>,
        /// Implementations to run (omit for all, use "list" to show available)
        #[arg(long, value_delimiter = ',', value_name = "IMPL")]
        impls: Vec<String>,
        /// Skip the pre-flight smoke test
        #[arg(long)]
        no_preflight: bool,
        /// Segmentation batch size (sets PYANNOTE_SEGMENTATION_BATCH_SIZE)
        #[arg(long)]
        seg_batch_size: Option<u32>,
        /// Embedding batch size (sets PYANNOTE_EMBEDDING_BATCH_SIZE)
        #[arg(long)]
        emb_batch_size: Option<u32>,
    },
}

#[derive(Subcommand)]
enum GpuCmd {
    /// Provision a new GPU instance
    Create {
        /// Instance name
        name: String,
        /// Backend provider
        #[arg(long, default_value = "runpod")]
        backend: Backend,
        /// GPU type (RunPod GPU type ID). Falls back through preferred list if omitted
        #[arg(long)]
        gpu_type: Option<String>,
        /// Minimum TFLOPS for GPU search (vast.ai only)
        #[arg(long, default_value_t = 10.0)]
        min_tflops: f64,
    },
    /// Git clone/pull + download models + build on a running instance
    Setup {
        /// Instance name (omit for fzf picker)
        name: Option<String>,
        /// Git branch to checkout
        #[arg(long, default_value = "master")]
        branch: String,
    },
    /// Open an interactive SSH session to the GPU instance
    Ssh {
        /// Instance name (omit for fzf picker)
        name: Option<String>,
    },
    /// Tear down GPU instance(s)
    #[command(alias = "rm")]
    Destroy {
        /// Instance name (omit for fzf picker)
        name: Option<String>,
        /// Destroy all instances
        #[arg(long)]
        all: bool,
    },
    /// Resume a stopped GPU instance
    Start {
        /// Instance name (omit for fzf picker)
        name: Option<String>,
    },
    /// Import an existing running pod as a local instance
    Import {
        /// Local instance name
        name: String,
        /// Backend provider
        #[arg(long, default_value = "runpod")]
        backend: Backend,
    },
    /// List all GPU instances
    #[command(alias = "ls")]
    List,
    /// Remove local instances that no longer exist on their backend
    Sync,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    match cli.cmd {
        Command::Models { cmd } => match cmd {
            ModelsCmd::Export => commands::models::export(),
            ModelsCmd::ExportCoreml => commands::models::export_coreml(),
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
                dataset,
                max_files,
                max_minutes,
                description,
                impls,
                no_preflight,
                seg_batch_size,
                emb_batch_size,
            } => commands::benchmark::der(
                &dataset,
                max_files.unwrap_or(u32::MAX),
                max_minutes.unwrap_or(u32::MAX),
                description.as_deref(),
                &impls,
                no_preflight,
                seg_batch_size,
                emb_batch_size,
            ),
        },
        Command::Gpu { cmd } => match cmd {
            GpuCmd::Create {
                name,
                backend,
                gpu_type,
                min_tflops,
            } => {
                let gpu_types: Vec<&str> = match &gpu_type {
                    Some(t) => vec![t.as_str()],
                    None => RUNPOD_GPU_FALLBACKS.to_vec(),
                };
                commands::gpu::create(&name, backend, &gpu_types, min_tflops)
            }
            GpuCmd::Setup { name, branch } => commands::gpu::setup(name.as_deref(), &branch),
            GpuCmd::Ssh { name } => commands::gpu::ssh(name.as_deref()),
            GpuCmd::Destroy { name, all } => commands::gpu::destroy(name.as_deref(), all),
            GpuCmd::Start { name } => commands::gpu::start(name.as_deref()),
            GpuCmd::Import { name, backend } => commands::gpu::import(&name, backend),
            GpuCmd::List => commands::gpu::list(),
            GpuCmd::Sync => commands::gpu::sync(),
        },
        Command::Diarize { mode, wav_files } => commands::diarize::run(mode, wav_files),
        Command::ProfileOrtEmbedding {
            mode,
            wav_path,
            iterations,
            log_every,
        } => commands::profile_ort_embedding::run(
            &mode,
            &wav_path.to_string_lossy(),
            iterations,
            log_every,
        ),
        Command::ProfileStages {
            mode,
            wav_path,
            iterations,
            log_every,
        } => {
            commands::profile_stages::run(&mode, &wav_path.to_string_lossy(), iterations, log_every)
        }
    }
}
