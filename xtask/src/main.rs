use std::path::PathBuf;

use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use tracing_subscriber::EnvFilter;
use xtask::commands;
use xtask::commands::diarize::DiarizeMode;

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
    /// Local benchmarks (DER evaluation, single-file, multi-tool)
    Bench {
        #[command(subcommand)]
        cmd: BenchCmd,
    },
    /// Remote GPU benchmarks via dstack
    Dstack {
        #[command(subcommand)]
        cmd: DstackCmd,
    },
    /// Download and manage benchmark datasets
    Dataset {
        #[command(subcommand)]
        cmd: DatasetCmd,
    },
    /// Run speaker diarization on WAV files
    Diarize {
        #[arg(long, default_value = "cpu", value_parser = clap::value_parser!(DiarizeMode))]
        mode: DiarizeMode,
        /// Path to models directory
        #[arg(long, env = "SPEAKRS_MODELS_DIR")]
        models_dir: Option<PathBuf>,
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
        /// Path to ONNX embedding model
        #[arg(long)]
        model_path: Option<PathBuf>,
        /// Batch size for stream-batched mode
        #[arg(long)]
        batch_size: Option<usize>,
        /// Use default ORT session configuration
        #[arg(long)]
        ort_defaults: bool,
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
enum BenchCmd {
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
    /// DER evaluation on benchmark datasets or a single file
    Der {
        /// Dataset to evaluate (use "all" for all datasets, "list" to show available)
        #[arg(long, default_value = "voxconverse-dev")]
        dataset: String,
        /// Single WAV file to evaluate (bypasses dataset loading)
        #[arg(long, requires = "rttm", conflicts_with_all = ["dataset", "max_files", "max_minutes"])]
        file: Option<PathBuf>,
        /// Reference RTTM file (required when --file is used)
        #[arg(long, requires = "file")]
        rttm: Option<PathBuf>,
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
        /// Seconds to sleep between implementations (for thermal cooldown)
        #[arg(long, short = 's')]
        sleep_between: Option<u64>,
    },
}

#[derive(Subcommand)]
enum DstackCmd {
    /// Run a GPU benchmark
    Bench {
        /// Run name (used for S3 path and dstack logs/attach/stop)
        name: String,
        #[arg(long, default_value = "voxconverse-dev")]
        dataset: String,
        #[arg(long, value_delimiter = ',')]
        impls: Vec<String>,
        #[arg(long)]
        max_files: Option<u32>,
        #[arg(long)]
        max_minutes: Option<u32>,
        /// Reuse existing fleet pod instead of provisioning a new one
        #[arg(long, short = 'R')]
        reuse: bool,
        /// Submit and exit immediately (default: attach to logs)
        #[arg(long, short = 'd')]
        detach: bool,
    },
    /// Run GPU benchmarks in parallel (one dataset per GPU)
    #[command(alias = "bp")]
    BenchParallel {
        /// Run name prefix (tasks named {name}-{dataset})
        name: String,
        /// Datasets to run (comma-separated, or "all")
        #[arg(long, value_delimiter = ',', default_value = "all")]
        dataset: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        impls: Vec<String>,
        #[arg(long)]
        max_files: Option<u32>,
        #[arg(long)]
        max_minutes: Option<u32>,
        /// Reuse existing fleet pod instead of provisioning a new one
        #[arg(long, short = 'R')]
        reuse: bool,
    },
    /// Start a reusable GPU fleet (30min idle timeout)
    Fleet,
    /// Reattach to a running task (logs + port forwarding, Ctrl+C safe)
    Attach { name: String },
    /// Stream logs from a running task (Ctrl+C safe)
    Logs { name: String },
    /// Show status of all dstack runs
    Ps,
    /// Stop a dstack run or fleet
    #[command(alias = "kill")]
    Stop { name: String },
    /// Start interactive GPU dev environment
    Dev,
    /// Download benchmark results from S3
    Download { name: String },
    /// Delete a path from the S3 bucket (with confirmation)
    Delete {
        /// S3 path to delete (e.g. "benchmarks/my-run" or "benchmarks/my-run/20260318-1430")
        path: String,
    },
}

#[derive(Subcommand)]
enum DatasetCmd {
    /// Download one or all datasets (use "list" as id to show available)
    Ensure {
        /// Dataset id, or "all"
        #[arg(default_value = "all")]
        id: String,
    },
    /// Upload local datasets to Tigris S3
    Upload {
        /// Dataset id, or "all"
        #[arg(default_value = "all")]
        id: String,
    },
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
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
        Command::Bench { cmd } => match cmd {
            BenchCmd::Run {
                source,
                python_device,
                runs,
                warmups,
                rust_mode,
            } => commands::benchmark::run(&source, &python_device, runs, warmups, &rust_mode),
            BenchCmd::Compare {
                source,
                runs,
                warmups,
            } => commands::benchmark::compare(&source, runs, warmups),
            BenchCmd::Der {
                dataset,
                file,
                rttm,
                max_files,
                max_minutes,
                description,
                impls,
                no_preflight,
                seg_batch_size,
                emb_batch_size,
                sleep_between,
            } => commands::benchmark::der(commands::benchmark::DerArgs {
                dataset_id: dataset,
                file,
                rttm,
                max_files: max_files.unwrap_or(u32::MAX),
                max_minutes: max_minutes.unwrap_or(u32::MAX),
                description,
                impls,
                no_preflight,
                seg_batch_size,
                emb_batch_size,
                sleep_between,
            }),
        },
        Command::Dstack { cmd } => match cmd {
            DstackCmd::Bench {
                name,
                dataset,
                impls,
                max_files,
                max_minutes,
                reuse,
                detach,
            } => commands::dstack::bench(
                &name,
                &dataset,
                &impls,
                max_files,
                max_minutes,
                reuse,
                detach,
            ),
            DstackCmd::BenchParallel {
                name,
                dataset,
                impls,
                max_files,
                max_minutes,
                reuse,
            } => commands::dstack::bench_parallel(
                &name,
                &dataset,
                &impls,
                max_files,
                max_minutes,
                reuse,
            ),
            DstackCmd::Fleet => commands::dstack::fleet(),
            DstackCmd::Attach { name } => commands::dstack::attach(&name),
            DstackCmd::Logs { name } => commands::dstack::logs(&name),
            DstackCmd::Ps => commands::dstack::ps(),
            DstackCmd::Stop { name } => commands::dstack::stop(&name),
            DstackCmd::Dev => commands::dstack::dev(),
            DstackCmd::Download { name } => commands::dstack::download(&name),
            DstackCmd::Delete { path } => commands::dstack::delete(&path),
        },
        Command::Dataset { cmd } => {
            use xtask::cmd::project_root;
            use xtask::datasets::{self, S5cmd};

            let base_dir = project_root().join("fixtures/datasets");

            match cmd {
                DatasetCmd::Ensure { id } => {
                    if id == "list" {
                        for ds_id in datasets::list_dataset_ids() {
                            println!("  {ds_id}");
                        }
                        return Ok(());
                    }

                    let targets = if id == "all" {
                        datasets::all_datasets()
                    } else {
                        vec![
                            datasets::find_dataset(&id)
                                .ok_or_else(|| color_eyre::eyre::eyre!("unknown dataset: {id}"))?,
                        ]
                    };

                    for ds in &targets {
                        println!("--- {} ---", ds.display_name);
                        ds.ensure(&base_dir)?;
                    }
                    Ok(())
                }
                DatasetCmd::Upload { id } => {
                    if !S5cmd::available() {
                        color_eyre::eyre::bail!("s5cmd not available or AWS_ACCESS_KEY_ID not set");
                    }

                    let targets = if id == "all" {
                        datasets::all_datasets()
                            .into_iter()
                            .filter(|d| d.id != "voxconverse-dev")
                            .collect()
                    } else {
                        vec![
                            datasets::find_dataset(&id)
                                .ok_or_else(|| color_eyre::eyre::eyre!("unknown dataset: {id}"))?,
                        ]
                    };

                    for ds in &targets {
                        let ds_dir = ds.dataset_dir(&base_dir);
                        if !ds_dir.join("wav").is_dir() || !ds_dir.join("rttm").is_dir() {
                            println!("Skipping {} (not downloaded yet)", ds.id);
                            continue;
                        }
                        println!("Uploading {}...", ds.id);
                        S5cmd::upload(&ds.id, &ds_dir)?;
                    }
                    Ok(())
                }
            }
        }
        Command::Diarize {
            mode,
            models_dir,
            wav_files,
        } => commands::diarize::run(mode, models_dir, wav_files),
        Command::ProfileOrtEmbedding {
            mode,
            wav_path,
            iterations,
            log_every,
            model_path,
            batch_size,
            ort_defaults,
        } => commands::profile_ort_embedding::run(
            &mode,
            &wav_path.to_string_lossy(),
            iterations,
            log_every,
            model_path,
            batch_size,
            ort_defaults,
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
