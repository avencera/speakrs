use std::path::PathBuf;

use crate::commands;
use crate::commands::diarize::{ChunkEmbeddingComputeUnits, DiarizeMode};
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;

#[derive(Parser)]
#[command(name = "xtask", about = "Development commands for speakrs")]
pub struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

impl Cli {
    pub fn run(self) -> Result<()> {
        self.cmd.run()
    }
}

#[derive(Subcommand)]
enum Command {
    /// Model commands
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Fixture generation
    Fixtures {
        #[command(subcommand)]
        cmd: FixturesCmd,
    },
    /// Informal RTTM timeline comparison
    Compare {
        #[command(subcommand)]
        cmd: CompareCmd,
    },
    /// Dataset DER measurement and scoring
    Benchmark {
        #[command(subcommand)]
        cmd: BenchmarkCmd,
    },
    /// Reproducible macOS performance and DER experiments
    MacExperiment {
        #[command(subcommand)]
        cmd: commands::mac_experiment::MacExperimentCommand,
    },
    /// Remote GPU benchmarks via dstack
    Dstack {
        #[command(subcommand)]
        cmd: DstackCmd,
    },
    /// Dataset commands
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
        /// Compute units for native embedding: all, cpu-and-neural-engine, cpu-only
        #[arg(long, default_value = "all", value_enum)]
        chunk_emb_compute_units: ChunkEmbeddingComputeUnits,
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
        /// Use the default ORT session config
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

impl Command {
    fn run(self) -> Result<()> {
        match self {
            Self::Models { cmd } => cmd.run(),
            Self::Fixtures { cmd } => cmd.run(),
            Self::Compare { cmd } => cmd.run(),
            Self::Benchmark { cmd } => cmd.run(),
            Self::MacExperiment { cmd } => cmd.run(),
            Self::Dstack { cmd } => cmd.run(),
            Self::Dataset { cmd } => cmd.run(),
            Self::Diarize {
                mode,
                models_dir,
                chunk_emb_compute_units,
                wav_files,
            } => commands::diarize::run(mode, models_dir, chunk_emb_compute_units, wav_files),
            Self::ProfileOrtEmbedding {
                mode,
                wav_path,
                iterations,
                log_every,
                model_path,
                batch_size,
                ort_defaults,
            } => commands::profile_ort_embedding::run(
                crate::counts::parse_ort_embedding_mode(&mode)?,
                &wav_path.to_string_lossy(),
                iterations,
                log_every,
                model_path,
                batch_size
                    .map(|value| crate::counts::nonzero_usize("batch-size", value))
                    .transpose()?
                    .map(|value| value.get()),
                ort_defaults,
            ),
            Self::ProfileStages {
                mode,
                wav_path,
                iterations,
                log_every,
            } => commands::profile_stages::run(
                crate::counts::parse_stage_mode(&mode)?,
                &wav_path.to_string_lossy(),
                iterations,
                log_every,
            ),
        }
    }
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Export ONNX models and PLDA params, then build CoreML bundles on macOS
    Export,
    /// Run CoreML model conversion only
    ExportCoreml,
    /// Compare CoreML and ONNX outputs
    CompareCoreml,
    /// Upload models to HuggingFace Hub
    Deploy,
}

impl ModelsCmd {
    fn run(self) -> Result<()> {
        match self {
            Self::Export => commands::models::export(),
            Self::ExportCoreml => commands::models::export_coreml(),
            Self::CompareCoreml => commands::models::compare_coreml(),
            Self::Deploy => commands::models::deploy(),
        }
    }
}

#[derive(Subcommand)]
enum FixturesCmd {
    /// Regenerate test fixtures via Python
    Generate,
}

impl FixturesCmd {
    fn run(self) -> Result<()> {
        match self {
            Self::Generate => commands::fixtures::generate(),
        }
    }
}

#[derive(Subcommand)]
enum CompareCmd {
    /// Informal RTTM timeline overlap (not speaker-assigned DER)
    Rttm { a: PathBuf, b: PathBuf },
}

impl CompareCmd {
    fn run(self) -> Result<()> {
        match self {
            Self::Rttm { a, b } => commands::compare::rttm(&a, &b),
        }
    }
}

#[derive(Subcommand)]
enum BenchmarkCmd {
    /// Measure implementations and write a schema version 3 run
    Run {
        /// Dataset to evaluate ("all" for all datasets, "list" to show available)
        #[arg(long, default_value = "voxconverse-dev")]
        dataset: String,
        /// Single WAV file to evaluate
        #[arg(long, requires = "rttm", conflicts_with_all = ["dataset", "max_files", "max_minutes"])]
        file: Option<PathBuf>,
        /// Reference RTTM file (required when --file is used)
        #[arg(long, requires = "file")]
        rttm: Option<PathBuf>,
        #[arg(long)]
        max_files: Option<u32>,
        #[arg(long)]
        max_minutes: Option<u32>,
        /// Short note for this benchmark run
        #[arg(long, short = 'd')]
        description: Option<String>,
        /// Implementations to run (omit for all, use "list" to show available)
        #[arg(long, value_delimiter = ',', value_name = "IMPL")]
        impls: Vec<String>,
        /// Skip the preflight smoke test
        #[arg(long)]
        no_preflight: bool,
        /// Override pyannote segmentation batch size
        #[arg(long)]
        seg_batch_size: Option<u32>,
        /// Override pyannote embedding batch size
        #[arg(long)]
        emb_batch_size: Option<u32>,
        /// Seconds to sleep between implementations
        #[arg(long, short = 's')]
        sleep_between: Option<u64>,
    },
    /// Score a stored schema version 3 run with authoritative DER
    Score {
        /// Directory created by `benchmark run`
        run_dir: PathBuf,
    },
}

impl BenchmarkCmd {
    fn run(self) -> Result<()> {
        match self {
            Self::Run {
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
            Self::Score { run_dir } => commands::benchmark::score(&run_dir),
        }
    }
}

#[derive(Subcommand)]
enum DstackCmd {
    /// Run a GPU benchmark
    Bench {
        /// Run name
        name: String,
        #[arg(long, default_value = "voxconverse-dev")]
        dataset: String,
        #[arg(long, value_delimiter = ',')]
        impls: Vec<String>,
        #[arg(long)]
        max_files: Option<u32>,
        #[arg(long)]
        max_minutes: Option<u32>,
        /// Reuse an existing fleet pod
        #[arg(long, short = 'R')]
        reuse: bool,
        /// Submit and exit immediately
        #[arg(long, short = 'd')]
        detach: bool,
    },
    /// Run GPU benchmarks in parallel
    #[command(alias = "bp")]
    BenchParallel {
        /// Run name prefix
        name: String,
        /// Datasets to run (comma-separated or "all")
        #[arg(long, value_delimiter = ',', default_value = "all")]
        dataset: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        impls: Vec<String>,
        #[arg(long)]
        max_files: Option<u32>,
        #[arg(long)]
        max_minutes: Option<u32>,
        /// Reuse an existing fleet pod
        #[arg(long, short = 'R')]
        reuse: bool,
    },
    /// Start a reusable GPU fleet
    Fleet,
    /// Reattach to a running task
    Attach { name: String },
    /// Stream logs from a running task
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
    /// Delete a path from the S3 bucket
    Delete {
        /// S3 path to delete
        path: String,
    },
}

impl DstackCmd {
    fn run(self) -> Result<()> {
        match self {
            Self::Bench {
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
            Self::BenchParallel {
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
            Self::Fleet => commands::dstack::fleet(),
            Self::Attach { name } => commands::dstack::attach(&name),
            Self::Logs { name } => commands::dstack::logs(&name),
            Self::Ps => commands::dstack::ps(),
            Self::Stop { name } => commands::dstack::stop(&name),
            Self::Dev => commands::dstack::dev(),
            Self::Download { name } => commands::dstack::download(&name),
            Self::Delete { path } => commands::dstack::delete(&path),
        }
    }
}

#[derive(Subcommand)]
enum DatasetCmd {
    /// Download one or all datasets
    Ensure {
        /// Dataset id, or "all"
        #[arg(default_value = "all")]
        id: String,
        /// Directory that contains dataset subdirectories
        #[arg(long, env = "SPEAKRS_DATASETS_DIR")]
        datasets_dir: Option<PathBuf>,
    },
    /// Upload local datasets to Tigris S3
    Upload {
        /// Dataset id, or "all"
        #[arg(default_value = "all")]
        id: String,
        /// Directory that contains dataset subdirectories
        #[arg(long, env = "SPEAKRS_DATASETS_DIR")]
        datasets_dir: Option<PathBuf>,
    },
}

impl DatasetCmd {
    fn run(self) -> Result<()> {
        use crate::cmd::project_root;
        use crate::datasets::{self, S5cmd};

        match self {
            Self::Ensure { id, datasets_dir } => {
                let base_dir =
                    datasets_dir.unwrap_or_else(|| project_root().join("fixtures/datasets"));
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
            Self::Upload { id, datasets_dir } => {
                let base_dir =
                    datasets_dir.unwrap_or_else(|| project_root().join("fixtures/datasets"));
                if !S5cmd::available() {
                    color_eyre::eyre::bail!("s5cmd not available or AWS_ACCESS_KEY_ID not set");
                }

                let targets = if id == "all" {
                    datasets::all_datasets()
                        .into_iter()
                        .filter(|dataset| dataset.id != "voxconverse-dev")
                        .collect()
                } else {
                    vec![
                        datasets::find_dataset(&id)
                            .ok_or_else(|| color_eyre::eyre::eyre!("unknown dataset: {id}"))?,
                    ]
                };

                for ds in &targets {
                    if let Err(error) = ds.snapshot(&base_dir) {
                        println!("Skipping {} (not a completed installation: {error})", ds.id);
                        continue;
                    }

                    println!("Uploading {}...", ds.id);
                    S5cmd::upload(&ds.id, &ds.dataset_dir(&base_dir))?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn parses_benchmark_run_and_score() {
        assert!(
            Cli::try_parse_from(["xtask", "benchmark", "run", "--dataset", "voxconverse-dev"])
                .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "xtask",
                "benchmark",
                "run",
                "--dataset",
                "voxconverse-dev",
                "--impls",
                "pmps,scm,scmf,sk"
            ])
            .is_ok()
        );
        assert!(Cli::try_parse_from(["xtask", "benchmark", "run", "--impls", "list"]).is_ok());
        assert!(Cli::try_parse_from(["xtask", "benchmark", "run", "--dataset", "list"]).is_ok());
        assert!(Cli::try_parse_from(["xtask", "benchmark", "score", "/tmp/run"]).is_ok());
        assert!(Cli::try_parse_from(["xtask", "compare", "rttm", "a.rttm", "b.rttm"]).is_ok());
    }

    #[test]
    fn rejects_removed_commands() {
        assert!(Cli::try_parse_from(["xtask", "compare", "run", "a.wav"]).is_err());
        assert!(Cli::try_parse_from(["xtask", "compare", "accuracy", "a.wav"]).is_err());
        assert!(Cli::try_parse_from(["xtask", "bench", "run", "a.wav"]).is_err());
        assert!(Cli::try_parse_from(["xtask", "bench", "compare", "a.wav"]).is_err());
        assert!(Cli::try_parse_from(["xtask", "bench", "der"]).is_err());
        assert!(Cli::try_parse_from(["xtask", "benchmark", "der"]).is_err());
    }
}
