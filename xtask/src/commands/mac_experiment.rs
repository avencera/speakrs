use std::path::PathBuf;

use clap::Subcommand;
use color_eyre::eyre::Result;

#[cfg(all(target_os = "macos", feature = "coreml"))]
mod batch_proof;
#[cfg(target_os = "macos")]
mod compute_plan;
mod domain;
mod execute;
mod identity;
#[cfg(all(target_os = "macos", feature = "coreml"))]
mod inference_comparison;
mod statistics;
mod store;
mod trace_analysis;

pub(crate) use domain::{MacExperimentSpec, ValidatedExperiment};

#[derive(Subcommand)]
pub enum MacExperimentCommand {
    /// Validate a versioned experiment specification without running inference
    Validate {
        /// JSON experiment specification
        #[arg(long)]
        spec: PathBuf,
    },
    /// Create and execute a new experiment run
    Run {
        /// JSON experiment specification
        #[arg(long)]
        spec: PathBuf,
    },
    /// Continue an experiment from its durable records
    Resume {
        /// Existing `_benchmarks/macos/<experiment-id>` directory
        run_dir: PathBuf,
    },
    /// Rebuild deterministic projections from durable records
    Summarize {
        /// Existing `_benchmarks/macos/<experiment-id>` directory
        run_dir: PathBuf,
    },
    /// Capture an admitted Instruments profile for an experiment workload
    Profile {
        /// JSON experiment specification
        #[arg(long)]
        spec: PathBuf,
    },
    /// Compare matching chunk and per-window embeddings at one segmentation step
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    CompareInference {
        /// Directory that contains the compiled CoreML models
        #[arg(long)]
        models_dir: PathBuf,
        /// 16 kHz mono PCM WAV input
        #[arg(long)]
        audio: PathBuf,
        /// Segmentation step pair to compare
        #[arg(long, value_enum)]
        step: InferenceComparisonStep,
        /// Durable JSON result path
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare current batch execution with repeated single-file execution
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    BatchProof {
        /// Directory that contains the compiled CoreML models
        #[arg(long)]
        models_dir: PathBuf,
        /// Input WAV files; at least two are required
        #[arg(long, num_args = 2..)]
        audio: Vec<PathBuf>,
        /// Prefix duration to use from each file
        #[arg(long, default_value_t = 120.0)]
        duration_seconds: f64,
        /// Measured repetitions per condition; must be even and at least six
        #[arg(long, default_value_t = 6)]
        repetitions: usize,
        /// Durable JSON result path
        #[arg(long)]
        output: PathBuf,
    },
    /// Internal isolated worker used to measure one repetition
    #[command(hide = true)]
    Worker {
        /// Existing run directory
        run_dir: PathBuf,
        /// Zero-based repetition index
        #[arg(long)]
        repetition: u32,
    },
    /// Internal worker used as an Instruments launch target
    #[command(hide = true)]
    ProfileWorker {
        /// Existing run directory
        run_dir: PathBuf,
        /// File index from the immutable run manifest
        #[arg(long)]
        file_index: usize,
        /// Structured output from the profiled workload
        #[arg(long)]
        stage_output: PathBuf,
    },
    /// Internal baseline worker used by A-B-B-A comparisons
    #[command(hide = true)]
    ComparisonWorker {
        /// Completed source baseline run
        baseline_run_dir: PathBuf,
        /// Candidate run that owns the comparison records
        candidate_run_dir: PathBuf,
        /// Zero-based comparison repetition
        #[arg(long)]
        repetition: u32,
    },
}

impl MacExperimentCommand {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Validate { spec } => validate(&spec),
            Self::Run { spec } => run_new(&spec),
            Self::Resume { run_dir } => resume(&run_dir),
            Self::Summarize { run_dir } => summarize(&run_dir),
            Self::Profile { spec } => profile(&spec),
            #[cfg(all(target_os = "macos", feature = "coreml"))]
            Self::CompareInference {
                models_dir,
                audio,
                step,
                output,
            } => inference_comparison::run(&models_dir, &audio, step, &output),
            #[cfg(all(target_os = "macos", feature = "coreml"))]
            Self::BatchProof {
                models_dir,
                audio,
                duration_seconds,
                repetitions,
                output,
            } => batch_proof::run(&models_dir, &audio, duration_seconds, repetitions, &output),
            Self::Worker {
                run_dir,
                repetition,
            } => execute::run_worker(&run_dir, repetition),
            Self::ProfileWorker {
                run_dir,
                file_index,
                stage_output,
            } => execute::run_profile_worker(&run_dir, file_index, &stage_output),
            Self::ComparisonWorker {
                baseline_run_dir,
                candidate_run_dir,
                repetition,
            } => execute::run_comparison_worker(&baseline_run_dir, &candidate_run_dir, repetition),
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
#[derive(Clone, Copy, clap::ValueEnum)]
pub enum InferenceComparisonStep {
    /// Exact 1.0-second phased chunks against 1.0-second per-window inference
    OneSecond,
    /// Chunk-wide against stitched 10-second filterbank normalization
    NormalizationScope,
}

fn validate(spec_path: &std::path::Path) -> Result<()> {
    let experiment = ValidatedExperiment::load(spec_path)?;
    println!(
        "valid macOS experiment '{}' ({} post-inference candidate(s), {} repetition(s))",
        experiment.id(),
        experiment.post_inference().len(),
        experiment.performance().repetitions()
    );
    Ok(())
}

fn run_new(spec_path: &std::path::Path) -> Result<()> {
    let experiment = ValidatedExperiment::load(spec_path)?;
    experiment.ensure_runnable()?;
    let worker = execute::build_worker_binary()?;
    let store = store::RunStore::create(&experiment, &worker)?;
    execute::run_managed_with_worker(&worker, &store, &experiment)
}

fn resume(run_dir: &std::path::Path) -> Result<()> {
    let (store, experiment) = store::RunStore::open(run_dir)?;
    experiment.ensure_runnable()?;
    execute::run_managed(&store, &experiment)
}

fn summarize(run_dir: &std::path::Path) -> Result<()> {
    let (store, experiment) = store::RunStore::open(run_dir)?;
    store.rebuild_projections(&experiment)
}

fn profile(spec_path: &std::path::Path) -> Result<()> {
    let experiment = ValidatedExperiment::load(spec_path)?;
    experiment.ensure_runnable()?;
    let worker = execute::build_worker_binary()?;
    store::profile(experiment, &worker)
}
