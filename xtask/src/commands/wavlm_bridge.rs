mod cache;
mod domain;
mod embedding_execution;
mod report;
mod run;
mod stage;

use std::path::PathBuf;

use clap::Subcommand;
use color_eyre::eyre::Result;

pub use domain::BridgeMode;

/// Standalone WavLM-to-Speakrs bridge commands
#[derive(Subcommand)]
pub enum WavlmBridgeCommand {
    /// Validate one imported segmentation bundle against canonical audio
    Validate {
        /// Published imported segmentation bundle directory
        #[arg(long)]
        bundle: PathBuf,
        /// Canonical 16 kHz mono PCM WAV
        #[arg(long)]
        audio: PathBuf,
        /// Optional immutable validation JSON output
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run named downstream recipes through the imported library path
    Run {
        /// Versioned bridge specification
        #[arg(long)]
        spec: PathBuf,
        /// Directory containing the embedding and PLDA assets
        #[arg(long)]
        models_dir: PathBuf,
        /// Imported embedding execution mode
        #[arg(long, value_enum)]
        mode: BridgeMode,
        /// New immutable run output directory
        #[arg(long)]
        output_dir: PathBuf,
        /// Optional immutable cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Recipe ID to run; repeat for a subset, or omit for all recipes
        #[arg(long = "recipe")]
        recipes: Vec<String>,
    },
    /// Join unchanged Speakrs, frozen Python WavLM, and hybrid system documents
    Report {
        /// Versioned bridge specification
        #[arg(long)]
        spec: PathBuf,
        /// Unchanged Speakrs system document
        #[arg(long = "unchanged-speakrs")]
        unchanged_speakrs: PathBuf,
        /// Frozen Python WavLM system document
        #[arg(long = "frozen-python")]
        frozen_python: PathBuf,
        /// Hybrid WavLM plus Speakrs system document
        #[arg(long)]
        hybrid: PathBuf,
        /// New immutable report output directory
        #[arg(long)]
        output_dir: PathBuf,
    },
}

impl WavlmBridgeCommand {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Validate {
                bundle,
                audio,
                output,
            } => {
                let document = run::validate_bundle_audio(&bundle, &audio)?;
                let bytes = serde_json::to_vec_pretty(&document)?;
                if let Some(output) = output {
                    write_new(&output, &bytes)?;
                    println!("wrote validation {}", output.display());
                } else {
                    println!("{}", String::from_utf8(bytes)?);
                }
                Ok(())
            }
            Self::Run {
                spec,
                models_dir,
                mode,
                output_dir,
                cache_dir,
                recipes,
            } => run::run(run::RunOptions {
                spec_path: spec,
                models_dir,
                mode,
                output_dir,
                cache_dir,
                recipe_ids: recipes,
            }),
            Self::Report {
                spec,
                unchanged_speakrs,
                frozen_python,
                hybrid,
                output_dir,
            } => report::run(report::ReportOptions {
                spec_path: spec,
                unchanged_speakrs,
                frozen_python,
                hybrid,
                output_dir,
            }),
        }
    }
}

fn write_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::fs::{self, OpenOptions};
    use std::io::Write;

    if path.exists() {
        color_eyre::eyre::bail!("output already exists: {}", path.display());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
