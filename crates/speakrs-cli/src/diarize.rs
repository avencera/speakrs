//! `speakrs diarize`: decode audio files, run the pipeline and print RTTM

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Args;
use eyre::{WrapErr, bail};
use speakrs::ExecutionMode;
use speakrs_sdk::{ModelStore, PrepareModelsOptions, RunOptions, SdkPipeline, parse_mode};

/// Diarize audio files and print RTTM to stdout
///
/// Several files run as one overlapped batch
#[derive(Args)]
pub(crate) struct DiarizeArgs {
    /// Audio files (WAV, FLAC, MP3, Ogg, M4A)
    #[arg(required = true)]
    audio: Vec<PathBuf>,

    /// Execution mode
    #[arg(long, default_value = "cpu", help = mode_help())]
    mode: String,

    /// Load models from this directory instead of the download cache
    #[arg(long, value_name = "DIR")]
    model_dir: Option<PathBuf>,

    /// Model download cache (default: the Hugging Face hub cache)
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,

    /// RTTM file id for a single input (default: file name without extension)
    #[arg(long, value_name = "ID")]
    file_id: Option<String>,

    /// Print exclusive segments, with at most one speaker at a time
    #[arg(long)]
    exclusive: bool,
}

pub(crate) fn run(args: DiarizeArgs) -> eyre::Result<()> {
    if let Some(file_id) = &args.file_id {
        if args.audio.len() != 1 {
            bail!("--file-id needs exactly one audio file");
        }
        // RTTM separates fields with spaces, so an id with whitespace shifts the columns
        if file_id.is_empty() || file_id.chars().any(char::is_whitespace) {
            bail!("--file-id must be non-empty and contain no whitespace, got `{file_id}`");
        }
    }

    // a batch names each file by its stem, so two with the same stem, after whitespace
    // becomes `_`, would print RTTM lines that can't be told apart
    let mut file_ids = HashSet::new();
    if args.audio.len() > 1
        && let Some(path) = args
            .audio
            .iter()
            .find(|path| !file_ids.insert(file_stem(path)))
    {
        bail!(
            "{} has the same name, without its extension, as another input, so their RTTM file ids would match; run it separately with --file-id",
            path.display()
        );
    }

    let mode = parse_mode(&args.mode)?;
    let modes = available_modes();
    // checked before preparing models, so an unsupported mode never downloads them
    if !modes.contains(&mode) {
        bail!(
            "this build does not support mode `{mode}`; it supports {}",
            mode_list(&modes)
        );
    }

    let store = args
        .cache_dir
        .map_or_else(ModelStore::default, ModelStore::new);
    let mut options = PrepareModelsOptions::new(mode);
    options.model_dir = args.model_dir;
    let prepared = store
        .prepare(options, &|file| eprintln!("downloading {file}"))
        .wrap_err("failed to prepare models")?;
    let mut pipeline = SdkPipeline::from_prepared(&prepared, mode, None, None)?;

    let results = match args.audio.as_slice() {
        [path] => {
            let file_id = args.file_id.unwrap_or_else(|| file_stem(path));
            let result = pipeline
                .diarize_file(path, &file_id, RunOptions::default())
                .wrap_err_with(|| format!("failed to diarize {}", path.display()))?;
            vec![result]
        }
        paths => {
            let files = paths
                .iter()
                .map(|path| (file_stem(path), path.clone()))
                .collect();
            pipeline.diarize_files(files, RunOptions::default())?
        }
    };

    let mut stdout = std::io::stdout().lock();
    for result in &results {
        stdout.write_all(result.rttm(None, args.exclusive).as_bytes())?;
    }
    stdout.flush()?;
    Ok(())
}

/// Modes this build can run
fn available_modes() -> Vec<ExecutionMode> {
    let mut modes = speakrs_sdk::available_modes();
    // a GPU-specific build enables CUDA in speakrs without the SDK's `cuda` feature
    if cfg!(feature = "_cuda") && !modes.contains(&ExecutionMode::Cuda) {
        modes.extend([ExecutionMode::Cuda, ExecutionMode::CudaFast]);
    }
    modes
}

fn mode_list(modes: &[ExecutionMode]) -> String {
    modes
        .iter()
        .map(|mode| mode.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn mode_help() -> String {
    format!(
        "Execution mode; this build supports {}",
        mode_list(&available_modes())
    )
}

/// The default RTTM file id: the file name without its extension, like the Python
/// command's `Path.stem`, with each whitespace character replaced by `_`
///
/// RTTM separates fields with spaces, so `team meeting.wav` would otherwise shift
/// every column after the id
fn file_stem(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    stem.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}
