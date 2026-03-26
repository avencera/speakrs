use std::path::Path;
use std::time::Duration;

use color_eyre::eyre::{Result, ensure};

use super::super::{BatchCommandRunner, ImplType, PyannoteBatchSizes, discover_files};
use super::run_speakrs_gpu;
use crate::cmd::wav_duration_seconds;
use crate::path::file_stem_string;

pub(super) fn preflight(
    datasets: &[crate::datasets::Dataset],
    datasets_dir: &Path,
    implementations: &[(&str, ImplType)],
    models_dir: &Path,
    root: &Path,
    pyannote_batch_sizes: PyannoteBatchSizes,
) -> Result<()> {
    let first_dataset = &datasets[0];
    first_dataset.ensure(datasets_dir)?;
    let first_dir = first_dataset.dataset_dir(datasets_dir);
    let preflight_files = discover_files(&first_dir, 1, f64::MAX)?;

    if preflight_files.is_empty() {
        eprintln!("warning: no files for preflight, skipping");
        return Ok(());
    }

    let (wav_path, _) = &preflight_files[0];
    let stem = file_stem_string(wav_path)?;
    let duration = wav_duration_seconds(wav_path).unwrap_or(0.0);
    println!();
    println!("=== Pre-flight check ({stem}, {duration:.0}s) ===");

    for (impl_name, impl_type) in implementations {
        let result = match impl_type {
            ImplType::Speakrs(mode) => run_speakrs_gpu(models_dir, &preflight_files, mode, None),
            ImplType::Pyannote(device) => BatchCommandRunner::pyannote(
                root,
                device,
                &[wav_path.as_path()],
                pyannote_batch_sizes,
            )
            .run_with_retries(Duration::from_secs(180)),
            _ => continue,
        };

        let output = result
            .map_err(|err| color_eyre::eyre::eyre!("preflight failed for {impl_name}: {err}"))?;

        ensure!(
            output
                .per_file_rttm
                .values()
                .any(|value| !value.trim().is_empty()),
            "preflight failed for {impl_name}: empty RTTM output"
        );

        println!("  {impl_name:<22} ok ({:.1}s)", output.total_seconds);
    }

    println!();
    Ok(())
}
