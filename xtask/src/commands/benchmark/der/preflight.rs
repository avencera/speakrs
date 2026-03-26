use std::collections::HashMap;
use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;

use super::run::DerBenchEnv;
use super::validate::selected_preflight_implementations;
use super::{DerArgs, PREFLIGHT_TIMEOUT, PyannoteBatchSizes};
use crate::cmd::wav_duration_seconds;
use crate::path::file_stem_string;

pub(super) fn preflight_check(
    root: &Path,
    file: &(PathBuf, PathBuf),
    models_dir: &Path,
    seg_model: &Path,
    emb_model: &Path,
    args: &DerArgs,
    pyannote_batch_sizes: PyannoteBatchSizes,
) -> Result<HashMap<String, String>> {
    let (wav_path, _) = file;
    let stem = file_stem_string(wav_path)?;
    let duration = wav_duration_seconds(wav_path).unwrap_or(0.0);
    println!();
    println!("=== Pre-flight check ({stem}, {duration:.0}s) ===");

    let env = DerBenchEnv::new(root, models_dir, seg_model, emb_model, pyannote_batch_sizes);
    let implementations = selected_preflight_implementations(&args.impls);
    let wav_paths = [wav_path.as_path()];
    let mut failures = HashMap::new();

    for (_cli_id, display_name, impl_type) in implementations {
        if let Some(reason) = env.skip_reason(&impl_type) {
            println!("  {display_name:<22} skipped ({reason})");
            continue;
        }

        match env.run_impl(
            &impl_type,
            &wav_paths,
            &[(wav_path.clone(), PathBuf::new())],
            PREFLIGHT_TIMEOUT,
        ) {
            Ok(batch_output)
                if batch_output
                    .per_file_rttm
                    .values()
                    .any(|value| !value.trim().is_empty()) =>
            {
                println!(
                    "  {display_name:<22} ok ({:.1}s)",
                    batch_output.total_seconds
                );
            }
            Ok(_) => {
                let reason = "empty RTTM output".to_string();
                println!("  {display_name:<22} FAILED: {reason}");
                failures.insert(display_name.to_string(), reason);
            }
            Err(err) => {
                let reason = err.to_string();
                println!("  {display_name:<22} FAILED: {reason}");
                failures.insert(display_name.to_string(), reason);
            }
        }
    }

    if !failures.is_empty() {
        let names: Vec<&str> = failures.keys().map(String::as_str).collect();
        println!();
        println!(
            "Skipping failed implementations for remaining datasets: {}",
            names.join(", ")
        );
    }

    println!();
    Ok(failures)
}
