use std::path::PathBuf;

use color_eyre::eyre::{Result, bail};

use super::{DerArgs, IMPL_REGISTRY};
use crate::commands::benchmark::ImplType;

pub(super) fn resolve_impl(name: &str) -> Option<usize> {
    IMPL_REGISTRY
        .iter()
        .position(|(cli_id, alias, _, _)| *cli_id == name || *alias == name)
}

pub(super) fn handle_list_requests(args: &DerArgs) -> Result<bool> {
    if args.impls.len() == 1 && args.impls[0] == "list" {
        println!("Available implementations:");
        for (cli_id, alias, display_name, _) in IMPL_REGISTRY {
            println!("  {alias:<4} {cli_id:<15} {display_name}");
        }
        return Ok(true);
    }

    if args.dataset_id == "list" && args.file.is_none() {
        println!("Available datasets:");
        for id in crate::datasets::list_dataset_ids() {
            println!("  {id}");
        }
        println!("  all  (run all datasets)");
        return Ok(true);
    }

    Ok(false)
}

pub(super) fn validate_impls(impls: &[String]) -> Result<()> {
    for id in impls {
        if id != "list" && resolve_impl(id).is_none() {
            let available: Vec<String> = IMPL_REGISTRY
                .iter()
                .map(|(cli_id, alias, _, _)| format!("{cli_id} ({alias})"))
                .collect();
            bail!(
                "unknown implementation: {id}. Available: {}",
                available.join(", ")
            );
        }
    }

    Ok(())
}

pub(super) fn validate_single_file_mode(
    file: &Option<PathBuf>,
    rttm: &Option<PathBuf>,
) -> Result<bool> {
    let Some(wav_path) = file else {
        return Ok(false);
    };
    let rttm_path = rttm
        .as_ref()
        .ok_or_else(|| color_eyre::eyre::eyre!("--rttm is required when using --file"))?;

    if !wav_path.exists() {
        bail!("WAV file not found: {}", wav_path.display());
    }
    if !rttm_path.exists() {
        bail!("RTTM file not found: {}", rttm_path.display());
    }

    Ok(true)
}

pub(super) fn resolve_eval_datasets(
    dataset_id: &str,
    single_file_mode: bool,
) -> Result<Vec<crate::datasets::Dataset>> {
    if single_file_mode {
        return Ok(Vec::new());
    }

    if dataset_id == "all" {
        return Ok(crate::datasets::all_datasets());
    }

    Ok(vec![crate::datasets::find_dataset(dataset_id).ok_or_else(
        || {
            color_eyre::eyre::eyre!(
                "unknown dataset: {dataset_id}. Use --dataset list to see available datasets"
            )
        },
    )?])
}

pub(super) fn selected_implementations(impls: &[String]) -> Vec<(&'static str, ImplType)> {
    IMPL_REGISTRY
        .iter()
        .filter(|(cli_id, alias, _, _)| {
            impls.is_empty() || impls.iter().any(|value| value == cli_id || value == alias)
        })
        .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
        .collect()
}

pub(super) fn selected_preflight_implementations(
    impls: &[String],
) -> Vec<(&'static str, &'static str, ImplType)> {
    IMPL_REGISTRY
        .iter()
        .filter(|(cli_id, alias, _, _)| {
            impls.is_empty() || impls.iter().any(|value| value == cli_id || value == alias)
        })
        .map(|(cli_id, _, display_name, impl_type)| (*cli_id, *display_name, *impl_type))
        .collect()
}

pub(super) fn der_build_features(impls: &[String]) -> Vec<String> {
    let active_impls: Vec<&ImplType> = if impls.is_empty() {
        IMPL_REGISTRY.iter().map(|(_, _, _, kind)| kind).collect()
    } else {
        IMPL_REGISTRY
            .iter()
            .filter(|(cli_id, alias, _, _)| {
                impls.iter().any(|value| value == cli_id || value == alias)
            })
            .map(|(_, _, _, kind)| kind)
            .collect()
    };

    let mut features = Vec::new();

    #[cfg(target_os = "macos")]
    let needs_coreml = active_impls
        .iter()
        .any(|kind| matches!(kind, ImplType::Speakrs(mode) if mode.starts_with("coreml")));
    #[cfg(not(target_os = "macos"))]
    let needs_coreml = false;

    let needs_cuda = active_impls
        .iter()
        .any(|kind| matches!(kind, ImplType::Speakrs("cuda" | "cuda-fast")));

    if needs_coreml {
        features.push("coreml".to_string());
    }
    if needs_cuda {
        features.push("cuda".to_string());
    }

    features
}
