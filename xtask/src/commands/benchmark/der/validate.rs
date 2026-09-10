use std::path::PathBuf;

use color_eyre::eyre::{Result, bail};

use super::DerArgs;
use crate::catalog::{ImplementationCatalog, ImplementationId, ImplementationSpec, RunnerKind};
use crate::commands::benchmark::ImplType;

fn to_impl_type(spec: &ImplementationSpec) -> ImplType {
    match spec.runner {
        RunnerKind::Speakrs(mode) => ImplType::Speakrs(mode.as_cli()),
        RunnerKind::Pyannote(device) => ImplType::Pyannote(device.as_cli()),
        RunnerKind::PyannoteRs => ImplType::PyannoteRs,
        RunnerKind::FluidAudio => ImplType::FluidAudioBench,
        RunnerKind::SpeakerKit => ImplType::SpeakerKitBench,
    }
}

pub(super) fn handle_list_requests(args: &DerArgs) -> Result<bool> {
    if args.impls.len() == 1 && args.impls[0] == "list" {
        println!("Available implementations:");
        for spec in ImplementationCatalog::all() {
            let alias = spec.aliases.first().copied().unwrap_or("");
            println!("  {alias:<4} {:<15} {}", spec.cli_name(), spec.display_name);
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
    ImplementationCatalog::resolve_many(impls).map(|_| ())
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

pub(super) fn selected_implementations(impls: &[String]) -> Vec<(ImplementationId, ImplType)> {
    ImplementationCatalog::resolve_many(impls)
        .unwrap_or_default()
        .into_iter()
        .map(|spec| (spec.id, to_impl_type(spec)))
        .collect()
}

pub(super) fn selected_preflight_implementations(
    impls: &[String],
) -> Vec<(ImplementationId, &'static str, ImplType)> {
    ImplementationCatalog::resolve_many(impls)
        .unwrap_or_default()
        .into_iter()
        .map(|spec| (spec.id, spec.display_name, to_impl_type(spec)))
        .collect()
}

pub(super) fn der_build_features(impls: &[String]) -> Vec<String> {
    let selected = ImplementationCatalog::resolve_many(impls).unwrap_or_default();
    ImplementationCatalog::cargo_features(&selected)
}
