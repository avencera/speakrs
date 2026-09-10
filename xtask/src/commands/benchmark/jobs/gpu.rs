use color_eyre::eyre::{Result, bail};

use super::super::ImplType;
use crate::catalog::{ImplementationCatalog, PyannoteDevice, RunnerKind, SpeakrsMode};

pub(super) fn resolve_gpu_impls(
    impls: &[String],
) -> Vec<(crate::catalog::ImplementationId, ImplType)> {
    let selected = if impls.is_empty() {
        ImplementationCatalog::gpu().collect::<Vec<_>>()
    } else {
        ImplementationCatalog::resolve_many(impls)
            .unwrap_or_default()
            .into_iter()
            .filter(|spec| ImplementationCatalog::gpu().any(|gpu| gpu.id == spec.id))
            .collect()
    };
    selected
        .into_iter()
        .map(|spec| {
            let impl_type = match spec.runner {
                RunnerKind::Speakrs(SpeakrsMode::Cuda) => ImplType::Speakrs("cuda"),
                RunnerKind::Speakrs(SpeakrsMode::CudaFast) => ImplType::Speakrs("cuda-fast"),
                RunnerKind::Pyannote(PyannoteDevice::Cuda) => ImplType::Pyannote("cuda"),
                other => unreachable!("gpu catalog entry {other:?}"),
            };
            (spec.id, impl_type)
        })
        .collect()
}

pub fn gpu_impls() -> Vec<(&'static str, &'static str, &'static str)> {
    ImplementationCatalog::gpu()
        .map(|spec| {
            (
                spec.cli_name(),
                spec.aliases.first().copied().unwrap_or(""),
                spec.display_name,
            )
        })
        .collect()
}

pub fn validate_gpu_impls(impls: &[String]) -> Result<()> {
    if impls.is_empty() {
        return Ok(());
    }
    let gpu: Vec<_> = ImplementationCatalog::gpu().collect();
    for id in impls {
        if !gpu
            .iter()
            .any(|spec| spec.cli_name() == id || spec.aliases.contains(&id.as_str()))
        {
            let available: Vec<String> = gpu
                .iter()
                .map(|spec| {
                    format!(
                        "{} ({})",
                        spec.cli_name(),
                        spec.aliases.first().copied().unwrap_or("")
                    )
                })
                .collect();
            bail!(
                "unknown implementation: {id}. Available: {}",
                available.join(", ")
            );
        }
    }
    Ok(())
}
