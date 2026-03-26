use color_eyre::eyre::{Result, bail};

use super::super::ImplType;

const GPU_IMPLS: &[(&str, &str, &str, ImplType)] = &[
    ("speakrs", "sg", "speakrs CUDA", ImplType::Speakrs("cuda")),
    (
        "speakrs-fast",
        "sgf",
        "speakrs CUDA Fast",
        ImplType::Speakrs("cuda-fast"),
    ),
    (
        "pyannote",
        "pg",
        "pyannote CUDA",
        ImplType::Pyannote("cuda"),
    ),
];

pub(super) fn resolve_gpu_impls(impls: &[String]) -> Vec<(&'static str, ImplType)> {
    if impls.is_empty() {
        GPU_IMPLS
            .iter()
            .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    } else {
        GPU_IMPLS
            .iter()
            .filter(|(cli_id, alias, _, _)| {
                impls.iter().any(|value| value == cli_id || value == alias)
            })
            .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    }
}

fn resolve_gpu_impl(name: &str) -> Option<usize> {
    GPU_IMPLS
        .iter()
        .position(|(cli_id, alias, _, _)| *cli_id == name || *alias == name)
}

pub fn gpu_impls() -> &'static [(&'static str, &'static str, &'static str, ImplType)] {
    GPU_IMPLS
}

pub fn validate_gpu_impls(impls: &[String]) -> Result<()> {
    for id in impls {
        if resolve_gpu_impl(id).is_none() {
            let available: Vec<String> = GPU_IMPLS
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
