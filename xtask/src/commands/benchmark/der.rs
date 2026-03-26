use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use color_eyre::eyre::{Result, bail};

use super::*;
use preflight::preflight_check;
use run::{DerRunContext, run_der_implementations};
use validate::{
    der_build_features, handle_list_requests, resolve_eval_datasets, validate_impls,
    validate_single_file_mode,
};

mod preflight;
pub(super) mod run;
mod validate;

pub(super) const IMPL_REGISTRY: &[(&str, &str, &str, ImplType)] = &[
    (
        "pyannote",
        "pmps",
        "pyannote MPS",
        ImplType::Pyannote("mps"),
    ),
    (
        "pyannote-cpu",
        "pcpu",
        "pyannote CPU",
        ImplType::Pyannote("cpu"),
    ),
    (
        "pyannote-cuda",
        "pg",
        "pyannote CUDA",
        ImplType::Pyannote("cuda"),
    ),
    (
        "coreml",
        "scm",
        "speakrs CoreML",
        ImplType::Speakrs("coreml"),
    ),
    (
        "coreml-fast",
        "scmf",
        "speakrs CoreML Fast",
        ImplType::Speakrs("coreml-fast"),
    ),
    ("cuda", "sg", "speakrs CUDA", ImplType::Speakrs("cuda")),
    (
        "cuda-fast",
        "sgf",
        "speakrs CUDA Fast",
        ImplType::Speakrs("cuda-fast"),
    ),
    ("cpu", "scpu", "speakrs CPU", ImplType::Speakrs("cpu")),
    ("fluidaudio", "fa", "FluidAudio", ImplType::FluidAudioBench),
    ("speakerkit", "sk", "SpeakerKit", ImplType::SpeakerKitBench),
    ("pyannote-rs", "prs", "pyannote-rs", ImplType::PyannoteRs),
];

pub struct DerArgs {
    pub dataset_id: String,
    pub file: Option<PathBuf>,
    pub rttm: Option<PathBuf>,
    pub max_files: u32,
    pub max_minutes: u32,
    pub description: Option<String>,
    pub impls: Vec<String>,
    pub no_preflight: bool,
    pub seg_batch_size: Option<u32>,
    pub emb_batch_size: Option<u32>,
    pub sleep_between: Option<u64>,
}

pub(crate) fn ensure_pyannote_rs_emb_model(path: &Path) -> Result<()> {
    run::ensure_pyannote_rs_emb_model(path)
}

pub fn der(args: DerArgs) -> Result<()> {
    let DerArgs {
        ref dataset_id,
        ref file,
        ref rttm,
        max_files,
        max_minutes,
        ref description,
        ref impls,
        no_preflight,
        seg_batch_size,
        emb_batch_size,
        sleep_between,
    } = args;
    let pyannote_batch_sizes = PyannoteBatchSizes::from_overrides(seg_batch_size, emb_batch_size);

    if handle_list_requests(&DerArgs {
        dataset_id: dataset_id.clone(),
        file: file.clone(),
        rttm: rttm.clone(),
        max_files,
        max_minutes,
        description: description.clone(),
        impls: impls.clone(),
        no_preflight,
        seg_batch_size,
        emb_batch_size,
        sleep_between,
    })? {
        return Ok(());
    }

    validate_impls(impls)?;

    let single_file_mode = validate_single_file_mode(file, rttm)?;
    let datasets = resolve_eval_datasets(dataset_id, single_file_mode)?;

    let root = project_root();

    println!("=== Building binaries ===");
    let build_features = der_build_features(impls);
    cargo_build_xtask(&build_features)?;

    let needs_pyannote_rs =
        impls.is_empty() || impls.iter().any(|impl_id| impl_id == "pyannote-rs");
    if needs_pyannote_rs
        && let Err(err) = run_cmd(
            Command::new("cargo")
                .args(["build", "--release"])
                .current_dir(root.join("scripts/pyannote_rs_bench")),
        )
    {
        eprintln!("warning: pyannote-rs bench build failed (skipping): {err}");
    }

    let models_dir = root.join("fixtures/models");
    let seg_model = models_dir.join("segmentation-3.0.onnx");
    let emb_model = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    if needs_pyannote_rs {
        ensure_pyannote_rs_emb_model(&emb_model)?;
    }

    let metadata = BenchmarkMetadata::collect();

    let eval_sets: Vec<(String, Vec<(PathBuf, PathBuf)>)> = if single_file_mode {
        let (wav_path, rttm_path) = match (file.clone(), rttm.clone()) {
            (Some(wav_path), Some(rttm_path)) => (wav_path, rttm_path),
            _ => bail!("single-file mode requires both --file and --rttm"),
        };
        let display_name = wav_path
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_else(|| "single-file".to_string());
        vec![(display_name, vec![(wav_path, rttm_path)])]
    } else {
        let fixtures_dir = root.join("fixtures/datasets");
        let mut sets = Vec::new();
        for dataset in &datasets {
            dataset.ensure(&fixtures_dir)?;
            let dataset_dir = dataset.dataset_dir(&fixtures_dir);
            let files = discover_files(&dataset_dir, max_files, max_minutes as f64)?;
            if files.is_empty() {
                eprintln!(
                    "No paired wav+rttm files found in {}",
                    dataset_dir.display()
                );
                continue;
            }
            sets.push((dataset.display_name.clone(), files));
        }
        sets
    };

    let preflight_failures: HashMap<String, String> = if no_preflight || eval_sets.is_empty() {
        HashMap::new()
    } else {
        let first_file = eval_sets[0]
            .1
            .iter()
            .min_by(|a, b| {
                wav_duration_seconds(&a.0)
                    .unwrap_or(f64::MAX)
                    .total_cmp(&wav_duration_seconds(&b.0).unwrap_or(f64::MAX))
            })
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("preflight requires at least one discovered audio file")
            })?;
        preflight_check(
            &root,
            first_file,
            &models_dir,
            &seg_model,
            &emb_model,
            &DerArgs {
                dataset_id: dataset_id.clone(),
                file: file.clone(),
                rttm: rttm.clone(),
                max_files,
                max_minutes,
                description: description.clone(),
                impls: impls.clone(),
                no_preflight,
                seg_batch_size,
                emb_batch_size,
                sleep_between,
            },
            pyannote_batch_sizes,
        )?
    };

    for (dataset_name, files) in &eval_sets {
        println!();
        println!("========== {dataset_name} ==========");

        let total_audio_seconds: f64 = files
            .iter()
            .map(|(wav, _)| wav_duration_seconds(wav).unwrap_or(0.0))
            .sum();
        let total_audio_minutes = total_audio_seconds / 60.0;

        let run_id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let run_dir = if eval_sets.len() > 1 {
            root.join("_benchmarks")
                .join(&run_id)
                .join(dataset_name.to_lowercase().replace(' ', "-"))
        } else {
            root.join("_benchmarks").join(&run_id)
        };
        fs::create_dir_all(&run_dir)?;

        if let Some(desc) = description.as_deref() {
            fs::write(run_dir.join("README.md"), format!("{desc}\n"))?;
        }

        println!(
            "Found {} files, {total_audio_minutes:.1} min total audio",
            files.len()
        );
        println!("Run ID: {run_id}");
        println!();

        let (implementations, all_results) = run_der_implementations(&DerRunContext {
            root: &root,
            run_dir: &run_dir,
            files,
            models_dir: &models_dir,
            seg_model: &seg_model,
            emb_model: &emb_model,
            impls,
            total_audio_seconds,
            preflight_failures: &preflight_failures,
            sleep_between: sleep_between.map(Duration::from_secs),
            pyannote_batch_sizes,
        })?;

        DerResultsWriter {
            run_dir: &run_dir,
            dataset_name,
            implementations: &implementations,
            results: &all_results,
            files,
            total_audio_minutes,
            collar: 0.0,
            description: description.as_deref(),
            max_files,
            max_minutes,
            metadata: &metadata,
            pyannote_batch_sizes,
        }
        .write()?;
    }

    Ok(())
}
