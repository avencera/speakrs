use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use color_eyre::eyre::{Result, bail};

use crate::cargo::cargo_build_xtask;
use crate::catalog::{ImplementationCatalog, RunnerKind};
use crate::cmd::{project_root, run_cmd, wav_duration_seconds};

use super::*;
use preflight::preflight_check;
use run::{DerRunContext, run_der_implementations};
use validate::{handle_list_requests, resolve_eval_datasets, validate_single_file_mode};

mod preflight;
pub(super) mod run;
mod validate;

type EvalSet = (
    super::run_store::DatasetIdentity,
    String,
    Vec<(PathBuf, PathBuf)>,
);

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

    let selected_impls = ImplementationCatalog::resolve_many(impls)?;

    let single_file_mode = validate_single_file_mode(file, rttm)?;
    let datasets = resolve_eval_datasets(dataset_id, single_file_mode)?;

    let root = project_root();

    let single_file_pair = if single_file_mode {
        Some((
            file.clone()
                .ok_or_else(|| color_eyre::eyre::eyre!("single-file mode requires --file"))?,
            rttm.clone()
                .ok_or_else(|| color_eyre::eyre::eyre!("single-file mode requires --rttm"))?,
        ))
    } else {
        None
    };

    let eval_sets: Vec<EvalSet> = if let Some((wav_path, rttm_path)) = &single_file_pair {
        let file_stem = wav_path
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_else(|| "single-file".to_owned());
        let identity = super::run_store::DatasetIdentity::single_file(file_stem.clone())?;
        vec![(
            identity,
            file_stem,
            vec![(wav_path.clone(), rttm_path.clone())],
        )]
    } else {
        let fixtures_dir = root.join("fixtures/datasets");
        let mut sets = Vec::new();
        for dataset in &datasets {
            dataset.ensure(&fixtures_dir)?;
            let snapshot = dataset.verified_snapshot(&fixtures_dir)?;
            let pairs = snapshot
                .files()
                .iter()
                .map(|file| {
                    let duration = file.duration_seconds();
                    (file.wav().to_owned(), file.rttm().to_owned(), duration)
                })
                .collect();
            let files =
                super::selection::select_pairs_for_benchmark(pairs, max_files, max_minutes as f64);
            if files.is_empty() {
                bail!(
                    "dataset {} snapshot produced no files after selection caps",
                    dataset.id
                );
            }
            sets.push((
                super::run_store::DatasetIdentity::catalog(dataset.catalog_id()?),
                dataset.display_name.clone(),
                files,
            ));
        }
        sets
    };

    ensure!(
        !eval_sets.is_empty(),
        "benchmark evaluation set contains no datasets"
    );

    println!("=== Building binaries ===");
    let build_features = ImplementationCatalog::cargo_features(&selected_impls);
    cargo_build_xtask(&build_features)?;

    let needs_pyannote_rs = selected_impls
        .iter()
        .any(|spec| matches!(spec.runner, RunnerKind::PyannoteRs));
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
    let suite_datasets = eval_sets
        .iter()
        .map(|(identity, _, _)| identity.clone())
        .collect();
    let suite = super::BenchmarkRun::create_with_dataset_identities(
        &root.join("_benchmarks"),
        selected_impls.iter().map(|spec| spec.id).collect(),
        suite_datasets,
        description.clone(),
        chrono::Local::now(),
        metadata.cpu.clone(),
    )?;

    let preflight_failures = if no_preflight || eval_sets.is_empty() {
        HashMap::new()
    } else {
        let first_file = eval_sets[0]
            .2
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
            &selected_impls,
            pyannote_batch_sizes,
        )?
    };

    for (dataset, dataset_name, files) in &eval_sets {
        println!();
        println!("========== {dataset_name} ==========");

        let total_audio_seconds: f64 = files
            .iter()
            .map(|(wav, _)| wav_duration_seconds(wav).unwrap_or(0.0))
            .sum();
        let total_audio_minutes = total_audio_seconds / 60.0;

        let run_id = suite.identity.run_id.as_str().to_owned();
        let run_dir = if eval_sets.len() > 1 {
            let dir = suite.root.join(dataset.id_string());
            fs::create_dir_all(&dir)?;
            dir
        } else {
            suite.root.clone()
        };

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
            files,
            models_dir: &models_dir,
            seg_model: &seg_model,
            emb_model: &emb_model,
            implementations: &selected_impls,
            total_audio_seconds,
            preflight_failures: &preflight_failures,
            sleep_between: sleep_between.map(Duration::from_secs),
            pyannote_batch_sizes,
        })?;

        DerResultsWriter {
            run_dir: &run_dir,
            run_identity: &suite.identity,
            dataset: dataset.clone(),
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
