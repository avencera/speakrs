use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use color_eyre::eyre::{Result, bail};

use super::*;
use crate::path::file_stem_string;

const IMPL_REGISTRY: &[(&str, &str, &str, ImplType)] = &[
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

fn resolve_impl(name: &str) -> Option<usize> {
    IMPL_REGISTRY
        .iter()
        .position(|(cli_id, alias, _, _)| *cli_id == name || *alias == name)
}

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

fn validate_single_file_mode(file: &Option<PathBuf>, rttm: &Option<PathBuf>) -> Result<bool> {
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

fn resolve_eval_datasets(
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

    if impls.len() == 1 && impls[0] == "list" {
        println!("Available implementations:");
        for (cli_id, alias, display_name, _) in IMPL_REGISTRY {
            println!("  {alias:<4} {cli_id:<15} {display_name}");
        }
        return Ok(());
    }

    if !impls.is_empty() {
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
    }

    if dataset_id == "list" && file.is_none() {
        println!("Available datasets:");
        for id in crate::datasets::list_dataset_ids() {
            println!("  {id}");
        }
        println!("  all  (run all datasets)");
        return Ok(());
    }

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
            impls,
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

type DerResults = (
    Vec<(&'static str, ImplType)>,
    HashMap<String, DerImplResult>,
);

struct DerRunContext<'a> {
    root: &'a Path,
    run_dir: &'a Path,
    files: &'a [(PathBuf, PathBuf)],
    models_dir: &'a Path,
    seg_model: &'a Path,
    emb_model: &'a Path,
    impls: &'a [String],
    total_audio_seconds: f64,
    preflight_failures: &'a HashMap<String, String>,
    sleep_between: Option<Duration>,
    pyannote_batch_sizes: PyannoteBatchSizes,
}

pub(super) fn write_impl_result(
    run_dir: &Path,
    impl_name: &str,
    result: &DerImplResult,
    total_audio_seconds: f64,
) {
    let slug = impl_name.to_lowercase().replace(' ', "-");
    let payload = serde_json::json!({
        "implementation": impl_name,
        "status": result.status,
        "reason": result.reason,
        "der": result.der,
        "missed": result.missed,
        "false_alarm": result.false_alarm,
        "confusion": result.confusion,
        "time": result.time,
        "files": result.files,
        "total_audio_seconds": total_audio_seconds,
    });
    let _ = fs::write(
        run_dir.join(format!("{slug}.json")),
        serde_json::to_string_pretty(&payload).unwrap_or_default() + "\n",
    );
}

fn run_der_implementations(ctx: &DerRunContext) -> Result<DerResults> {
    let speakrs_binary = ctx.root.join("target/release/xtask");
    let pyannote_rs_binary = ctx
        .root
        .join("scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs");
    let fluidaudio_bench_dir = ctx.root.join("scripts/fluidaudio-bench");
    let speakerkit_bench_dir = ctx.root.join("scripts/speakerkit-bench");
    let implementations: Vec<(&str, ImplType)> = IMPL_REGISTRY
        .iter()
        .filter(|(cli_id, alias, _, _)| {
            ctx.impls.is_empty()
                || ctx
                    .impls
                    .iter()
                    .any(|value| value == cli_id || value == alias)
        })
        .map(|(_, _, display_name, impl_type)| (*display_name, *impl_type))
        .collect();

    let wav_paths: Vec<&Path> = ctx.files.iter().map(|(wav, _)| wav.as_path()).collect();
    let mut all_results: HashMap<String, DerImplResult> = HashMap::new();
    let batch_timeout = Duration::from_secs_f64((ctx.total_audio_seconds * 5.0).max(120.0));

    for (impl_name, impl_type) in &implementations {
        println!("Running {impl_name}...");

        if let Some(reason) = ctx.preflight_failures.get(*impl_name) {
            println!("  → skipped (preflight failed): {reason}");
            println!();
            let result = DerImplResult::failed(format!("preflight failed: {reason}"));
            write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
            all_results.insert(impl_name.to_string(), result);
            continue;
        }

        if let Some(reason) = der_skip_reason(
            ctx.root,
            impl_type,
            &fluidaudio_bench_dir,
            &speakerkit_bench_dir,
            &pyannote_rs_binary,
            ctx.seg_model,
            ctx.emb_model,
        ) {
            println!("  → skipped: {reason}");
            println!();
            let result = DerImplResult::skipped(reason);
            write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
            all_results.insert(impl_name.to_string(), result);
            continue;
        }

        let benchmark_result = match impl_type {
            ImplType::Speakrs(mode) => {
                BatchCommandRunner::speakrs(&speakrs_binary, mode, ctx.models_dir, &wav_paths)
                    .run_with_retries(batch_timeout)
            }
            ImplType::FluidAudioBench => {
                if let Err(err) = run_cmd(
                    Command::new("swift")
                        .args(["build", "-c", "release", "--package-path"])
                        .arg(&fluidaudio_bench_dir),
                ) {
                    Err(err)
                } else {
                    BatchCommandRunner::binary(
                        &fluidaudio_bench_dir.join(".build/release/fluidaudio-bench"),
                        &wav_paths,
                    )
                    .run_with_retries(batch_timeout)
                }
            }
            ImplType::SpeakerKitBench => {
                if let Err(err) = run_cmd(
                    Command::new("swift")
                        .args(["build", "-c", "release", "--package-path"])
                        .arg(&speakerkit_bench_dir),
                ) {
                    Err(err)
                } else {
                    BatchCommandRunner::binary(
                        &speakerkit_bench_dir.join(".build/release/speakerkit-bench"),
                        &wav_paths,
                    )
                    .run_with_retries(batch_timeout)
                }
            }
            ImplType::PyannoteRs => PyannoteRsFileRunner::new(
                pyannote_rs_binary.clone(),
                ctx.seg_model.to_path_buf(),
                ctx.emb_model.to_path_buf(),
            )
            .run(ctx.files),
            ImplType::Pyannote(device) => {
                BatchCommandRunner::pyannote(ctx.root, device, &wav_paths, ctx.pyannote_batch_sizes)
                    .run_with_retries(batch_timeout)
            }
        };

        let benchmark_output = match benchmark_result {
            Ok(result) => result,
            Err(err) => {
                println!("  → failed: {err}");
                println!();
                let result = DerImplResult::failed(err.to_string());
                write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
                all_results.insert(impl_name.to_string(), result);
                continue;
            }
        };

        let acc = DerAccumulation::compute(ctx.files, &benchmark_output.per_file_rttm)?;
        let (der_pct, miss_pct, fa_pct, conf_pct) = acc.der_percentages();
        let rtfx = ctx.total_audio_seconds / benchmark_output.total_seconds;
        if let Some(der) = der_pct {
            println!(
                "  → DER: {der:.1}%, Missed: {:.1}%, FA: {:.1}%, Confusion: {:.1}%, Time: {:.1}s, RTFx: {rtfx:.1}x",
                miss_pct.unwrap_or(0.0),
                fa_pct.unwrap_or(0.0),
                conf_pct.unwrap_or(0.0),
                benchmark_output.total_seconds,
            );
        } else {
            println!(
                "  → N/A, Time: {:.1}s, RTFx: {rtfx:.1}x",
                benchmark_output.total_seconds
            );
        }
        println!();

        let result = DerImplResult::completed(
            der_pct,
            miss_pct,
            fa_pct,
            conf_pct,
            benchmark_output.total_seconds,
            acc.file_count,
        );
        write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
        all_results.insert(impl_name.to_string(), result);

        if let Some(delay) = ctx.sleep_between {
            println!(
                "  Sleeping {}s before next implementation...",
                delay.as_secs()
            );
            std::thread::sleep(delay);
        }
    }

    Ok((implementations, all_results))
}

fn der_build_features(impls: &[String]) -> Vec<String> {
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

fn preflight_check(
    root: &Path,
    file: &(PathBuf, PathBuf),
    models_dir: &Path,
    seg_model: &Path,
    emb_model: &Path,
    impls: &[String],
    pyannote_batch_sizes: PyannoteBatchSizes,
) -> Result<HashMap<String, String>> {
    let (wav_path, _) = file;
    let stem = file_stem_string(wav_path)?;
    let duration = wav_duration_seconds(wav_path).unwrap_or(0.0);
    println!();
    println!("=== Pre-flight check ({stem}, {duration:.0}s) ===");

    let speakrs_binary = root.join("target/release/xtask");
    let pyannote_rs_binary =
        root.join("scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs");
    let fluidaudio_bench_dir = root.join("scripts/fluidaudio-bench");
    let speakerkit_bench_dir = root.join("scripts/speakerkit-bench");

    let implementations: Vec<(&str, &str, ImplType)> = if impls.is_empty() {
        IMPL_REGISTRY
            .iter()
            .map(|(cli_id, _, display_name, impl_type)| (*cli_id, *display_name, *impl_type))
            .collect()
    } else {
        IMPL_REGISTRY
            .iter()
            .filter(|(cli_id, alias, _, _)| {
                impls.iter().any(|value| value == cli_id || value == alias)
            })
            .map(|(cli_id, _, display_name, impl_type)| (*cli_id, *display_name, *impl_type))
            .collect()
    };

    let wav_paths: Vec<&Path> = vec![wav_path.as_path()];
    let mut failures: HashMap<String, String> = HashMap::new();

    for (_cli_id, display_name, impl_type) in &implementations {
        if let Some(reason) = der_skip_reason(
            root,
            impl_type,
            &fluidaudio_bench_dir,
            &speakerkit_bench_dir,
            &pyannote_rs_binary,
            seg_model,
            emb_model,
        ) {
            println!("  {display_name:<22} skipped ({reason})");
            continue;
        }

        let result = match impl_type {
            ImplType::Speakrs(mode) => {
                BatchCommandRunner::speakrs(&speakrs_binary, mode, models_dir, &wav_paths)
                    .run_with_retries(PREFLIGHT_TIMEOUT)
            }
            ImplType::FluidAudioBench => {
                if let Err(err) = run_cmd(
                    Command::new("swift")
                        .args(["build", "-c", "release", "--package-path"])
                        .arg(&fluidaudio_bench_dir),
                ) {
                    Err(err)
                } else {
                    BatchCommandRunner::binary(
                        &fluidaudio_bench_dir.join(".build/release/fluidaudio-bench"),
                        &wav_paths,
                    )
                    .run_with_retries(PREFLIGHT_TIMEOUT)
                }
            }
            ImplType::SpeakerKitBench => {
                if let Err(err) = run_cmd(
                    Command::new("swift")
                        .args(["build", "-c", "release", "--package-path"])
                        .arg(&speakerkit_bench_dir),
                ) {
                    Err(err)
                } else {
                    BatchCommandRunner::binary(
                        &speakerkit_bench_dir.join(".build/release/speakerkit-bench"),
                        &wav_paths,
                    )
                    .run_with_retries(PREFLIGHT_TIMEOUT)
                }
            }
            ImplType::PyannoteRs => PyannoteRsFileRunner::new(
                pyannote_rs_binary.clone(),
                seg_model.to_path_buf(),
                emb_model.to_path_buf(),
            )
            .run(&[(wav_path.clone(), PathBuf::new())]),
            ImplType::Pyannote(device) => {
                BatchCommandRunner::pyannote(root, device, &wav_paths, pyannote_batch_sizes)
                    .run_with_retries(PREFLIGHT_TIMEOUT)
            }
        };

        match result {
            Ok(batch_output) => {
                let has_output = batch_output
                    .per_file_rttm
                    .values()
                    .any(|value| !value.trim().is_empty());
                if has_output {
                    println!(
                        "  {display_name:<22} ok ({:.1}s)",
                        batch_output.total_seconds
                    );
                } else {
                    let reason = "empty RTTM output".to_string();
                    println!("  {display_name:<22} FAILED: {reason}");
                    failures.insert(display_name.to_string(), reason);
                }
            }
            Err(err) => {
                let reason = format!("{err}");
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

fn der_skip_reason(
    root: &Path,
    impl_type: &ImplType,
    fluidaudio_bench_dir: &Path,
    speakerkit_bench_dir: &Path,
    pyannote_rs_binary: &Path,
    seg_model: &Path,
    emb_model: &Path,
) -> Option<String> {
    match impl_type {
        ImplType::Pyannote(_) => (!root.join("scripts/pyannote-bench/diarize.py").exists())
            .then(|| "scripts/pyannote-bench/diarize.py not found".to_string()),
        ImplType::Speakrs(mode) => {
            #[cfg(not(target_os = "macos"))]
            if mode.starts_with("coreml") {
                return Some("CoreML not available on this platform".to_string());
            }
            let _ = mode;
            None
        }
        ImplType::FluidAudioBench => (!fluidaudio_bench_dir.join("Package.swift").exists())
            .then(|| "scripts/fluidaudio-bench/Package.swift not found".to_string()),
        ImplType::SpeakerKitBench => (!speakerkit_bench_dir.join("Package.swift").exists())
            .then(|| "scripts/speakerkit-bench/Package.swift not found".to_string()),
        ImplType::PyannoteRs => {
            if !pyannote_rs_binary.exists() {
                Some("pyannote-rs bench binary not found".to_string())
            } else if !seg_model.exists() {
                Some(format!(
                    "segmentation model not found: {}",
                    seg_model.display()
                ))
            } else if !emb_model.exists() {
                Some(format!(
                    "embedding model not found: {}",
                    emb_model.display()
                ))
            } else {
                None
            }
        }
    }
}

pub(super) fn ensure_pyannote_rs_emb_model(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    println!("Downloading pyannote-rs embedding model...");
    run_cmd(
        Command::new("curl")
            .args(["-L", "-o"])
            .arg(path)
            .arg("https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx"),
    )
}
