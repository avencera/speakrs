use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::Result;

use crate::audio::prepare_audio;
use crate::cargo::{cargo_build, features_for_mode};
use crate::cmd::{capture_cmd, project_root, run_cmd, wav_duration_seconds};
use crate::fluidaudio;

// ---------------------------------------------------------------------------
// benchmark run — speakrs vs pyannote timing
// ---------------------------------------------------------------------------

pub fn run(
    source: &str,
    python_device: &str,
    runs: u32,
    warmups: u32,
    rust_mode: &str,
) -> Result<()> {
    let (wav, _tmp) = prepare_audio(source)?;
    let features = features_for_mode(rust_mode);

    println!();
    println!("=== Building Rust binary ===");
    cargo_build("diarize", &features)?;

    let root = project_root();
    let binary = root.join("target/release/diarize");
    let audio_seconds = wav_duration_seconds(&wav)?;

    println!();
    println!("=== Benchmark ===");

    let rust_result = bench_tool(
        "Rust",
        &[
            binary.to_str().unwrap(),
            "--mode",
            rust_mode,
            wav.to_str().unwrap(),
        ],
        runs,
        warmups,
    )?;

    let python_result = bench_tool(
        "Python",
        &[
            "uv",
            "run",
            "scripts/diarize_pyannote.py",
            "--device",
            python_device,
            wav.to_str().unwrap(),
        ],
        runs,
        warmups,
    )?;

    println!(
        "Audio duration: {:.2} minutes ({audio_seconds:.1}s)",
        audio_seconds / 60.0
    );
    println!("Runs: {runs} measured, {warmups} warmup");
    println!("Rust mode: {rust_mode}");
    println!("Python device: {python_device}");
    println!();
    println!("Name\tmean_s\tmin_s\tspeed");
    for (name, times) in [("Rust", &rust_result), ("Python", &python_result)] {
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("{name}\t{mean:.3}\t{min:.3}\t{:.2}x", audio_seconds / mean);
    }

    Ok(())
}

fn bench_tool(name: &str, command: &[&str], runs: u32, warmups: u32) -> Result<Vec<f64>> {
    let root = project_root();

    for _ in 0..warmups {
        run_once(command, &root)?;
    }

    let mut times = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let elapsed = run_once(command, &root)?;
        times.push(elapsed);
        print!("  {name}: {elapsed:.2}s\r");
    }
    println!("  {name}: done ({runs} runs)");
    Ok(times)
}

fn run_once(command: &[&str], cwd: &Path) -> Result<f64> {
    let (elapsed, _) = capture_cmd(
        Command::new(command[0])
            .args(&command[1..])
            .current_dir(cwd),
        600,
    )?;
    Ok(elapsed.as_secs_f64())
}

// ---------------------------------------------------------------------------
// benchmark compare — multi-tool comparison
// ---------------------------------------------------------------------------

struct RunResult {
    name: String,
    seconds: f64,
    rttm: String,
    speakers: usize,
    segments: usize,
}

impl RunResult {
    fn new(name: &str, seconds: f64, rttm: String) -> Self {
        let mut speakers_set = std::collections::HashSet::new();
        let mut segments = 0;
        for line in rttm.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.first() == Some(&"SPEAKER") && parts.len() >= 8 {
                segments += 1;
                speakers_set.insert(parts[7].to_string());
            }
        }
        Self {
            name: name.to_string(),
            seconds,
            rttm,
            speakers: speakers_set.len(),
            segments,
        }
    }
}

pub fn compare(source: &str, runs: u32, warmups: u32) -> Result<()> {
    let (wav, _tmp) = prepare_audio(source)?;
    let wav_str = wav.to_string_lossy();
    let root = project_root();

    println!();
    println!("=== Building binaries ===");
    cargo_build("diarize", &["native-coreml".to_string()])?;
    run_cmd(
        Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(root.join("scripts/pyannote_rs_bench")),
    )?;

    let audio_seconds = wav_duration_seconds(&wav)?;
    let models_dir = root.join("fixtures/models");
    let seg_model = models_dir.join("segmentation-3.0.onnx");
    let emb_model = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    ensure_pyannote_rs_emb_model(&emb_model)?;

    println!();
    println!("=== Running benchmarks ===");

    let speakrs_binary = root.join("target/release/diarize");
    let pyannote_rs_binary =
        root.join("scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs");

    let implementations: Vec<(&str, Vec<String>)> = vec![
        (
            "pyannote MPS",
            vec![
                "uv".into(),
                "run".into(),
                "scripts/diarize_pyannote.py".into(),
                "--device".into(),
                "mps".into(),
                wav_str.to_string(),
            ],
        ),
        (
            "speakrs CoreML",
            vec![
                speakrs_binary.to_string_lossy().into(),
                "--mode".into(),
                "coreml".into(),
                wav_str.to_string(),
            ],
        ),
        (
            "speakrs MiniCoreML",
            vec![
                speakrs_binary.to_string_lossy().into(),
                "--mode".into(),
                "mini-coreml".into(),
                wav_str.to_string(),
            ],
        ),
        (
            "pyannote-rs",
            vec![
                pyannote_rs_binary.to_string_lossy().into(),
                wav_str.to_string(),
                seg_model.to_string_lossy().into(),
                emb_model.to_string_lossy().into(),
            ],
        ),
    ];

    let has_fluidaudio = find_fluidaudio().is_some();

    let mut results: Vec<RunResult> = Vec::new();

    for (name, cmd_args) in &implementations {
        let cmd_refs: Vec<&str> = cmd_args.iter().map(String::as_str).collect();

        for _ in 0..warmups {
            run_capture_impl(&cmd_refs, &root);
        }

        let mut best_time = f64::INFINITY;
        let mut best_rttm = String::new();
        for _ in 0..runs {
            let (elapsed, rttm) = run_capture_impl(&cmd_refs, &root);
            if elapsed < best_time {
                best_time = elapsed;
                best_rttm = rttm;
            }
        }
        results.push(RunResult::new(name, best_time, best_rttm));
        println!("  {name}: {best_time:.2}s");
    }

    if has_fluidaudio {
        let fluidaudio_path = find_fluidaudio().unwrap();

        for _ in 0..warmups {
            run_fluidaudio_impl(&fluidaudio_path, &wav_str);
        }

        let mut best_time = f64::INFINITY;
        let mut best_rttm = String::new();
        for _ in 0..runs {
            let (elapsed, rttm) = run_fluidaudio_impl(&fluidaudio_path, &wav_str);
            if elapsed < best_time {
                best_time = elapsed;
                best_rttm = rttm;
            }
        }
        results.push(RunResult::new("FluidAudio", best_time, best_rttm));
        println!("  FluidAudio: {best_time:.2}s");
    }

    // pyannote MPS is the reference
    let ref_rttm = results[0].rttm.clone();

    let minutes = (audio_seconds / 60.0) as u32;
    let secs = audio_seconds % 60.0;
    println!();
    println!(
        "Audio: {minutes}:{secs:04.1} ({audio_seconds:.1}s)  |  Warmups: {warmups}  |  Runs: {runs}"
    );
    println!();

    let name_w = 22;
    println!(
        "{:<name_w$} {:>8} {:>9} {:>9} {:>10}",
        "Implementation", "Time", "Speakers", "Segments", "Parity %"
    );
    println!("{}", "─".repeat(name_w + 8 + 9 + 9 + 10 + 4));

    for (i, result) in results.iter().enumerate() {
        let time_str = format!("{:.2}s", result.seconds);
        let speakers_str = if result.segments > 0 {
            result.speakers.to_string()
        } else {
            "—".to_string()
        };
        let segments_str = result.segments.to_string();

        let parity_str = if i == 0 {
            "(reference)".to_string()
        } else if result.segments == 0 {
            "N/A".to_string()
        } else {
            match timeline_overlap_pct(&ref_rttm, &result.rttm) {
                Some(pct) => format!("{pct:.1}%"),
                None => "N/A".to_string(),
            }
        };

        println!(
            "{:<name_w$} {:>8} {:>9} {:>9} {:>10}",
            result.name, time_str, speakers_str, segments_str, parity_str
        );
    }

    if results
        .iter()
        .any(|r| r.name == "pyannote-rs" && r.segments == 0)
    {
        println!();
        println!("Note: pyannote-rs returned 0 segments. It only emits segments when");
        println!("speech→silence transitions occur; continuous speech files produce no output.");
    }

    Ok(())
}

fn run_capture_impl(command: &[&str], cwd: &Path) -> (f64, String) {
    let result = capture_cmd(
        Command::new(command[0])
            .args(&command[1..])
            .current_dir(cwd),
        600,
    );
    match result {
        Ok((elapsed, stdout)) => (elapsed.as_secs_f64(), stdout),
        Err(_) => (0.0, String::new()),
    }
}

fn run_fluidaudio_impl(fluidaudio_path: &Path, wav_path: &str) -> (f64, String) {
    let json_tmp = std::env::temp_dir().join(format!("fa-{}.json", std::process::id()));

    let start = std::time::Instant::now();
    let result = capture_cmd(
        Command::new("swift")
            .args(["run", "-c", "release", "--package-path"])
            .arg(fluidaudio_path)
            .args(["fluidaudiocli", "process"])
            .arg(wav_path)
            .args(["--mode", "offline", "--output"])
            .arg(&json_tmp),
        600,
    );
    let elapsed = start.elapsed().as_secs_f64();

    if result.is_err() || !json_tmp.exists() {
        let _ = fs::remove_file(&json_tmp);
        return (elapsed, String::new());
    }

    let rttm = fluidaudio::json_to_rttm(&json_tmp, "file1").unwrap_or_default();
    let _ = fs::remove_file(&json_tmp);
    (elapsed, rttm)
}

fn timeline_overlap_pct(ref_rttm: &str, test_rttm: &str) -> Option<f64> {
    let ref_intervals = parse_rttm_intervals(ref_rttm);
    let test_intervals = parse_rttm_intervals(test_rttm);

    if ref_intervals.is_empty() || test_intervals.is_empty() {
        return None;
    }

    let mut merged = test_intervals;
    merged.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut m = vec![merged[0]];
    for &(start, end) in &merged[1..] {
        let last = m.last_mut().unwrap();
        if start <= last.1 {
            last.1 = last.1.max(end);
        } else {
            m.push((start, end));
        }
    }

    let ref_total: f64 = ref_intervals.iter().map(|(s, e)| e - s).sum();
    if ref_total <= 0.0 {
        return None;
    }

    let mut covered = 0.0;
    for &(rs, re) in &ref_intervals {
        for &(ms, me) in &m {
            if ms >= re {
                break;
            }
            if me <= rs {
                continue;
            }
            covered += re.min(me) - rs.max(ms);
        }
    }

    Some(covered / ref_total * 100.0)
}

fn parse_rttm_intervals(rttm: &str) -> Vec<(f64, f64)> {
    rttm.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.first() != Some(&"SPEAKER") || parts.len() < 5 {
                return None;
            }
            let start: f64 = parts[3].parse().ok()?;
            let dur: f64 = parts[4].parse().ok()?;
            Some((start, start + dur))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// benchmark der — DER evaluation on VoxConverse dev set
// ---------------------------------------------------------------------------

pub fn der(dataset_id: &str, max_files: u32, max_minutes: u32) -> Result<()> {
    if dataset_id == "list" {
        println!("Available datasets:");
        for id in crate::datasets::list_dataset_ids() {
            println!("  {id}");
        }
        println!("  all  (run all datasets)");
        return Ok(());
    }

    let datasets: Vec<Box<dyn crate::datasets::Dataset>> = if dataset_id == "all" {
        crate::datasets::all_datasets()
    } else {
        vec![crate::datasets::find_dataset(dataset_id).ok_or_else(|| {
            color_eyre::eyre::eyre!(
                "unknown dataset: {dataset_id}. Use --dataset list to see available datasets"
            )
        })?]
    };

    let root = project_root();

    println!("=== Building binaries ===");
    cargo_build("diarize", &["native-coreml".to_string()])?;
    if let Err(e) = run_cmd(
        Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(root.join("scripts/pyannote_rs_bench")),
    ) {
        eprintln!("warning: pyannote-rs bench build failed (skipping): {e}");
    }

    let models_dir = root.join("fixtures/models");
    let seg_model = models_dir.join("segmentation-3.0.onnx");
    let emb_model = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    ensure_pyannote_rs_emb_model(&emb_model)?;

    // SAFETY: single-threaded CLI, no other threads reading env vars
    unsafe { std::env::set_var("SPEAKRS_MODELS_DIR", &models_dir) };

    let fixtures_dir = root.join("fixtures");

    for dataset in &datasets {
        println!();
        println!("========== {} ==========", dataset.display_name());

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

        let total_audio_seconds: f64 = files
            .iter()
            .map(|(wav, _)| wav_duration_seconds(wav).unwrap_or(0.0))
            .sum();
        let total_audio_minutes = total_audio_seconds / 60.0;

        let run_id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let run_dir = if datasets.len() > 1 {
            root.join("benchmarks").join(&run_id).join(dataset.id())
        } else {
            root.join("benchmarks").join(&run_id)
        };
        fs::create_dir_all(&run_dir)?;

        println!(
            "Found {} files, {total_audio_minutes:.1} min total audio",
            files.len()
        );
        println!("Run ID: {run_id}");
        println!();

        let (implementations, all_results) =
            run_der_implementations(&root, &files, &seg_model, &emb_model)?;

        save_der_results(
            &run_dir,
            dataset.display_name(),
            &implementations,
            &all_results,
            &files,
            total_audio_minutes,
            0.0,
        )?;
    }

    Ok(())
}

type DerResults = (
    Vec<(&'static str, ImplType)>,
    HashMap<String, DerImplResult>,
);

fn run_der_implementations(
    root: &Path,
    files: &[(PathBuf, PathBuf)],
    seg_model: &Path,
    emb_model: &Path,
) -> Result<DerResults> {
    let speakrs_binary = root.join("target/release/diarize");
    let pyannote_rs_binary =
        root.join("scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs");

    let fluidaudio_bench_dir = root.join("scripts/fluidaudio-bench");
    let has_fluidaudio_bench = fluidaudio_bench_dir.join("Package.swift").exists();
    let has_pyannote_rs = pyannote_rs_binary.exists() && seg_model.exists() && emb_model.exists();

    let mut implementations: Vec<(&str, ImplType)> = Vec::new();

    if root.join("scripts/diarize_pyannote.py").exists() {
        implementations.push(("pyannote MPS", ImplType::Pyannote));
    }
    implementations.push(("speakrs CoreML", ImplType::Speakrs("coreml")));
    implementations.push(("speakrs CoreML Fast", ImplType::Speakrs("coreml-fast")));
    if has_fluidaudio_bench {
        implementations.push(("FluidAudio", ImplType::FluidAudioBench));
    }
    if has_pyannote_rs {
        implementations.push(("pyannote-rs", ImplType::PyannoteRs));
    }

    let wav_paths: Vec<&Path> = files.iter().map(|(w, _)| w.as_path()).collect();
    let mut all_results: HashMap<String, DerImplResult> = HashMap::new();

    for (impl_name, impl_type) in &implementations {
        println!("Running {impl_name}...");

        let (total_time, per_file_rttm) = match impl_type {
            ImplType::Speakrs(mode) => run_speakrs_batch(&speakrs_binary, mode, &wav_paths)?,
            ImplType::FluidAudioBench => {
                run_cmd(
                    Command::new("swift")
                        .args(["build", "-c", "release", "--package-path"])
                        .arg(&fluidaudio_bench_dir),
                )?;
                run_batch_binary(
                    &fluidaudio_bench_dir.join(".build/release/fluidaudio-bench"),
                    &wav_paths,
                )?
            }
            ImplType::PyannoteRs => run_per_file(impl_name, files, |wav| {
                let cmd_args = [
                    pyannote_rs_binary.to_string_lossy().to_string(),
                    wav.to_string_lossy().to_string(),
                    seg_model.to_string_lossy().to_string(),
                    emb_model.to_string_lossy().to_string(),
                ];
                let refs: Vec<&str> = cmd_args.iter().map(String::as_str).collect();
                run_capture_impl(&refs, root)
            })?,
            ImplType::Pyannote => run_per_file(impl_name, files, |wav| {
                let tmp =
                    std::env::temp_dir().join(format!("pyannote-{}.rttm", std::process::id()));
                let tmp_str = tmp.to_string_lossy().to_string();
                let wav_str = wav.to_string_lossy().to_string();
                let cmd_args = vec![
                    "uv",
                    "run",
                    "scripts/diarize_pyannote.py",
                    "--device",
                    "mps",
                    "--output",
                    &tmp_str,
                    &wav_str,
                ];
                let (elapsed, _) = run_capture_impl(&cmd_args, root);
                let rttm = fs::read_to_string(&tmp).unwrap_or_default();
                let _ = fs::remove_file(&tmp);
                (elapsed, rttm)
            })?,
        };

        let mut total_missed = 0.0;
        let mut total_fa = 0.0;
        let mut total_confusion = 0.0;
        let mut total_ref = 0.0;
        let mut file_count = 0;

        for (wav_path, rttm_path) in files {
            let ref_text = fs::read_to_string(rttm_path)?;
            let ref_segs = speakrs::metrics::parse_rttm(&ref_text);

            let hyp_text = per_file_rttm
                .get(&wav_path.file_stem().unwrap().to_string_lossy().to_string())
                .cloned()
                .unwrap_or_default();

            if hyp_text.trim().is_empty() {
                file_count += 1;
                total_ref += ref_segs.iter().map(|s| s.duration()).sum::<f64>();
                total_missed += ref_segs.iter().map(|s| s.duration()).sum::<f64>();
                continue;
            }

            let hyp_segs = speakrs::metrics::parse_rttm(&hyp_text);
            let der_result = speakrs::metrics::compute_der(&ref_segs, &hyp_segs);
            total_missed += der_result.missed;
            total_fa += der_result.false_alarm;
            total_confusion += der_result.confusion;
            total_ref += der_result.total;
            file_count += 1;
        }

        let (der_pct, miss_pct, fa_pct, conf_pct) = if total_ref > 0.0 {
            (
                Some((total_missed + total_fa + total_confusion) / total_ref * 100.0),
                Some(total_missed / total_ref * 100.0),
                Some(total_fa / total_ref * 100.0),
                Some(total_confusion / total_ref * 100.0),
            )
        } else {
            (None, None, None, None)
        };

        if let Some(d) = der_pct {
            println!("  → DER: {d:.1}%, Time: {total_time:.1}s");
        } else {
            println!("  → N/A, Time: {total_time:.1}s");
        }
        println!();

        all_results.insert(
            impl_name.to_string(),
            DerImplResult {
                der: der_pct,
                missed: miss_pct,
                false_alarm: fa_pct,
                confusion: conf_pct,
                time: total_time,
                files: file_count,
            },
        );
    }

    Ok((implementations, all_results))
}

enum ImplType {
    Speakrs(&'static str),
    Pyannote,
    PyannoteRs,
    FluidAudioBench,
}

#[derive(serde::Serialize)]
struct DerImplResult {
    der: Option<f64>,
    missed: Option<f64>,
    false_alarm: Option<f64>,
    confusion: Option<f64>,
    time: f64,
    files: usize,
}

fn run_speakrs_batch(
    binary: &Path,
    mode: &str,
    wav_paths: &[&Path],
) -> Result<(f64, HashMap<String, String>)> {
    let mut cmd = Command::new(binary);
    cmd.arg("--mode").arg(mode);
    for p in wav_paths {
        cmd.arg(p);
    }

    let (elapsed, stdout) = capture_cmd(&mut cmd, 1200)?;
    let per_file = split_rttm_by_file_id(&stdout);

    println!(
        "  batch: {:.1}s ({} files)",
        elapsed.as_secs_f64(),
        wav_paths.len()
    );
    Ok((elapsed.as_secs_f64(), per_file))
}

fn run_batch_binary(binary: &Path, wav_paths: &[&Path]) -> Result<(f64, HashMap<String, String>)> {
    let mut cmd = Command::new(binary);
    for p in wav_paths {
        cmd.arg(p);
    }

    let (elapsed, stdout) = capture_cmd(&mut cmd, 1200)?;
    let per_file = split_rttm_by_file_id(&stdout);

    println!(
        "  batch: {:.1}s ({} files)",
        elapsed.as_secs_f64(),
        wav_paths.len()
    );
    Ok((elapsed.as_secs_f64(), per_file))
}

fn run_per_file(
    _name: &str,
    files: &[(PathBuf, PathBuf)],
    runner: impl Fn(&Path) -> (f64, String),
) -> Result<(f64, HashMap<String, String>)> {
    let mut total_time = 0.0;
    let mut per_file = HashMap::new();

    for (wav_path, _) in files {
        let (elapsed, rttm) = runner(wav_path);
        total_time += elapsed;
        let stem = wav_path.file_stem().unwrap().to_string_lossy().to_string();
        per_file.insert(stem.clone(), rttm);
        println!("  {stem}: {elapsed:.1}s");
    }

    Ok((total_time, per_file))
}

fn split_rttm_by_file_id(stdout: &str) -> HashMap<String, String> {
    let mut per_file: HashMap<String, Vec<&str>> = HashMap::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.first() == Some(&"SPEAKER") && parts.len() >= 2 {
            per_file.entry(parts[1].to_string()).or_default().push(line);
        }
    }
    per_file
        .into_iter()
        .map(|(k, v)| (k, v.join("\n") + "\n"))
        .collect()
}

fn discover_files(
    dataset_dir: &Path,
    max_files: u32,
    max_minutes: f64,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    let wav_dir = dataset_dir.join("wav");
    let rttm_dir = dataset_dir.join("rttm");

    let (wav_dir, rttm_dir) = if wav_dir.exists() && rttm_dir.exists() {
        (wav_dir, rttm_dir)
    } else {
        (dataset_dir.to_path_buf(), dataset_dir.to_path_buf())
    };

    let mut pairs: Vec<(PathBuf, PathBuf, f64)> = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(&wav_dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let wav_path = entry.path();
        if wav_path.extension().is_some_and(|e| e == "wav") {
            let stem = wav_path.file_stem().unwrap();
            let rttm_path = rttm_dir.join(format!("{}.rttm", stem.to_string_lossy()));
            if rttm_path.exists()
                && let Ok(dur) = wav_duration_seconds(&wav_path)
            {
                pairs.push((wav_path, rttm_path, dur));
            }
        }
    }

    // sort by duration (shortest first)
    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let mut selected = Vec::new();
    let mut total_minutes = 0.0;

    for (wav, rttm, dur) in pairs {
        if selected.len() >= max_files as usize {
            break;
        }
        if total_minutes + dur / 60.0 > max_minutes && !selected.is_empty() {
            break;
        }
        selected.push((wav, rttm));
        total_minutes += dur / 60.0;
    }

    Ok(selected)
}

fn save_der_results(
    run_dir: &Path,
    dataset_name: &str,
    implementations: &[(&str, ImplType)],
    results: &HashMap<String, DerImplResult>,
    files: &[(PathBuf, PathBuf)],
    total_audio_minutes: f64,
    collar: f64,
) -> Result<()> {
    let mut json_results = serde_json::Map::new();
    for (name, _) in implementations {
        if let Some(r) = results.get(*name) {
            json_results.insert(name.to_string(), serde_json::to_value(r)?);
        }
    }

    let data = serde_json::json!({
        "dataset": dataset_name,
        "run_id": run_dir.file_name().unwrap().to_string_lossy(),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "files": files.len(),
        "total_audio_minutes": (total_audio_minutes * 10.0).round() / 10.0,
        "collar": collar,
        "file_list": files.iter().map(|(w, _)| w.file_stem().unwrap().to_string_lossy().to_string()).collect::<Vec<_>>(),
        "results": json_results,
    });

    let json_path = run_dir.join("results.json");
    fs::write(&json_path, serde_json::to_string_pretty(&data)? + "\n")?;

    // text summary
    let mut lines = Vec::new();
    lines.push(format!(
        "{dataset_name} DER ({} files, {total_audio_minutes:.1} min, collar={collar:.0}ms)",
        files.len()
    ));
    lines.push(String::new());

    let name_w = 22;
    let header = format!(
        "{:<name_w$} {:>8} {:>10} {:>13} {:>12} {:>8}",
        "Implementation", "DER%", "Missed%", "FalseAlarm%", "Confusion%", "Time"
    );
    lines.push(header.clone());
    lines.push("─".repeat(header.len()));

    for (impl_name, _) in implementations {
        if let Some(r) = results.get(*impl_name) {
            let (der_str, miss_str, fa_str, conf_str) = match r.der {
                Some(d) => (
                    format!("{d:.1}%"),
                    format!("{:.1}%", r.missed.unwrap_or(0.0)),
                    format!("{:.1}%", r.false_alarm.unwrap_or(0.0)),
                    format!("{:.1}%", r.confusion.unwrap_or(0.0)),
                ),
                None => (
                    "N/A".to_string(),
                    "—".to_string(),
                    "—".to_string(),
                    "—".to_string(),
                ),
            };
            let time_str = format!("{:.1}s", r.time);
            lines.push(format!(
                "{:<name_w$} {:>8} {:>10} {:>13} {:>12} {:>8}",
                impl_name, der_str, miss_str, fa_str, conf_str, time_str
            ));
        }
    }

    let txt_path = run_dir.join("results.txt");
    fs::write(&txt_path, lines.join("\n") + "\n")?;

    println!("\nResults saved to {}/", run_dir.display());

    // also print the summary to stdout
    for line in &lines {
        println!("{line}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn ensure_pyannote_rs_emb_model(path: &Path) -> Result<()> {
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

fn find_fluidaudio() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FLUIDAUDIO_PATH") {
        let p = PathBuf::from(path);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let default = PathBuf::from(home).join(".cache/cmd/repos/github.com/FluidInference/FluidAudio");
    if default.is_dir() {
        Some(default)
    } else {
        None
    }
}
