use std::collections::HashMap;
use std::ffi::OsString;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, bail};
use wait_timeout::ChildExt;

use crate::audio::prepare_audio;
use crate::cargo::{cargo_build_xtask, features_for_mode};
use crate::cmd::{find_fluidaudio, project_root, run_cmd, wav_duration_seconds};
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
    cargo_build_xtask(&features)?;

    let root = project_root();
    let binary = root.join("target/release/xtask");
    let audio_seconds = wav_duration_seconds(&wav)?;

    println!();
    println!("=== Benchmark ===");

    let rust_result = bench_tool(
        "Rust",
        &[
            binary.to_str().unwrap(),
            "diarize",
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
    let mut benchmark_command = Command::new(command[0]);
    benchmark_command.args(&command[1..]).current_dir(cwd);
    let output = capture_benchmark_cmd(&mut benchmark_command, Duration::from_secs(30 * 60))?;
    Ok(output.elapsed_seconds)
}

// ---------------------------------------------------------------------------
// benchmark compare — multi-tool comparison
// ---------------------------------------------------------------------------

struct RunResult {
    name: String,
    mean_seconds: f64,
    min_seconds: f64,
    rttm: String,
    speakers: usize,
    segments: usize,
}

impl RunResult {
    fn new(name: &str, mean_seconds: f64, min_seconds: f64, rttm: String) -> Self {
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
            mean_seconds,
            min_seconds,
            rttm,
            speakers: speakers_set.len(),
            segments,
        }
    }
}

struct SingleRunOutput {
    elapsed_seconds: f64,
    rttm: String,
}

struct BatchRunOutput {
    total_seconds: f64,
    per_file_rttm: HashMap<String, String>,
}

#[derive(Clone)]
struct CommandSpec {
    program: OsString,
    args: Vec<OsString>,
    current_dir: Option<PathBuf>,
}

impl CommandSpec {
    fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            current_dir: None,
        }
    }

    fn from_argv(argv: &[String]) -> Self {
        debug_assert!(!argv.is_empty());
        let mut command_spec = Self::new(argv[0].clone());
        for arg in &argv[1..] {
            command_spec = command_spec.arg(arg.clone());
        }
        command_spec
    }

    fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    fn current_dir(mut self, current_dir: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(current_dir.into());
        self
    }

    fn build_command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        if let Some(current_dir) = &self.current_dir {
            command.current_dir(current_dir);
        }
        command
    }
}

enum CompareOutcome {
    Completed(RunResult),
    Skipped { name: String, reason: String },
    Failed { name: String, reason: String },
}

enum CompareRunner {
    Command(CommandSpec),
    FluidAudio {
        fluidaudio_path: PathBuf,
        wav_path: String,
        timeout: Duration,
    },
}

impl CompareRunner {
    fn run_once(&self) -> Result<SingleRunOutput> {
        match self {
            Self::Command(command_spec) => {
                let mut command = command_spec.build_command();
                capture_benchmark_cmd(&mut command, Duration::from_secs(30 * 60))
            }
            Self::FluidAudio {
                fluidaudio_path,
                wav_path,
                timeout,
            } => run_fluidaudio_impl(fluidaudio_path, wav_path, *timeout),
        }
    }
}

struct CompareRecorder<'a> {
    results: &'a mut Vec<CompareOutcome>,
    warmups: u32,
    runs: u32,
}

impl<'a> CompareRecorder<'a> {
    fn record(&mut self, name: &str, runner: &CompareRunner) {
        match run_compare_impl(name, self.warmups, self.runs, runner) {
            Ok(result) => {
                println!(
                    "  {name}: mean {:.2}s, min {:.2}s",
                    result.mean_seconds, result.min_seconds
                );
                self.results.push(CompareOutcome::Completed(result));
            }
            Err(err) => {
                println!("  {name}: FAILED ({err})");
                self.results.push(CompareOutcome::Failed {
                    name: name.to_string(),
                    reason: err.to_string(),
                });
            }
        }
    }
}

pub fn compare(source: &str, runs: u32, warmups: u32) -> Result<()> {
    let (wav, _tmp) = prepare_audio(source)?;
    let wav_str = wav.to_string_lossy();
    let root = project_root();

    println!();
    println!("=== Building binaries ===");
    cargo_build_xtask(&["coreml".to_string()])?;
    let pyannote_rs_build = run_cmd(
        Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(root.join("scripts/pyannote_rs_bench")),
    );

    let audio_seconds = wav_duration_seconds(&wav)?;
    let models_dir = root.join("fixtures/models");
    let seg_model = models_dir.join("segmentation-3.0.onnx");
    let emb_model = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    ensure_pyannote_rs_emb_model(&emb_model)?;

    println!();
    println!("=== Running benchmarks ===");

    let speakrs_binary = root.join("target/release/xtask");
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
                "diarize".into(),
                "--mode".into(),
                "coreml".into(),
                wav_str.to_string(),
            ],
        ),
        (
            "speakrs CoreML Fast",
            vec![
                speakrs_binary.to_string_lossy().into(),
                "diarize".into(),
                "--mode".into(),
                "coreml-fast".into(),
                wav_str.to_string(),
            ],
        ),
    ];

    let fluidaudio_path = find_fluidaudio();
    let pyannote_rs_available = pyannote_rs_build.is_ok() && pyannote_rs_binary.exists();

    let mut results: Vec<CompareOutcome> = Vec::new();

    let mut compare_recorder = CompareRecorder {
        results: &mut results,
        warmups,
        runs,
    };

    for (name, command_args) in &implementations {
        let runner =
            CompareRunner::Command(CommandSpec::from_argv(command_args).current_dir(&root));
        compare_recorder.record(name, &runner);
    }

    if let Some(fluidaudio_path) = &fluidaudio_path {
        let runner = CompareRunner::FluidAudio {
            fluidaudio_path: fluidaudio_path.clone(),
            wav_path: wav_str.to_string(),
            timeout: Duration::from_secs(30 * 60),
        };
        compare_recorder.record("FluidAudio", &runner);
    } else {
        compare_recorder.results.push(CompareOutcome::Skipped {
            name: "FluidAudio".to_string(),
            reason: "fluidaudio checkout not found".to_string(),
        });
    }

    if pyannote_rs_available {
        let cmd_args = [
            pyannote_rs_binary.to_string_lossy().to_string(),
            wav_str.to_string(),
            seg_model.to_string_lossy().to_string(),
            emb_model.to_string_lossy().to_string(),
        ];
        let runner = CompareRunner::Command(CommandSpec::from_argv(&cmd_args).current_dir(&root));
        compare_recorder.record("pyannote-rs", &runner);
    } else {
        let reason = pyannote_rs_build
            .err()
            .map(|err| format!("pyannote-rs bench build failed: {err}"))
            .unwrap_or_else(|| "pyannote-rs bench binary not found".to_string());
        compare_recorder.results.push(CompareOutcome::Skipped {
            name: "pyannote-rs".to_string(),
            reason,
        });
    }

    let ref_rttm = results.iter().find_map(|result| match result {
        CompareOutcome::Completed(run) if run.name == "pyannote MPS" => Some(run.rttm.clone()),
        _ => None,
    });

    let minutes = (audio_seconds / 60.0) as u32;
    let secs = audio_seconds % 60.0;
    println!();
    println!(
        "Audio: {minutes}:{secs:04.1} ({audio_seconds:.1}s)  |  Warmups: {warmups}  |  Runs: {runs}"
    );
    println!();

    let name_w = 22;
    println!(
        "{:<name_w$} {:>9} {:>9} {:>9} {:>9} {:>10}  Status",
        "Implementation", "Mean", "Min", "Speakers", "Segments", "Parity %"
    );
    println!("{}", "─".repeat(name_w + 9 + 9 + 9 + 9 + 10 + 10));

    for result in &results {
        match result {
            CompareOutcome::Completed(run) => {
                let mean_str = format!("{:.2}s", run.mean_seconds);
                let min_str = format!("{:.2}s", run.min_seconds);
                let speakers_str = if run.segments > 0 {
                    run.speakers.to_string()
                } else {
                    "—".to_string()
                };
                let segments_str = run.segments.to_string();
                let parity_str = match ref_rttm.as_ref() {
                    Some(_) if run.name == "pyannote MPS" => "(reference)".to_string(),
                    Some(reference) if run.segments > 0 => {
                        timeline_overlap_pct(reference, &run.rttm)
                            .map(|pct| format!("{pct:.1}%"))
                            .unwrap_or_else(|| "N/A".to_string())
                    }
                    _ => "N/A".to_string(),
                };

                println!(
                    "{:<name_w$} {:>9} {:>9} {:>9} {:>9} {:>10}  ok",
                    run.name, mean_str, min_str, speakers_str, segments_str, parity_str
                );
            }
            CompareOutcome::Skipped { name, reason } => {
                println!(
                    "{:<name_w$} {:>9} {:>9} {:>9} {:>9} {:>10}  skipped ({reason})",
                    name, "—", "—", "—", "—", "N/A"
                );
            }
            CompareOutcome::Failed { name, reason } => {
                println!(
                    "{:<name_w$} {:>9} {:>9} {:>9} {:>9} {:>10}  failed ({reason})",
                    name, "—", "—", "—", "—", "N/A"
                );
            }
        }
    }

    if results.iter().any(|result| {
        matches!(
            result,
            CompareOutcome::Completed(run) if run.name == "pyannote-rs" && run.segments == 0
        )
    }) {
        println!();
        println!("Note: pyannote-rs returned 0 segments. It only emits segments when");
        println!("speech→silence transitions occur; continuous speech files produce no output.");
    }

    Ok(())
}

fn run_compare_impl(
    name: &str,
    warmups: u32,
    runs: u32,
    runner: &CompareRunner,
) -> Result<RunResult> {
    if runs == 0 {
        bail!("runs must be greater than 0");
    }

    for _ in 0..warmups {
        runner.run_once()?;
    }

    let mut measurements = Vec::with_capacity(runs as usize);
    let mut representative_rttm = None;
    for _ in 0..runs {
        let output = runner.run_once()?;
        if representative_rttm.is_none() {
            representative_rttm = Some(output.rttm);
        }
        measurements.push(output.elapsed_seconds);
    }

    let representative_rttm = representative_rttm.unwrap_or_default();
    Ok(summarize_compare_runs(
        name,
        &measurements,
        representative_rttm,
    ))
}

fn summarize_compare_runs(name: &str, measurements: &[f64], rttm: String) -> RunResult {
    let mean_seconds = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let min_seconds = measurements.iter().copied().fold(f64::INFINITY, f64::min);
    RunResult::new(name, mean_seconds, min_seconds, rttm)
}

fn run_fluidaudio_impl(
    fluidaudio_path: &Path,
    wav_path: &str,
    timeout: Duration,
) -> Result<SingleRunOutput> {
    let json_tmp = std::env::temp_dir().join(format!("fa-{}.json", std::process::id()));
    let mut benchmark_command = Command::new("swift");
    benchmark_command
        .args(["run", "-c", "release", "--package-path"])
        .arg(fluidaudio_path)
        .args(["fluidaudiocli", "process"])
        .arg(wav_path)
        .args(["--mode", "offline", "--output"])
        .arg(&json_tmp);
    let output = capture_benchmark_cmd(&mut benchmark_command, timeout)?;

    if !json_tmp.exists() {
        bail!("fluidaudio did not produce output JSON");
    }

    let rttm = fluidaudio::json_to_rttm(&json_tmp, "file1")?;
    let _ = fs::remove_file(&json_tmp);
    Ok(SingleRunOutput {
        elapsed_seconds: output.elapsed_seconds,
        rttm,
    })
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

const IMPL_REGISTRY: &[(&str, &str, ImplType)] = &[
    ("pyannote", "pyannote MPS", ImplType::Pyannote("mps")),
    ("pyannote-cpu", "pyannote CPU", ImplType::Pyannote("cpu")),
    ("pyannote-cuda", "pyannote CUDA", ImplType::Pyannote("cuda")),
    ("coreml", "speakrs CoreML", ImplType::Speakrs("coreml")),
    (
        "coreml-fast",
        "speakrs CoreML Fast",
        ImplType::Speakrs("coreml-fast"),
    ),
    ("cuda", "speakrs CUDA", ImplType::Speakrs("cuda")),
    ("cpu", "speakrs CPU", ImplType::Speakrs("cpu")),
    ("fluidaudio", "FluidAudio", ImplType::FluidAudioBench),
    ("pyannote-rs", "pyannote-rs", ImplType::PyannoteRs),
];

pub fn der(
    dataset_id: &str,
    max_files: u32,
    max_minutes: u32,
    description: Option<&str>,
    impls: &[String],
    no_preflight: bool,
) -> Result<()> {
    if impls.len() == 1 && impls[0] == "list" {
        println!("Available implementations:");
        for (cli_id, display_name, _) in IMPL_REGISTRY {
            println!("  {cli_id:<15} {display_name}");
        }
        return Ok(());
    }

    if !impls.is_empty() {
        for id in impls {
            if id != "list" && !IMPL_REGISTRY.iter().any(|(cli_id, _, _)| cli_id == id) {
                let available: Vec<&str> = IMPL_REGISTRY.iter().map(|(id, _, _)| *id).collect();
                color_eyre::eyre::bail!(
                    "unknown implementation: {id}. Available: {}",
                    available.join(", ")
                );
            }
        }
    }

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
    let build_features = der_build_features(impls);
    cargo_build_xtask(&build_features)?;

    let needs_pyannote_rs = impls.is_empty() || impls.iter().any(|i| i == "pyannote-rs");
    if needs_pyannote_rs
        && let Err(e) = run_cmd(
            Command::new("cargo")
                .args(["build", "--release"])
                .current_dir(root.join("scripts/pyannote_rs_bench")),
        )
    {
        eprintln!("warning: pyannote-rs bench build failed (skipping): {e}");
    }

    let models_dir = root.join("fixtures/models");
    let seg_model = models_dir.join("segmentation-3.0.onnx");
    let emb_model = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    if needs_pyannote_rs {
        ensure_pyannote_rs_emb_model(&emb_model)?;
    }

    // SAFETY: single-threaded CLI, no other threads reading env vars
    unsafe { std::env::set_var("SPEAKRS_MODELS_DIR", &models_dir) };

    let fixtures_dir = root.join("fixtures/datasets");

    // pre-flight: run all implementations on the shortest file from the first dataset
    let preflight_failures: HashMap<String, String> = if no_preflight {
        HashMap::new()
    } else {
        // ensure first dataset is available for preflight
        let first_dataset = &datasets[0];
        first_dataset.ensure(&fixtures_dir)?;
        let first_dir = first_dataset.dataset_dir(&fixtures_dir);
        let preflight_files = discover_files(&first_dir, 1, f64::MAX)?;
        if preflight_files.is_empty() {
            eprintln!("warning: no files for preflight, skipping");
            HashMap::new()
        } else {
            preflight_check(&root, &preflight_files[0], &seg_model, &emb_model, impls)?
        }
    };

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
            root.join("_benchmarks").join(&run_id).join(dataset.id())
        } else {
            root.join("_benchmarks").join(&run_id)
        };
        fs::create_dir_all(&run_dir)?;

        if let Some(desc) = description {
            fs::write(run_dir.join("README.md"), format!("{desc}\n"))?;
        }

        println!(
            "Found {} files, {total_audio_minutes:.1} min total audio",
            files.len()
        );
        println!("Run ID: {run_id}");
        println!();

        let (implementations, all_results) = run_der_implementations(
            &root,
            &files,
            &seg_model,
            &emb_model,
            impls,
            total_audio_seconds,
            &preflight_failures,
        )?;

        save_der_results(
            &run_dir,
            dataset.display_name(),
            &implementations,
            &all_results,
            &files,
            total_audio_minutes,
            0.0,
            description,
            max_files,
            max_minutes,
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
    impls: &[String],
    total_audio_seconds: f64,
    preflight_failures: &HashMap<String, String>,
) -> Result<DerResults> {
    let speakrs_binary = root.join("target/release/xtask");
    let pyannote_rs_binary =
        root.join("scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs");

    let fluidaudio_bench_dir = root.join("scripts/fluidaudio-bench");
    let implementations: Vec<(&str, ImplType)> = if impls.is_empty() {
        IMPL_REGISTRY
            .iter()
            .map(|(_, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    } else {
        IMPL_REGISTRY
            .iter()
            .filter(|(cli_id, _, _)| impls.iter().any(|i| i == cli_id))
            .map(|(_, display_name, impl_type)| (*display_name, *impl_type))
            .collect()
    };

    let wav_paths: Vec<&Path> = files.iter().map(|(w, _)| w.as_path()).collect();
    let mut all_results: HashMap<String, DerImplResult> = HashMap::new();

    // 5x realtime timeout with a 2 minute minimum
    let batch_timeout = Duration::from_secs_f64((total_audio_seconds * 5.0).max(120.0));

    for (impl_name, impl_type) in &implementations {
        println!("Running {impl_name}...");

        // skip if preflight failed for this implementation
        if let Some(reason) = preflight_failures.get(*impl_name) {
            println!("  → skipped (preflight failed): {reason}");
            println!();
            all_results.insert(
                impl_name.to_string(),
                DerImplResult::failed(format!("preflight failed: {reason}")),
            );
            continue;
        }

        if let Some(reason) = der_skip_reason(
            root,
            impl_type,
            &fluidaudio_bench_dir,
            &pyannote_rs_binary,
            seg_model,
            emb_model,
        ) {
            println!("  → skipped: {reason}");
            println!();
            all_results.insert(impl_name.to_string(), DerImplResult::skipped(reason));
            continue;
        }

        let benchmark_result = match impl_type {
            ImplType::Speakrs(mode) => {
                BatchCommandRunner::speakrs(&speakrs_binary, mode, &wav_paths)
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
            ImplType::PyannoteRs => PyannoteRsFileRunner::new(
                pyannote_rs_binary.clone(),
                seg_model.to_path_buf(),
                emb_model.to_path_buf(),
            )
            .run(files),
            ImplType::Pyannote(device) => BatchCommandRunner::pyannote(root, device, &wav_paths)
                .run_with_retries(batch_timeout),
        };

        let benchmark_output = match benchmark_result {
            Ok(result) => result,
            Err(err) => {
                println!("  → failed: {err}");
                println!();
                all_results.insert(
                    impl_name.to_string(),
                    DerImplResult::failed(err.to_string()),
                );
                continue;
            }
        };

        let acc = DerAccumulation::compute(files, &benchmark_output.per_file_rttm)?;
        let (der_pct, miss_pct, fa_pct, conf_pct) = acc.der_percentages();

        let rtfx = total_audio_seconds / benchmark_output.total_seconds;
        if let Some(d) = der_pct {
            println!(
                "  → DER: {d:.1}%, Time: {:.1}s, RTFx: {rtfx:.1}x",
                benchmark_output.total_seconds
            );
        } else {
            println!(
                "  → N/A, Time: {:.1}s, RTFx: {rtfx:.1}x",
                benchmark_output.total_seconds
            );
        }
        println!();

        all_results.insert(
            impl_name.to_string(),
            DerImplResult::completed(
                der_pct,
                miss_pct,
                fa_pct,
                conf_pct,
                benchmark_output.total_seconds,
                acc.file_count,
            ),
        );
    }

    Ok((implementations, all_results))
}

#[derive(Clone, Copy)]
enum ImplType {
    Speakrs(&'static str),
    Pyannote(&'static str),
    PyannoteRs,
    FluidAudioBench,
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum DerImplStatus {
    Completed,
    Skipped,
    Failed,
}

#[derive(serde::Serialize)]
struct DerImplResult {
    status: DerImplStatus,
    reason: Option<String>,
    der: Option<f64>,
    missed: Option<f64>,
    false_alarm: Option<f64>,
    confusion: Option<f64>,
    time: Option<f64>,
    files: usize,
}

impl DerImplResult {
    fn completed(
        der: Option<f64>,
        missed: Option<f64>,
        false_alarm: Option<f64>,
        confusion: Option<f64>,
        time: f64,
        files: usize,
    ) -> Self {
        Self {
            status: DerImplStatus::Completed,
            reason: None,
            der,
            missed,
            false_alarm,
            confusion,
            time: Some(time),
            files,
        }
    }

    fn skipped(reason: String) -> Self {
        Self {
            status: DerImplStatus::Skipped,
            reason: Some(reason),
            der: None,
            missed: None,
            false_alarm: None,
            confusion: None,
            time: None,
            files: 0,
        }
    }

    fn failed(reason: String) -> Self {
        Self {
            status: DerImplStatus::Failed,
            reason: Some(reason),
            der: None,
            missed: None,
            false_alarm: None,
            confusion: None,
            time: None,
            files: 0,
        }
    }
}

struct DerAccumulation {
    missed: f64,
    false_alarm: f64,
    confusion: f64,
    total_ref: f64,
    file_count: usize,
}

impl DerAccumulation {
    fn compute(
        files: &[(PathBuf, PathBuf)],
        per_file_rttm: &HashMap<String, String>,
    ) -> Result<Self> {
        let mut acc = Self {
            missed: 0.0,
            false_alarm: 0.0,
            confusion: 0.0,
            total_ref: 0.0,
            file_count: 0,
        };

        for (wav_path, rttm_path) in files {
            let ref_text = fs::read_to_string(rttm_path)?;
            let ref_segs = speakrs::metrics::parse_rttm(&ref_text);

            let hyp_text = per_file_rttm
                .get(&wav_path.file_stem().unwrap().to_string_lossy().to_string())
                .cloned()
                .unwrap_or_default();

            if hyp_text.trim().is_empty() {
                acc.file_count += 1;
                let ref_duration: f64 = ref_segs.iter().map(|s| s.duration()).sum();
                acc.total_ref += ref_duration;
                acc.missed += ref_duration;
                continue;
            }

            let hyp_segs = speakrs::metrics::parse_rttm(&hyp_text);
            let der_result = speakrs::metrics::compute_der(&ref_segs, &hyp_segs);
            acc.missed += der_result.missed;
            acc.false_alarm += der_result.false_alarm;
            acc.confusion += der_result.confusion;
            acc.total_ref += der_result.total;
            acc.file_count += 1;
        }

        Ok(acc)
    }

    fn der_percentages(&self) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        if self.total_ref > 0.0 {
            (
                Some((self.missed + self.false_alarm + self.confusion) / self.total_ref * 100.0),
                Some(self.missed / self.total_ref * 100.0),
                Some(self.false_alarm / self.total_ref * 100.0),
                Some(self.confusion / self.total_ref * 100.0),
            )
        } else {
            (None, None, None, None)
        }
    }
}

struct BatchCommandRunner {
    command_spec: CommandSpec,
}

impl BatchCommandRunner {
    fn speakrs(binary: &Path, mode: &str, wav_paths: &[&Path]) -> Self {
        let mut command_spec = CommandSpec::new(binary.as_os_str().to_os_string())
            .arg("diarize")
            .arg("--mode")
            .arg(mode.to_string());
        for wav_path in wav_paths {
            command_spec = command_spec.arg(wav_path.as_os_str().to_os_string());
        }
        Self { command_spec }
    }

    fn pyannote(root: &Path, device: &str, wav_paths: &[&Path]) -> Self {
        let uv_path = std::env::var("HOME")
            .ok()
            .map(|home| PathBuf::from(home).join(".local/bin/uv"))
            .filter(|path| path.exists())
            .unwrap_or_else(|| "uv".into());

        let mut command_spec = CommandSpec::new(uv_path.into_os_string())
            .current_dir(root)
            .arg("run")
            .arg("scripts/diarize_pyannote.py")
            .arg("--device")
            .arg(device.to_string());
        for wav_path in wav_paths {
            command_spec = command_spec.arg(wav_path.as_os_str().to_os_string());
        }
        Self { command_spec }
    }

    fn binary(binary: &Path, wav_paths: &[&Path]) -> Self {
        let mut command_spec = CommandSpec::new(binary.as_os_str().to_os_string());
        for wav_path in wav_paths {
            command_spec = command_spec.arg(wav_path.as_os_str().to_os_string());
        }
        Self { command_spec }
    }

    fn run_with_retries(&self, timeout: Duration) -> Result<BatchRunOutput> {
        for attempt in 0..=MAX_RETRIES {
            let mut benchmark_command = self.command_spec.build_command();
            match capture_benchmark_cmd(&mut benchmark_command, timeout) {
                Ok(output) => {
                    return Ok(BatchRunOutput {
                        total_seconds: output.elapsed_seconds,
                        per_file_rttm: split_rttm_by_file_id(&output.rttm),
                    });
                }
                Err(err) => {
                    let is_timeout = err
                        .downcast_ref::<BenchmarkError>()
                        .is_some_and(|benchmark_error| benchmark_error.is_timeout());
                    if is_timeout || attempt >= MAX_RETRIES {
                        return Err(err);
                    }
                    eprintln!(
                        "  attempt {}/{} failed: {err}, retrying...",
                        attempt + 1,
                        MAX_RETRIES + 1
                    );
                }
            }
        }
        unreachable!()
    }
}

struct PyannoteRsFileRunner {
    binary: PathBuf,
    seg_model: PathBuf,
    emb_model: PathBuf,
}

impl PyannoteRsFileRunner {
    fn new(binary: PathBuf, seg_model: PathBuf, emb_model: PathBuf) -> Self {
        Self {
            binary,
            seg_model,
            emb_model,
        }
    }

    fn run(&self, files: &[(PathBuf, PathBuf)]) -> Result<BatchRunOutput> {
        let mut total_seconds = 0.0;
        let mut per_file_rttm = HashMap::new();
        let total_files = files.len();

        for (file_idx, (wav_path, _)) in files.iter().enumerate() {
            let timeout = Duration::from_secs_f64(
                (wav_duration_seconds(wav_path).unwrap_or(60.0) * 5.0).max(120.0),
            );
            let output = self.run_single(wav_path, timeout)?;
            total_seconds += output.elapsed_seconds;

            let stem = wav_path.file_stem().unwrap().to_string_lossy().to_string();
            per_file_rttm.insert(stem.clone(), output.rttm);

            let average_seconds = total_seconds / (file_idx + 1) as f64;
            let remaining_seconds = (total_files - file_idx - 1) as f64 * average_seconds;
            let eta = format_eta(remaining_seconds);
            let total_elapsed = format_eta(total_seconds);
            eprintln!(
                "  [{}/{}] {stem}: {:.1}s (elapsed {total_elapsed}, ETA {eta}) [{}]",
                file_idx + 1,
                total_files,
                output.elapsed_seconds,
                now_stamp()
            );
        }

        Ok(BatchRunOutput {
            total_seconds,
            per_file_rttm,
        })
    }

    fn run_single(&self, wav_path: &Path, timeout: Duration) -> Result<SingleRunOutput> {
        let mut benchmark_command = Command::new(&self.binary);
        benchmark_command
            .arg(wav_path)
            .arg(&self.seg_model)
            .arg(&self.emb_model);
        capture_benchmark_cmd(&mut benchmark_command, timeout)
    }
}

const MAX_RETRIES: u32 = 3;

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

    Ok(select_pairs_for_benchmark(pairs, max_files, max_minutes))
}

#[allow(clippy::too_many_arguments)]
fn save_der_results(
    run_dir: &Path,
    dataset_name: &str,
    implementations: &[(&str, ImplType)],
    results: &HashMap<String, DerImplResult>,
    files: &[(PathBuf, PathBuf)],
    total_audio_minutes: f64,
    collar: f64,
    description: Option<&str>,
    max_files: u32,
    max_minutes: u32,
) -> Result<()> {
    let seg_batch: u32 = std::env::var("PYANNOTE_SEGMENTATION_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let emb_batch: u32 = std::env::var("PYANNOTE_EMBEDDING_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    let file_list = files
        .iter()
        .map(|(w, _)| w.file_stem().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let mut json_results = serde_json::Map::new();
    for (name, _) in implementations {
        if let Some(r) = results.get(*name) {
            json_results.insert(name.to_string(), serde_json::to_value(r)?);
        }
    }

    let mut data = serde_json::json!({
        "dataset": dataset_name,
        "run_id": run_dir.file_name().unwrap().to_string_lossy(),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "files": files.len(),
        "total_audio_minutes": (total_audio_minutes * 10.0).round() / 10.0,
        "collar": collar,
        "selection_policy": "shortest_first_by_duration",
        "selection_limits": {
            "max_files": max_files,
            "max_minutes": max_minutes,
        },
        "pyannote_batch_sizes": {
            "note": "device-dependent defaults, override via PYANNOTE_*_BATCH_SIZE env vars",
            "segmentation": seg_batch,
            "embedding": emb_batch,
        },
        "file_list": file_list,
        "results": json_results,
    });

    if let Some(desc) = description {
        data["description"] = serde_json::Value::String(desc.to_string());
    }

    let json_path = run_dir.join("results.json");
    fs::write(&json_path, serde_json::to_string_pretty(&data)? + "\n")?;

    // text summary
    let mut lines = Vec::new();
    lines.push(format!(
        "{dataset_name} DER ({} files, {total_audio_minutes:.1} min, collar={collar:.0}ms)",
        files.len()
    ));
    lines.push(format!(
        "Selection: shortest-first by duration, capped at max_files={max_files}, max_minutes={max_minutes}"
    ));
    lines.push(format!(
        "pyannote batch sizes: seg={seg_batch}, emb={emb_batch}"
    ));
    lines.push(format!("Files: {}", file_list.join(", ")));
    if let Some(desc) = description {
        lines.push(format!("Description: {desc}"));
    }
    lines.push(String::new());

    let total_audio_seconds = total_audio_minutes * 60.0;
    let name_w = 22;
    let header = format!(
        "{:<name_w$} {:>8} {:>10} {:>13} {:>12} {:>8} {:>7}  {}",
        "Implementation", "DER%", "Missed%", "FalseAlarm%", "Confusion%", "Time", "RTFx", "Status"
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
            let time_str = r
                .time
                .map(|time| format!("{time:.1}s"))
                .unwrap_or_else(|| "—".to_string());
            let rtfx_str = r
                .time
                .map(|time| format!("{:.1}x", total_audio_seconds / time))
                .unwrap_or_else(|| "—".to_string());
            let status_str = match r.status {
                DerImplStatus::Completed => "ok".to_string(),
                DerImplStatus::Skipped => format!(
                    "skipped ({})",
                    r.reason.as_deref().unwrap_or("no reason recorded")
                ),
                DerImplStatus::Failed => format!(
                    "failed ({})",
                    r.reason.as_deref().unwrap_or("no reason recorded")
                ),
            };
            lines.push(format!(
                "{:<name_w$} {:>8} {:>10} {:>13} {:>12} {:>8} {:>7}  {}",
                impl_name, der_str, miss_str, fa_str, conf_str, time_str, rtfx_str, status_str
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

/// Possible failure modes for `capture_benchmark_cmd`
enum BenchmarkError {
    /// Process exceeded its time budget
    Timeout { program: String, timeout: Duration },
    /// Process exited with non-zero / signal
    ProcessFailed {
        program: String,
        status: std::process::ExitStatus,
    },
    /// Other I/O error
    Other(color_eyre::eyre::Report),
}

impl BenchmarkError {
    fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout { program, timeout } => {
                write!(f, "{program} timed out after {}s", timeout.as_secs())
            }
            Self::ProcessFailed { program, status } => {
                write!(f, "{program} failed with {status}")
            }
            Self::Other(e) => write!(f, "{e}"),
        }
    }
}

fn capture_benchmark_cmd(cmd: &mut Command, timeout: Duration) -> Result<SingleRunOutput> {
    let program = cmd.get_program().to_string_lossy().to_string();

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| BenchmarkError::Other(e.into()))?;

    // drain stdout in a background thread to prevent pipe buffer deadlock
    let stdout_handle = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut r = stdout_handle;
        r.read_to_end(&mut buf).ok();
        buf
    });

    let start = Instant::now();
    match child
        .wait_timeout(timeout)
        .map_err(|e| BenchmarkError::Other(e.into()))?
    {
        Some(status) => {
            let elapsed = start.elapsed().as_secs_f64();
            if !status.success() {
                return Err(BenchmarkError::ProcessFailed { program, status }.into());
            }
            let stdout_bytes = reader.join().unwrap_or_default();
            Ok(SingleRunOutput {
                elapsed_seconds: elapsed,
                rttm: String::from_utf8_lossy(&stdout_bytes).to_string(),
            })
        }
        None => {
            // timed out — kill and reap
            let _ = child.kill();
            let _ = child.wait();
            drop(reader);
            Err(BenchmarkError::Timeout { program, timeout }.into())
        }
    }
}

impl std::fmt::Debug for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for BenchmarkError {}

/// Derive cargo features needed for the selected DER implementations
fn der_build_features(impls: &[String]) -> Vec<String> {
    let active_impls: Vec<&ImplType> = if impls.is_empty() {
        IMPL_REGISTRY.iter().map(|(_, _, t)| t).collect()
    } else {
        IMPL_REGISTRY
            .iter()
            .filter(|(cli_id, _, _)| impls.iter().any(|i| i == cli_id))
            .map(|(_, _, t)| t)
            .collect()
    };

    let mut features = Vec::new();

    #[cfg(target_os = "macos")]
    let needs_coreml = active_impls
        .iter()
        .any(|t| matches!(t, ImplType::Speakrs(m) if m.starts_with("coreml")));
    #[cfg(not(target_os = "macos"))]
    let needs_coreml = false;

    let needs_cuda = active_impls
        .iter()
        .any(|t| matches!(t, ImplType::Speakrs("cuda")));

    if needs_coreml {
        features.push("coreml".to_string());
    }
    if needs_cuda {
        features.push("cuda".to_string());
    }

    features
}

const PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(180);

/// Run each selected implementation on a single short file as a smoke test
fn preflight_check(
    root: &Path,
    file: &(PathBuf, PathBuf),
    seg_model: &Path,
    emb_model: &Path,
    impls: &[String],
) -> Result<HashMap<String, String>> {
    let (wav_path, _rttm_path) = file;
    let stem = wav_path.file_stem().unwrap().to_string_lossy().to_string();
    let dur = wav_duration_seconds(wav_path).unwrap_or(0.0);
    println!();
    println!("=== Pre-flight check ({stem}, {dur:.0}s) ===");

    let speakrs_binary = root.join("target/release/xtask");
    let pyannote_rs_binary =
        root.join("scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs");
    let fluidaudio_bench_dir = root.join("scripts/fluidaudio-bench");

    let implementations: Vec<(&str, &str, ImplType)> = if impls.is_empty() {
        IMPL_REGISTRY
            .iter()
            .map(|(cli_id, display_name, impl_type)| (*cli_id, *display_name, *impl_type))
            .collect()
    } else {
        IMPL_REGISTRY
            .iter()
            .filter(|(cli_id, _, _)| impls.iter().any(|i| i == cli_id))
            .map(|(cli_id, display_name, impl_type)| (*cli_id, *display_name, *impl_type))
            .collect()
    };

    let wav_paths: Vec<&Path> = vec![wav_path.as_path()];
    let mut failures: HashMap<String, String> = HashMap::new();

    for (_cli_id, display_name, impl_type) in &implementations {
        if let Some(reason) = der_skip_reason(
            root,
            impl_type,
            &fluidaudio_bench_dir,
            &pyannote_rs_binary,
            seg_model,
            emb_model,
        ) {
            println!("  {display_name:<22} skipped ({reason})");
            continue;
        }

        let result = match impl_type {
            ImplType::Speakrs(mode) => {
                BatchCommandRunner::speakrs(&speakrs_binary, mode, &wav_paths)
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
            ImplType::PyannoteRs => PyannoteRsFileRunner::new(
                pyannote_rs_binary.clone(),
                seg_model.to_path_buf(),
                emb_model.to_path_buf(),
            )
            .run(&[(wav_path.clone(), PathBuf::new())]),
            ImplType::Pyannote(device) => BatchCommandRunner::pyannote(root, device, &wav_paths)
                .run_with_retries(PREFLIGHT_TIMEOUT),
        };

        match result {
            Ok(batch_output) => {
                let has_output = batch_output
                    .per_file_rttm
                    .values()
                    .any(|v| !v.trim().is_empty());
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

fn format_eta(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.0}s")
    } else {
        let mins = (seconds / 60.0).floor() as u64;
        let secs = (seconds % 60.0).round() as u64;
        format!("{mins}m {secs:02}s")
    }
}

fn now_stamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

fn der_skip_reason(
    root: &Path,
    impl_type: &ImplType,
    fluidaudio_bench_dir: &Path,
    pyannote_rs_binary: &Path,
    seg_model: &Path,
    emb_model: &Path,
) -> Option<String> {
    match impl_type {
        ImplType::Pyannote(_) => (!root.join("scripts/diarize_pyannote.py").exists())
            .then(|| "scripts/diarize_pyannote.py not found".to_string()),
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

fn select_pairs_for_benchmark(
    mut pairs: Vec<(PathBuf, PathBuf, f64)>,
    max_files: u32,
    max_minutes: f64,
) -> Vec<(PathBuf, PathBuf)> {
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

    selected
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_compare_runs_uses_mean_and_min() {
        let result = summarize_compare_runs(
            "speakrs CoreML",
            &[6.0, 4.0, 5.0],
            "SPEAKER file1 1 0.000000 1.000000 <NA> <NA> spk1 <NA> <NA>\n".to_string(),
        );

        assert_eq!(result.name, "speakrs CoreML");
        assert_eq!(result.mean_seconds, 5.0);
        assert_eq!(result.min_seconds, 4.0);
        assert_eq!(result.segments, 1);
        assert_eq!(result.speakers, 1);
    }

    #[test]
    fn select_pairs_for_benchmark_prefers_shortest_files() {
        let pairs = vec![
            (PathBuf::from("long.wav"), PathBuf::from("long.rttm"), 180.0),
            (
                PathBuf::from("short.wav"),
                PathBuf::from("short.rttm"),
                30.0,
            ),
            (
                PathBuf::from("medium.wav"),
                PathBuf::from("medium.rttm"),
                60.0,
            ),
        ];

        let selected = select_pairs_for_benchmark(pairs, 2, 2.0);
        let names = selected
            .iter()
            .map(|(wav, _)| wav.file_stem().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["short".to_string(), "medium".to_string()]);
    }

    #[test]
    fn split_rttm_by_file_id_groups_lines() {
        let grouped = split_rttm_by_file_id(concat!(
            "SPEAKER a 1 0.0 1.0 <NA> <NA> spk1 <NA> <NA>\n",
            "SPEAKER b 1 0.5 1.0 <NA> <NA> spk2 <NA> <NA>\n",
            "SPEAKER a 1 1.0 0.5 <NA> <NA> spk1 <NA> <NA>\n",
        ));

        assert_eq!(
            grouped.get("a").unwrap(),
            "SPEAKER a 1 0.0 1.0 <NA> <NA> spk1 <NA> <NA>\n\
SPEAKER a 1 1.0 0.5 <NA> <NA> spk1 <NA> <NA>\n"
        );
        assert_eq!(
            grouped.get("b").unwrap(),
            "SPEAKER b 1 0.5 1.0 <NA> <NA> spk2 <NA> <NA>\n"
        );
    }
}
