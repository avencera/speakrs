use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use color_eyre::eyre::Result;

use crate::cmd::wav_duration_seconds;
use crate::path::file_stem_string;

use super::report::{format_eta, now_stamp};
use super::runner::{
    BatchRunOutput, BenchmarkError, CommandSpec, SingleRunOutput, capture_benchmark_cmd,
};

pub struct BenchmarkMetadata {
    pub git_sha: String,
    pub gpu: String,
    pub cpu: String,
    pub region: String,
}

impl BenchmarkMetadata {
    pub fn collect() -> Self {
        Self {
            git_sha: env!("GIT_SHA").to_string(),
            gpu: detect_gpu(),
            cpu: detect_cpu(),
            region: std::env::var("DSTACK_REGION").unwrap_or_else(|_| "local".to_string()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PyannoteBatchSizes {
    pub segmentation: Option<u32>,
    pub embedding: Option<u32>,
}

impl PyannoteBatchSizes {
    pub fn from_overrides(segmentation: Option<u32>, embedding: Option<u32>) -> Self {
        Self {
            segmentation,
            embedding,
        }
    }

    pub fn summary_values(self) -> (String, String) {
        (
            self.segmentation
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".to_string()),
            self.embedding
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".to_string()),
        )
    }
}

#[derive(Clone, Copy)]
pub enum ImplType {
    Speakrs(&'static str),
    Pyannote(&'static str),
    PyannoteRs,
    FluidAudioBench,
    SpeakerKitBench,
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DerImplStatus {
    Completed,
    Skipped,
    Failed,
}

#[derive(serde::Serialize)]
pub struct DerImplResult {
    pub status: DerImplStatus,
    pub reason: Option<String>,
    pub der: Option<f64>,
    pub missed: Option<f64>,
    pub false_alarm: Option<f64>,
    pub confusion: Option<f64>,
    pub time: Option<f64>,
    pub files: usize,
}

impl DerImplResult {
    pub fn completed(
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

    pub fn skipped(reason: String) -> Self {
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

    pub fn failed(reason: String) -> Self {
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

pub struct DerAccumulation {
    pub missed: f64,
    pub false_alarm: f64,
    pub confusion: f64,
    pub total_ref: f64,
    pub file_count: usize,
}

impl DerAccumulation {
    pub fn compute(
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
            let stem = file_stem_string(wav_path)?;

            let hyp_text = per_file_rttm.get(&stem).cloned().unwrap_or_default();

            if hyp_text.trim().is_empty() {
                acc.file_count += 1;
                let ref_duration: f64 = ref_segs.iter().map(|segment| segment.duration()).sum();
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

    pub fn der_percentages(&self) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
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

pub struct BatchCommandRunner {
    command_spec: CommandSpec,
}

impl BatchCommandRunner {
    pub fn speakrs(binary: &Path, mode: &str, models_dir: &Path, wav_paths: &[&Path]) -> Self {
        let mut command_spec = CommandSpec::new(binary.as_os_str().to_os_string())
            .arg("diarize")
            .arg("--mode")
            .arg(mode.to_string())
            .arg("--models-dir")
            .arg(models_dir.as_os_str().to_os_string());
        for wav_path in wav_paths {
            command_spec = command_spec.arg(wav_path.as_os_str().to_os_string());
        }
        Self { command_spec }
    }

    pub fn pyannote(
        root: &Path,
        device: &str,
        wav_paths: &[&Path],
        batch_sizes: PyannoteBatchSizes,
    ) -> Self {
        let uv_path = std::env::var("HOME")
            .ok()
            .map(|home| PathBuf::from(home).join(".local/bin/uv"))
            .filter(|path| path.exists())
            .unwrap_or_else(|| "uv".into());

        let mut command_spec = CommandSpec::new(uv_path.into_os_string())
            .current_dir(root)
            .arg("run")
            .arg("--project")
            .arg("scripts/pyannote-bench")
            .arg("python")
            .arg("scripts/pyannote-bench/diarize.py")
            .arg("--device")
            .arg(device.to_string());
        if let Some(segmentation) = batch_sizes.segmentation {
            command_spec = command_spec
                .env("PYANNOTE_SEGMENTATION_BATCH_SIZE", segmentation.to_string())
                .arg("--segmentation-batch-size")
                .arg(segmentation.to_string());
        }
        if let Some(embedding) = batch_sizes.embedding {
            command_spec = command_spec
                .env("PYANNOTE_EMBEDDING_BATCH_SIZE", embedding.to_string())
                .arg("--embedding-batch-size")
                .arg(embedding.to_string());
        }
        for wav_path in wav_paths {
            command_spec = command_spec.arg(wav_path.as_os_str().to_os_string());
        }
        Self { command_spec }
    }

    pub fn binary(binary: &Path, wav_paths: &[&Path]) -> Self {
        let mut command_spec = CommandSpec::new(binary.as_os_str().to_os_string());
        for wav_path in wav_paths {
            command_spec = command_spec.arg(wav_path.as_os_str().to_os_string());
        }
        Self { command_spec }
    }

    pub fn run_with_retries(&self, timeout: Duration) -> Result<BatchRunOutput> {
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

pub(crate) struct PyannoteRsFileRunner {
    binary: PathBuf,
    seg_model: PathBuf,
    emb_model: PathBuf,
}

impl PyannoteRsFileRunner {
    pub(crate) fn new(binary: PathBuf, seg_model: PathBuf, emb_model: PathBuf) -> Self {
        Self {
            binary,
            seg_model,
            emb_model,
        }
    }

    pub(crate) fn run(&self, files: &[(PathBuf, PathBuf)]) -> Result<BatchRunOutput> {
        let mut total_seconds = 0.0;
        let mut per_file_rttm = HashMap::new();
        let total_files = files.len();

        for (file_idx, (wav_path, _)) in files.iter().enumerate() {
            let timeout = Duration::from_secs_f64(
                (wav_duration_seconds(wav_path).unwrap_or(60.0) * 5.0).max(120.0),
            );
            let output = self.run_single(wav_path, timeout)?;
            total_seconds += output.elapsed_seconds;

            let stem = file_stem_string(wav_path)?;
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

pub const MAX_RETRIES: u32 = 3;
pub const PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(180);

pub fn split_rttm_by_file_id(stdout: &str) -> HashMap<String, String> {
    let mut per_file: HashMap<String, Vec<&str>> = HashMap::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.first() == Some(&"SPEAKER") && parts.len() >= 2 {
            per_file.entry(parts[1].to_string()).or_default().push(line);
        }
    }
    per_file
        .into_iter()
        .map(|(file_id, lines)| (file_id, lines.join("\n") + "\n"))
        .collect()
}

fn detect_gpu() -> String {
    Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            stdout.lines().next().map(|line| line.trim().to_string())
        })
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn detect_cpu() -> String {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split_once(':'))
                    .map(|(_, value)| value.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    #[cfg(not(target_os = "linux"))]
    {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    }
}
