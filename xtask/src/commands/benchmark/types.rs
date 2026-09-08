use std::collections::{HashMap, HashSet};
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

/// Per-file DER components, denominator, and speaker counts.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct PerFileDerResult {
    /// File identifier used by the dataset manifest and RTTM rows.
    pub file_id: String,
    /// Reference RTTM file used to calculate this score.
    pub reference_rttm: PathBuf,
    /// Total reference speaker-time used as the DER denominator.
    pub reference_speaker_time: f64,
    /// Missed reference speaker-time in seconds.
    pub missed: f64,
    /// False-alarm speaker-time in seconds.
    pub false_alarm: f64,
    /// Confused speaker-time in seconds.
    pub confusion: f64,
    /// Total DER percentage, or `None` when reference speaker-time is zero.
    pub der: Option<f64>,
    /// Missed-speech percentage, or `None` when reference speaker-time is zero.
    pub missed_percent: Option<f64>,
    /// False-alarm percentage, or `None` when reference speaker-time is zero.
    pub false_alarm_percent: Option<f64>,
    /// Speaker-confusion percentage, or `None` when reference speaker-time is zero.
    pub confusion_percent: Option<f64>,
    /// Number of unique reference speakers.
    pub reference_speaker_count: usize,
    /// Number of unique predicted speakers.
    pub predicted_speaker_count: usize,
}

impl PerFileDerResult {
    fn from_segments(
        file_id: String,
        reference_rttm: PathBuf,
        reference: &[speakrs::Segment],
        hypothesis: &[speakrs::Segment],
        der_result: speakrs::metrics::DerResult,
    ) -> Self {
        let reference_speaker_time = der_result.total;
        let error_duration = der_result.missed + der_result.false_alarm + der_result.confusion;

        Self {
            file_id,
            reference_rttm,
            reference_speaker_time,
            missed: der_result.missed,
            false_alarm: der_result.false_alarm,
            confusion: der_result.confusion,
            der: percentage(error_duration, reference_speaker_time),
            missed_percent: percentage(der_result.missed, reference_speaker_time),
            false_alarm_percent: percentage(der_result.false_alarm, reference_speaker_time),
            confusion_percent: percentage(der_result.confusion, reference_speaker_time),
            reference_speaker_count: count_speakers(reference),
            predicted_speaker_count: count_speakers(hypothesis),
        }
    }
}

pub struct DerAccumulation {
    pub missed: f64,
    pub false_alarm: f64,
    pub confusion: f64,
    pub total_ref: f64,
    pub file_count: usize,
    per_file_results: Vec<PerFileDerResult>,
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
            per_file_results: Vec::with_capacity(files.len()),
        };

        for (wav_path, rttm_path) in files {
            let ref_text = fs::read_to_string(rttm_path)?;
            let ref_segs = speakrs::metrics::parse_rttm(&ref_text);
            let stem = file_stem_string(wav_path)?;

            let hyp_text = per_file_rttm.get(&stem).cloned().unwrap_or_default();
            let hyp_segs = speakrs::metrics::parse_rttm(&hyp_text);
            let der_result = speakrs::metrics::compute_der(&ref_segs, &hyp_segs);

            let file_result = PerFileDerResult::from_segments(
                stem,
                rttm_path.clone(),
                &ref_segs,
                &hyp_segs,
                der_result,
            );
            acc.missed += file_result.missed;
            acc.false_alarm += file_result.false_alarm;
            acc.confusion += file_result.confusion;
            acc.total_ref += file_result.reference_speaker_time;
            acc.file_count += 1;
            acc.per_file_results.push(file_result);
        }

        Ok(acc)
    }

    /// Return per-file scores in the same order as the input manifest.
    pub fn per_file(&self) -> &[PerFileDerResult] {
        &self.per_file_results
    }

    pub fn der_percentages(&self) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        (
            percentage(
                self.missed + self.false_alarm + self.confusion,
                self.total_ref,
            ),
            percentage(self.missed, self.total_ref),
            percentage(self.false_alarm, self.total_ref),
            percentage(self.confusion, self.total_ref),
        )
    }
}

fn count_speakers(segments: &[speakrs::Segment]) -> usize {
    segments
        .iter()
        .map(|segment| &segment.speaker)
        .collect::<HashSet<_>>()
        .len()
}

fn percentage(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then_some(numerator / denominator * 100.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_file_results_serialize_der_components_and_counts() {
        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("first.rttm");
        fs::write(
            &reference_path,
            "SPEAKER first 1 0.0 2.0 <NA> <NA> ref-a <NA> <NA>\n",
        )
        .unwrap();

        let files = vec![(PathBuf::from("first.wav"), reference_path.clone())];
        let hypotheses = HashMap::from([(
            "first".to_string(),
            "SPEAKER first 1 0.0 1.0 <NA> <NA> hyp-a <NA> <NA>\n\
SPEAKER first 1 1.0 1.0 <NA> <NA> hyp-b <NA> <NA>\n"
                .to_string(),
        )]);

        let accumulation = DerAccumulation::compute(&files, &hypotheses).unwrap();
        let result = &accumulation.per_file_results[0];

        assert_eq!(result.file_id, "first");
        assert_eq!(result.reference_rttm, reference_path);
        assert_eq!(result.reference_speaker_time, 2.0);
        assert_eq!(result.missed, 0.0);
        assert_eq!(result.false_alarm, 0.0);
        assert_eq!(result.confusion, 1.0);
        assert_eq!(result.der, Some(50.0));
        assert_eq!(result.missed_percent, Some(0.0));
        assert_eq!(result.false_alarm_percent, Some(0.0));
        assert_eq!(result.confusion_percent, Some(50.0));
        assert_eq!(result.reference_speaker_count, 1);
        assert_eq!(result.predicted_speaker_count, 2);

        let json = serde_json::to_value(result).unwrap();
        assert_eq!(json["file_id"], "first");
        assert!(
            json["reference_rttm"]
                .as_str()
                .unwrap()
                .ends_with("first.rttm")
        );
        assert_eq!(json["reference_speaker_time"], 2.0);
        assert_eq!(json["der"], 50.0);
        assert_eq!(json["reference_speaker_count"], 1);
        assert_eq!(json["predicted_speaker_count"], 2);
    }

    #[test]
    fn aggregate_totals_equal_per_file_totals_with_empty_hypothesis() {
        let directory = tempfile::tempdir().unwrap();
        let first_reference_path = directory.path().join("first.rttm");
        let second_reference_path = directory.path().join("second.rttm");
        fs::write(
            &first_reference_path,
            "SPEAKER first 1 0.0 2.0 <NA> <NA> ref-a <NA> <NA>\n",
        )
        .unwrap();
        fs::write(
            &second_reference_path,
            concat!(
                "SPEAKER second 1 0.0 1.0 <NA> <NA> ref-a <NA> <NA>\n",
                "SPEAKER second 1 1.0 1.0 <NA> <NA> ref-b <NA> <NA>\n",
            ),
        )
        .unwrap();

        let files = vec![
            (PathBuf::from("first.wav"), first_reference_path),
            (PathBuf::from("second.wav"), second_reference_path),
        ];
        let hypotheses = HashMap::from([(
            "first".to_string(),
            "SPEAKER first 1 0.0 1.0 <NA> <NA> hyp-a <NA> <NA>\n\
SPEAKER first 1 1.0 1.0 <NA> <NA> hyp-b <NA> <NA>\n"
                .to_string(),
        )]);

        let accumulation = DerAccumulation::compute(&files, &hypotheses).unwrap();
        let file_results = accumulation.per_file();

        assert_eq!(file_results.len(), 2);
        assert_eq!(file_results[1].file_id, "second");
        assert_eq!(file_results[1].missed, 2.0);
        assert_eq!(file_results[1].false_alarm, 0.0);
        assert_eq!(file_results[1].confusion, 0.0);
        assert_eq!(file_results[1].reference_speaker_time, 2.0);
        assert_eq!(file_results[1].der, Some(100.0));
        assert_eq!(file_results[1].reference_speaker_count, 2);
        assert_eq!(file_results[1].predicted_speaker_count, 0);

        let missed: f64 = file_results.iter().map(|result| result.missed).sum();
        let false_alarm: f64 = file_results.iter().map(|result| result.false_alarm).sum();
        let confusion: f64 = file_results.iter().map(|result| result.confusion).sum();
        let reference_speaker_time: f64 = file_results
            .iter()
            .map(|result| result.reference_speaker_time)
            .sum();

        assert!((accumulation.missed - missed).abs() < f64::EPSILON);
        assert!((accumulation.false_alarm - false_alarm).abs() < f64::EPSILON);
        assert!((accumulation.confusion - confusion).abs() < f64::EPSILON);
        assert!((accumulation.total_ref - reference_speaker_time).abs() < f64::EPSILON);
        assert_eq!(accumulation.file_count, file_results.len());
    }

    #[test]
    fn zero_reference_duration_has_no_percentages() {
        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("empty.rttm");
        fs::write(&reference_path, "").unwrap();

        let files = vec![(PathBuf::from("empty.wav"), reference_path)];
        let accumulation = DerAccumulation::compute(&files, &HashMap::new()).unwrap();
        let result = &accumulation.per_file_results[0];

        assert_eq!(result.reference_speaker_time, 0.0);
        assert_eq!(result.der, None);
        assert_eq!(result.missed_percent, None);
        assert_eq!(result.false_alarm_percent, None);
        assert_eq!(result.confusion_percent, None);
        assert_eq!(accumulation.der_percentages(), (None, None, None, None));
    }
}
