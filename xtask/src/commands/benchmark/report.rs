use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;

use super::*;
use crate::path::file_stem_string;

pub struct DerResultsWriter<'a> {
    pub run_dir: &'a Path,
    pub dataset_name: &'a str,
    pub implementations: &'a [(&'static str, ImplType)],
    pub results: &'a HashMap<String, DerImplResult>,
    pub files: &'a [(PathBuf, PathBuf)],
    pub total_audio_minutes: f64,
    pub collar: f64,
    pub description: Option<&'a str>,
    pub max_files: u32,
    pub max_minutes: u32,
    pub metadata: &'a BenchmarkMetadata,
    pub pyannote_batch_sizes: PyannoteBatchSizes,
}

impl<'a> DerResultsWriter<'a> {
    pub fn write(&self) -> Result<()> {
        let json_payload = self.json_payload()?;
        fs::write(
            self.run_dir.join("results.json"),
            serde_json::to_string_pretty(&json_payload)? + "\n",
        )?;

        let summary_lines = self.summary_lines()?;
        fs::write(
            self.run_dir.join("results.txt"),
            summary_lines.join("\n") + "\n",
        )?;

        println!("\nResults saved to {}/", self.run_dir.display());
        for line in &summary_lines {
            println!("{line}");
        }

        Ok(())
    }

    fn json_payload(&self) -> Result<serde_json::Value> {
        let file_list = self.file_list()?;
        let mut json_results = serde_json::Map::new();

        for (implementation_name, _) in self.implementations {
            if let Some(result) = self.results.get(*implementation_name) {
                json_results.insert(
                    implementation_name.to_string(),
                    serde_json::to_value(result)?,
                );
            }
        }

        let metadata = self.metadata;
        let run_id = self
            .run_dir
            .file_name()
            .ok_or_else(|| color_eyre::eyre::eyre!("run dir is missing a final path component"))?
            .to_string_lossy()
            .to_string();
        let mut payload = serde_json::json!({
            "dataset": self.dataset_name,
            "run_id": run_id,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "git_sha": metadata.git_sha,
            "gpu": metadata.gpu,
            "cpu": metadata.cpu,
            "region": metadata.region,
            "files": self.files.len(),
            "total_audio_minutes": (self.total_audio_minutes * 10.0).round() / 10.0,
            "collar": self.collar,
            "selection_policy": "shortest_first_by_duration",
            "selection_limits": {
                "max_files": self.max_files,
                "max_minutes": self.max_minutes,
            },
            "pyannote_batch_sizes": {
                "note": "null means the script default (cuda=32, non-cuda=16)",
                "segmentation_override": self.pyannote_batch_sizes.segmentation,
                "embedding_override": self.pyannote_batch_sizes.embedding,
            },
            "file_list": file_list,
            "results": json_results,
        });

        if let Some(description) = self.description {
            payload["description"] = serde_json::Value::String(description.to_string());
        }

        Ok(payload)
    }

    fn summary_lines(&self) -> Result<Vec<String>> {
        let (seg_batch_size, emb_batch_size) = self.pyannote_batch_sizes.summary_values();
        let file_list = self.file_list()?;
        let total_audio_seconds = self.total_audio_minutes * 60.0;
        let name_width = 22;
        let header = format!(
            "{:<name_width$} {:>8} {:>10} {:>13} {:>12} {:>8} {:>7}  {}",
            "Implementation",
            "DER%",
            "Missed%",
            "FalseAlarm%",
            "Confusion%",
            "Time",
            "RTFx",
            "Status"
        );

        let metadata = self.metadata;
        let mut lines = vec![
            format!(
                "{} DER ({} files, {:.1} min, collar={:.0}ms)",
                self.dataset_name,
                self.files.len(),
                self.total_audio_minutes,
                self.collar
            ),
            format!(
                "Commit: {}  GPU: {}  CPU: {}  Region: {}",
                metadata.git_sha, metadata.gpu, metadata.cpu, metadata.region
            ),
            format!(
                "Selection: greedy by duration, executed in input order, capped at max_files={}, max_minutes={}",
                self.max_files, self.max_minutes
            ),
            format!("pyannote batch sizes: seg={seg_batch_size}, emb={emb_batch_size}"),
            format!("Files: {}", file_list.join(", ")),
        ];
        if let Some(description) = self.description {
            lines.push(format!("Description: {description}"));
        }
        lines.push(String::new());
        lines.push(header.clone());
        lines.push("─".repeat(header.len()));

        for (implementation_name, _) in self.implementations {
            if let Some(result) = self.results.get(*implementation_name) {
                let (der_str, missed_str, false_alarm_str, confusion_str) = match result.der {
                    Some(der) => (
                        format!("{der:.1}%"),
                        format!("{:.1}%", result.missed.unwrap_or(0.0)),
                        format!("{:.1}%", result.false_alarm.unwrap_or(0.0)),
                        format!("{:.1}%", result.confusion.unwrap_or(0.0)),
                    ),
                    None => (
                        "N/A".to_string(),
                        "—".to_string(),
                        "—".to_string(),
                        "—".to_string(),
                    ),
                };
                let time_str = result
                    .time
                    .map(|time| format!("{time:.1}s"))
                    .unwrap_or_else(|| "—".to_string());
                let rtfx_str = result
                    .time
                    .map(|time| format!("{:.1}x", total_audio_seconds / time))
                    .unwrap_or_else(|| "—".to_string());
                let status_str = match result.status {
                    DerImplStatus::Completed => "ok".to_string(),
                    DerImplStatus::Skipped => format!(
                        "skipped ({})",
                        result.reason.as_deref().unwrap_or("no reason recorded")
                    ),
                    DerImplStatus::Failed => format!(
                        "failed ({})",
                        result.reason.as_deref().unwrap_or("no reason recorded")
                    ),
                };
                lines.push(format!(
                    "{:<name_width$} {:>8} {:>10} {:>13} {:>12} {:>8} {:>7}  {}",
                    implementation_name,
                    der_str,
                    missed_str,
                    false_alarm_str,
                    confusion_str,
                    time_str,
                    rtfx_str,
                    status_str
                ));
            }
        }

        Ok(lines)
    }

    fn file_list(&self) -> Result<Vec<String>> {
        self.files
            .iter()
            .map(|(wav_path, _)| file_stem_string(wav_path))
            .collect()
    }
}

pub fn format_eta(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.0}s")
    } else {
        let mins = (seconds / 60.0).floor() as u64;
        let secs = (seconds % 60.0).round() as u64;
        format!("{mins}m {secs:02}s")
    }
}

pub fn now_stamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}
