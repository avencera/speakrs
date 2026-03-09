use std::fs;
use std::path::Path;

use color_eyre::eyre::{Result, bail};
use serde::Deserialize;

#[derive(Deserialize)]
struct FluidAudioOutput {
    segments: Vec<FluidAudioSegment>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FluidAudioSegment {
    start_time_seconds: f64,
    end_time_seconds: f64,
    speaker_id: serde_json::Value,
}

/// Convert a FluidAudio JSON result file into RTTM text
pub fn json_to_rttm(json_path: &Path, file_id: &str) -> Result<String> {
    let content = fs::read_to_string(json_path)?;
    let output: FluidAudioOutput = serde_json::from_str(&content)?;

    if output.segments.is_empty() {
        bail!("FluidAudio JSON contains no segments");
    }

    let mut lines = Vec::with_capacity(output.segments.len());
    for seg in &output.segments {
        let start = seg.start_time_seconds;
        let duration = (seg.end_time_seconds - start).max(0.0);
        let speaker = match &seg.speaker_id {
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        lines.push(format!(
            "SPEAKER {file_id} 1 {start:.6} {duration:.6} <NA> <NA> {speaker} <NA> <NA>"
        ));
    }

    Ok(lines.join("\n") + "\n")
}
