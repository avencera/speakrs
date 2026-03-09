use std::path::Path;
use std::process::{Command, Stdio};

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;

/// Convert any audio file to 16kHz mono WAV using ffmpeg.
/// No-op if output already exists
pub fn convert_to_16k_mono(input: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        return Ok(());
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    run_cmd(
        Command::new("ffmpeg")
            .args(["-y", "-i"])
            .arg(input)
            .args(["-ar", "16000", "-ac", "1"])
            .arg(output)
            .stdout(Stdio::null())
            .stderr(Stdio::null()),
    )
}

/// Parse a Praat TextGrid file and write RTTM format.
/// Handles both long-form and short-form TextGrid formats
pub fn textgrid_to_rttm(textgrid_path: &Path, rttm_path: &Path, file_id: &str) -> Result<()> {
    let content = std::fs::read_to_string(textgrid_path)?;
    let intervals = parse_textgrid_intervals(&content)?;

    let mut lines = Vec::new();
    for (start, end, speaker) in &intervals {
        let duration = end - start;
        if duration <= 0.0 {
            continue;
        }
        lines.push(format!(
            "SPEAKER {file_id} 1 {start:.3} {duration:.3} <NA> <NA> {speaker} <NA> <NA>"
        ));
    }

    if let Some(parent) = rttm_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(rttm_path, lines.join("\n") + "\n")?;
    Ok(())
}

const SILENCE_LABELS: &[&str] = &["", "sil", "sp", "spn", "noise", "<sil>", "<noise>"];

fn is_silence_label(label: &str) -> bool {
    let lower = label.to_lowercase();
    SILENCE_LABELS.iter().any(|s| *s == lower)
}

/// Parse intervals from a TextGrid file, returning (start, end, speaker) tuples
fn parse_textgrid_intervals(content: &str) -> Result<Vec<(f64, f64, String)>> {
    let lines: Vec<&str> = content.lines().collect();

    // detect short-form TextGrid (no "xmin =" lines, just raw values)
    let is_short_form = !content.contains("xmin =");

    if is_short_form {
        parse_textgrid_short(&lines)
    } else {
        parse_textgrid_long(&lines)
    }
}

/// Parse long-form TextGrid (with "xmin = ...", "xmax = ...", "text = ..." lines)
fn parse_textgrid_long(lines: &[&str]) -> Result<Vec<(f64, f64, String)>> {
    let mut intervals = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // look for interval entries
        if line.starts_with("xmin =") {
            let xmin = parse_value(line, "xmin =")?;
            i += 1;
            if i >= lines.len() {
                break;
            }
            let xmax_line = lines[i].trim();
            if !xmax_line.starts_with("xmax =") {
                i += 1;
                continue;
            }
            let xmax = parse_value(xmax_line, "xmax =")?;
            i += 1;
            if i >= lines.len() {
                break;
            }
            let text_line = lines[i].trim();
            if !text_line.starts_with("text =") {
                i += 1;
                continue;
            }
            let text = parse_text_value(text_line);
            if !is_silence_label(&text) {
                intervals.push((xmin, xmax, text));
            }
        }
        i += 1;
    }

    Ok(intervals)
}

/// Parse short-form TextGrid (raw values without field names)
fn parse_textgrid_short(lines: &[&str]) -> Result<Vec<(f64, f64, String)>> {
    let mut intervals = Vec::new();
    let mut i = 0;

    // find "IntervalTier" markers and parse following intervals
    while i < lines.len() {
        let line = lines[i].trim().trim_matches('"');
        if line == "IntervalTier" {
            // skip: tier name, xmin, xmax
            i += 1; // tier name
            i += 1; // tier xmin
            i += 1; // tier xmax
            i += 1; // now at interval count
            if i >= lines.len() {
                break;
            }
            let count: usize = lines[i].trim().parse().unwrap_or(0);
            i += 1;

            for _ in 0..count {
                if i + 2 >= lines.len() {
                    break;
                }
                let xmin: f64 = lines[i].trim().parse().unwrap_or(0.0);
                i += 1;
                let xmax: f64 = lines[i].trim().parse().unwrap_or(0.0);
                i += 1;
                let text = lines[i].trim().trim_matches('"').to_string();
                i += 1;

                if !is_silence_label(&text) && xmax > xmin {
                    intervals.push((xmin, xmax, text));
                }
            }
        } else {
            i += 1;
        }
    }

    Ok(intervals)
}

fn parse_value(line: &str, prefix: &str) -> Result<f64> {
    let val_str = line
        .strip_prefix(prefix)
        .unwrap_or(line)
        .trim()
        .trim_end_matches(char::is_whitespace);
    val_str
        .parse::<f64>()
        .map_err(|e| color_eyre::eyre::eyre!("failed to parse '{val_str}' as f64: {e}"))
}

fn parse_text_value(line: &str) -> String {
    let after_eq = line
        .strip_prefix("text =")
        .or_else(|| line.strip_prefix("text="))
        .unwrap_or(line)
        .trim();
    // remove surrounding quotes
    after_eq.trim_matches('"').trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_long_form_textgrid() {
        let content = r#"File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "speaker"
        xmin = 0
        xmax = 10.0
        intervals: size = 3
        intervals [1]:
            xmin = 0.0
            xmax = 2.5
            text = "spk1"
        intervals [2]:
            xmin = 2.5
            xmax = 5.0
            text = ""
        intervals [3]:
            xmin = 5.0
            xmax = 8.0
            text = "spk2"
"#;
        let intervals = parse_textgrid_intervals(content).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0], (0.0, 2.5, "spk1".to_string()));
        assert_eq!(intervals[1], (5.0, 8.0, "spk2".to_string()));
    }

    #[test]
    fn parse_short_form_textgrid() {
        let content = r#"File type = "ooTextFile"
Object class = "TextGrid"

0
10.0
<exists>
1
"IntervalTier"
"speaker"
0
10.0
3
0.0
2.5
"spk1"
2.5
5.0
""
5.0
8.0
"spk2"
"#;
        let intervals = parse_textgrid_intervals(content).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0], (0.0, 2.5, "spk1".to_string()));
        assert_eq!(intervals[1], (5.0, 8.0, "spk2".to_string()));
    }

    #[test]
    fn filters_silence_labels() {
        let content = r#"File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "speaker"
        xmin = 0
        xmax = 10.0
        intervals: size = 4
        intervals [1]:
            xmin = 0.0
            xmax = 1.0
            text = "sil"
        intervals [2]:
            xmin = 1.0
            xmax = 2.0
            text = "<noise>"
        intervals [3]:
            xmin = 2.0
            xmax = 3.0
            text = "SIL"
        intervals [4]:
            xmin = 3.0
            xmax = 5.0
            text = "speaker1"
"#;
        let intervals = parse_textgrid_intervals(content).unwrap();
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].2, "speaker1");
    }
}
