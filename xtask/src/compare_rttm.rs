use std::collections::HashMap;
use std::path::Path;

use color_eyre::eyre::Result;

struct Segment {
    start: f64,
    duration: f64,
    speaker: String,
}

fn parse_rttm(content: &str) -> Vec<Segment> {
    content
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.first() != Some(&"SPEAKER") || parts.len() < 8 {
                return None;
            }
            Some(Segment {
                start: parts[3].parse().ok()?,
                duration: parts[4].parse().ok()?,
                speaker: parts[7].to_string(),
            })
        })
        .collect()
}

fn total_speech(segments: &[Segment]) -> f64 {
    segments.iter().map(|s| s.duration).sum()
}

fn speaker_durations(segments: &[Segment]) -> HashMap<&str, f64> {
    let mut map = HashMap::new();
    for seg in segments {
        *map.entry(seg.speaker.as_str()).or_insert(0.0) += seg.duration;
    }
    map
}

fn format_speaker_durations(durations: &HashMap<&str, f64>) -> String {
    let mut pairs: Vec<_> = durations.iter().collect();
    pairs.sort_by_key(|(k, _)| k.to_string());
    let inner: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}: {v:.1}s")).collect();
    format!("{{{}}}", inner.join(", "))
}

/// Compute how much of segs_a's speech time is covered by segs_b (any speaker)
fn timeline_overlap(segs_a: &[Segment], segs_b: &[Segment]) -> f64 {
    let mut intervals: Vec<(f64, f64)> = segs_b
        .iter()
        .map(|s| (s.start, s.start + s.duration))
        .collect();
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));

    if intervals.is_empty() {
        return 0.0;
    }

    // merge overlapping intervals
    let mut merged = vec![intervals[0]];
    for &(start, end) in &intervals[1..] {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
            continue;
        }
        merged.push((start, end));
    }

    let mut covered = 0.0;
    for seg in segs_a {
        let seg_start = seg.start;
        let seg_end = seg.start + seg.duration;
        for &(b_start, b_end) in &merged {
            if b_start >= seg_end {
                break;
            }
            if b_end <= seg_start {
                continue;
            }
            covered += seg_end.min(b_end) - seg_start.max(b_start);
        }
    }
    covered
}

/// Compare two RTTM files and print a summary table
pub fn compare_rttm_files(a: &Path, b: &Path) -> Result<()> {
    let a_content = std::fs::read_to_string(a)?;
    let b_content = std::fs::read_to_string(b)?;

    let a_segs = parse_rttm(&a_content);
    let b_segs = parse_rttm(&b_content);

    let a_total = total_speech(&a_segs);
    let b_total = total_speech(&b_segs);
    let a_speakers = speaker_durations(&a_segs);
    let b_speakers = speaker_durations(&b_segs);

    println!("{:30} {:>10} {:>10}", "", "A", "B");
    println!("{}", "─".repeat(52));
    println!("{:30} {:10} {:10}", "Segments", a_segs.len(), b_segs.len());
    println!(
        "{:30} {:10} {:10}",
        "Speakers",
        a_speakers.len(),
        b_speakers.len()
    );
    println!(
        "{:30} {:>10.1} {:>10.1}",
        "Total speech (s)", a_total, b_total
    );
    println!();

    println!("Per-speaker duration:");
    println!("  A: {}", format_speaker_durations(&a_speakers));
    println!("  B: {}", format_speaker_durations(&b_speakers));
    println!();

    if a_total > 0.0 && b_total > 0.0 {
        let a_covered = timeline_overlap(&a_segs, &b_segs);
        let b_covered = timeline_overlap(&b_segs, &a_segs);
        println!(
            "{:30} {:>9.1}%",
            "A speech covered by B",
            a_covered / a_total * 100.0
        );
        println!(
            "{:30} {:>9.1}%",
            "B speech covered by A",
            b_covered / b_total * 100.0
        );

        let audio_end = a_segs
            .iter()
            .chain(b_segs.iter())
            .map(|s| s.start + s.duration)
            .fold(0.0_f64, f64::max);
        println!("{:30} {:>10.1}", "Audio span (s)", audio_end);
    }

    Ok(())
}
