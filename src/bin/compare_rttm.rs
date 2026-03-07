use std::collections::HashMap;
use std::fs;

struct Segment {
    start: f64,
    duration: f64,
    speaker: String,
}

fn parse_rttm(path: &str) -> Vec<Segment> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("failed to read {path}: {e}");
        std::process::exit(1);
    });

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
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if intervals.is_empty() {
        return 0.0;
    }

    // merge overlapping intervals
    let mut merged = vec![intervals[0]];
    for &(start, end) in &intervals[1..] {
        let last = merged.last_mut().unwrap();
        if start <= last.1 {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: compare_rttm <rust.rttm> <python.rttm>");
        std::process::exit(1);
    }

    let rust_segs = parse_rttm(&args[1]);
    let py_segs = parse_rttm(&args[2]);

    let rust_total = total_speech(&rust_segs);
    let py_total = total_speech(&py_segs);
    let rust_speakers = speaker_durations(&rust_segs);
    let py_speakers = speaker_durations(&py_segs);

    println!("{:30} {:>10} {:>10}", "", "Rust", "Python");
    println!("{}", "─".repeat(52));
    println!(
        "{:30} {:10} {:10}",
        "Segments",
        rust_segs.len(),
        py_segs.len()
    );
    println!(
        "{:30} {:10} {:10}",
        "Speakers",
        rust_speakers.len(),
        py_speakers.len()
    );
    println!(
        "{:30} {:>10.1} {:>10.1}",
        "Total speech (s)", rust_total, py_total
    );
    println!();

    println!("Per-speaker duration:");
    println!("  Rust:   {}", format_speaker_durations(&rust_speakers));
    println!("  Python: {}", format_speaker_durations(&py_speakers));
    println!();

    if rust_total > 0.0 && py_total > 0.0 {
        let rust_covered = timeline_overlap(&rust_segs, &py_segs);
        let py_covered = timeline_overlap(&py_segs, &rust_segs);
        println!(
            "{:30} {:>9.1}%",
            "Rust speech covered by Python",
            rust_covered / rust_total * 100.0
        );
        println!(
            "{:30} {:>9.1}%",
            "Python speech covered by Rust",
            py_covered / py_total * 100.0
        );

        let audio_end = rust_segs
            .iter()
            .chain(py_segs.iter())
            .map(|s| s.start + s.duration)
            .fold(0.0_f64, f64::max);
        println!("{:30} {:>10.1}", "Audio span (s)", audio_end);
    }
}
