mod support;

use std::fs;
use std::path::Path;

use speakrs::pipeline::{DiarizationPipeline, FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS};
use speakrs::reconstruct::make_exclusive;
use speakrs::segment::{Segment, to_segments};

use support::{ExampleResult, load_models, load_wav_samples};

struct TranscriptRow {
    start: f64,
    end: f64,
    text: String,
}

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: cargo run --example assign_transcript_speakers -- <models-dir> <audio.wav> <transcript.tsv>"
        );
        std::process::exit(1);
    }

    let models_dir = Path::new(&args[1]);
    let audio_path = Path::new(&args[2]);
    let transcript_path = Path::new(&args[3]);

    let mut models = load_models(models_dir)?;
    let audio = load_wav_samples(audio_path)?;
    let mut pipeline = DiarizationPipeline::new(&mut models.0, &mut models.1, models_dir)?;
    let result = pipeline.run(&audio)?;

    let mut exclusive = result.discrete_diarization.clone();
    make_exclusive(&mut exclusive.0);
    let segments = to_segments(&exclusive.0, FRAME_STEP_SECONDS, FRAME_DURATION_SECONDS);
    let transcript = load_transcript(transcript_path)?;

    println!("start\tend\tspeaker\ttext");
    for row in transcript {
        let speaker = dominant_speaker(&segments, row.start, row.end)
            .unwrap_or("UNKNOWN")
            .to_owned();
        println!(
            "{:.3}\t{:.3}\t{}\t{}",
            row.start, row.end, speaker, row.text
        );
    }

    Ok(())
}

fn load_transcript(path: &Path) -> ExampleResult<Vec<TranscriptRow>> {
    let content = fs::read_to_string(path)?;
    let mut rows = Vec::new();

    for (line_idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let mut fields = line.splitn(3, '\t');
        let start = fields
            .next()
            .ok_or_else(|| format!("line {} is missing start time", line_idx + 1))?
            .parse::<f64>()?;
        let end = fields
            .next()
            .ok_or_else(|| format!("line {} is missing end time", line_idx + 1))?
            .parse::<f64>()?;
        let text = fields
            .next()
            .ok_or_else(|| format!("line {} is missing transcript text", line_idx + 1))?
            .to_owned();

        rows.push(TranscriptRow { start, end, text });
    }

    Ok(rows)
}

fn dominant_speaker(segments: &[Segment], start: f64, end: f64) -> Option<&str> {
    let mut best_speaker = None;
    let mut best_overlap = 0.0f64;

    for segment in segments {
        let overlap = overlap_seconds(segment.start, segment.end, start, end);
        if overlap > best_overlap {
            best_overlap = overlap;
            best_speaker = Some(segment.speaker.as_str());
        }
    }

    best_speaker
}

fn overlap_seconds(lhs_start: f64, lhs_end: f64, rhs_start: f64, rhs_end: f64) -> f64 {
    (lhs_end.min(rhs_end) - lhs_start.max(rhs_start)).max(0.0)
}
