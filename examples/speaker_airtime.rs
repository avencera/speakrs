mod support;

use std::collections::BTreeMap;
use std::path::Path;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

use support::{ExampleResult, load_wav_samples};

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: cargo run --features cpu --example speaker_airtime -- <models-dir> <audio.wav>"
        );
        std::process::exit(1);
    }

    let models_dir = Path::new(&args[1]);
    let audio_path = Path::new(&args[2]);

    let audio = load_wav_samples(audio_path)?;
    let mut pipeline = OwnedDiarizationPipeline::from_dir(models_dir, ExecutionMode::Cpu)?;
    let result = pipeline.run(&audio)?;
    let segments = result.discrete_diarization.to_segments();

    let mut airtime = BTreeMap::<String, f64>::new();
    for segment in segments {
        *airtime.entry(segment.speaker).or_default() += segment.duration();
    }

    println!("speaker\ttotal_seconds");
    for (speaker, seconds) in airtime {
        println!("{speaker}\t{seconds:.3}");
    }

    Ok(())
}
