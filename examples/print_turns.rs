mod support;

use std::path::Path;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

use support::{ExampleResult, load_wav_samples};

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: cargo run --features cpu --example print_turns -- <models-dir> <audio.wav>"
        );
        std::process::exit(1);
    }

    let models_dir = Path::new(&args[1]);
    let audio_path = Path::new(&args[2]);

    let audio = load_wav_samples(audio_path)?;
    let mut pipeline = OwnedDiarizationPipeline::from_dir(models_dir, ExecutionMode::Cpu)?;
    let result = pipeline.run(&audio)?;
    let segments = result.discrete_diarization.to_segments();

    println!("start\tend\tspeaker");
    for segment in segments {
        println!(
            "{:.3}\t{:.3}\t{}",
            segment.start, segment.end, segment.speaker
        );
    }

    Ok(())
}
