mod support;

use std::path::Path;

use speakrs::pipeline::{DiarizationPipeline, FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS};
use speakrs::segment::to_segments;

use support::{ExampleResult, load_models, load_wav_samples};

fn main() -> ExampleResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: cargo run --example print_turns -- <models-dir> <audio.wav>");
        std::process::exit(1);
    }

    let models_dir = Path::new(&args[1]);
    let audio_path = Path::new(&args[2]);

    let mut models = load_models(models_dir)?;
    let audio = load_wav_samples(audio_path)?;
    let mut pipeline = DiarizationPipeline::new(&mut models.0, &mut models.1, models_dir)?;
    let result = pipeline.run(&audio)?;
    let segments = to_segments(
        &result.discrete_diarization,
        FRAME_STEP_SECONDS,
        FRAME_DURATION_SECONDS,
    );

    println!("start\tend\tspeaker");
    for segment in segments {
        println!(
            "{:.3}\t{:.3}\t{}",
            segment.start, segment.end, segment.speaker
        );
    }

    Ok(())
}
