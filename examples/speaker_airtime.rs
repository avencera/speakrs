mod support;

use std::collections::BTreeMap;
use std::path::Path;

use speakrs::pipeline::{DiarizationPipeline, FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS};
use speakrs::segment::to_segments;

use support::{ExampleResult, load_models, load_wav_samples};

fn main() -> ExampleResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: cargo run --example speaker_airtime -- <models-dir> <audio.wav>");
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
