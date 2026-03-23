mod support;

use std::path::Path;

use speakrs::pipeline::DiarizationPipeline;

use support::{ExampleResult, file_id_from_path, load_models, load_wav_samples};

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("Usage: cargo run --example diarize_wav -- <models-dir> <audio.wav> [file-id]");
        std::process::exit(1);
    }

    let models_dir = Path::new(&args[1]);
    let audio_path = Path::new(&args[2]);
    let file_id = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| file_id_from_path(audio_path));

    let mut models = load_models(models_dir)?;
    let audio = load_wav_samples(audio_path)?;
    let mut pipeline = DiarizationPipeline::new(&mut models.0, &mut models.1, models_dir)?;
    let result = pipeline.run(&audio)?;

    print!("{}", result.rttm(&file_id));
    Ok(())
}
