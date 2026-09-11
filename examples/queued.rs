mod support;

use std::path::{Path, PathBuf};
use std::thread;

use speakrs::{ExecutionMode, PipelineBuilder, QueuedDiarizationRequest};

use support::{ExampleResult, file_id_from_path, load_wav_samples};

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: cargo run --example queued -- <models-dir> <audio.wav> [audio2.wav ...]");
        std::process::exit(1);
    }

    let models_dir = Path::new(&args[1]);
    let audio_paths: Vec<PathBuf> = args[2..].iter().map(PathBuf::from).collect();
    let (tx, rx) = PipelineBuilder::from_dir(models_dir, ExecutionMode::Cpu)?.build_queued()?;

    let mut handles = Vec::with_capacity(audio_paths.len());
    for audio_path in audio_paths {
        let tx = tx.clone();
        let file_id = file_id_from_path(&audio_path);
        let audio = load_wav_samples(&audio_path)?;
        handles.push(thread::spawn(move || {
            tx.try_push(QueuedDiarizationRequest::new(file_id, audio))
                .map(|_| ())
        }));
    }
    drop(tx);

    for handle in handles {
        handle
            .join()
            .map_err(|_| "queue sender thread panicked")??;
    }

    for result in rx {
        let result = result?;
        print!("{}", result.result?.rttm(&result.file_id));
    }

    Ok(())
}
