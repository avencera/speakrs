//! Print the RTTM the core pipeline produces, as ground truth for binding tests
//!
//! Usage: `reference_rttm <mode> <models-dir> <16 kHz mono 16-bit wav> <file-id> [runs]`
//!
//! Prints the RTTM, a `# exclusive` line, then the exclusive RTTM. It uses the
//! core crate and hound directly, not the SDK, so it checks the bindings
//! independently. With `runs` > 1 it repeats the run and prints the median
//! pipeline time to stderr

use std::time::Instant;

use speakrs_sdk::speakrs::OwnedDiarizationPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, models_dir, wav, file_id, rest @ ..] = args.as_slice() else {
        return Err("usage: reference_rttm <mode> <models-dir> <wav> <file-id> [runs]".into());
    };
    let runs: usize = rest.first().map_or(Ok(1), |runs| runs.parse())?;
    let mode = speakrs_sdk::parse_mode(mode)?;

    let mut reader = hound::WavReader::open(wav)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 || spec.bits_per_sample != 16 {
        return Err(format!("expected 16 kHz mono 16-bit WAV, got {spec:?}").into());
    }
    let audio = reader
        .samples::<i16>()
        .map(|sample| sample.map(|sample| f32::from(sample) / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;

    let mut pipeline = OwnedDiarizationPipeline::from_dir(models_dir, mode)?;
    let mut timings = Vec::with_capacity(runs);
    let mut result = None;
    for _ in 0..runs.max(1) {
        let start = Instant::now();
        result = Some(pipeline.run_with_file_id(&audio, file_id)?);
        timings.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let result = result.expect("at least one run");
    print!("{}", result.rttm(file_id));
    println!("# exclusive");
    print!(
        "{}",
        speakrs_sdk::speakrs::segment::to_rttm(&result.exclusive_segments, file_id)
    );

    timings.sort_by(f64::total_cmp);
    eprintln!(
        "median_pipeline_ms={:.1} runs={}",
        timings[timings.len() / 2],
        timings.len()
    );
    Ok(())
}
