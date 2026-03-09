use std::path::Path;

use eyre::bail;
use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};

fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (wav_path, seg_model_path, emb_model_path) = match args.as_slice() {
        [_, wav, seg, emb] => (wav.as_str(), seg.as_str(), emb.as_str()),
        _ => bail!(
            "Usage: diarize-pyannote-rs <wav_path> <segmentation_model.onnx> <embedding_model.onnx>"
        ),
    };

    assert!(
        Path::new(seg_model_path).exists(),
        "segmentation model not found: {seg_model_path}"
    );
    assert!(
        Path::new(emb_model_path).exists(),
        "embedding model not found: {emb_model_path}"
    );

    let (samples, sample_rate) = pyannote_rs::read_wav(wav_path)?;
    let max_speakers = 10;

    let mut extractor = EmbeddingExtractor::new(emb_model_path)?;
    let mut manager = EmbeddingManager::new(max_speakers);

    let segments = pyannote_rs::get_segments(&samples, sample_rate, seg_model_path)?;

    let file_id = "file1";
    for segment in segments {
        let segment = match segment {
            Ok(s) => s,
            Err(e) => {
                eprintln!("segment error: {e:?}");
                continue;
            }
        };

        let embedding = match extractor.compute(&segment.samples) {
            Ok(e) => e.collect::<Vec<f32>>(),
            Err(_) => continue,
        };

        let speaker_id = if manager.get_all_speakers().len() >= max_speakers {
            manager.get_best_speaker_match(embedding).unwrap_or(0)
        } else {
            manager.search_speaker(embedding, 0.5).unwrap_or(0)
        };

        let duration = segment.end - segment.start;
        println!(
            "SPEAKER {file_id} 1 {:.6} {:.6} <NA> <NA> SPEAKER_{speaker_id:02} <NA> <NA>",
            segment.start, duration
        );
    }

    Ok(())
}
