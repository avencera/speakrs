use std::fs;
use std::path::Path;

use ndarray::{Array2, Array3, s};
use speakrs::aggregate::overlap_add;
use speakrs::binarize::{BinarizeConfig, binarize};
use speakrs::clustering::vbx::{VbxConfig, cluster};
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::powerset::PowersetMapping;
use speakrs::reconstruct::{count_speakers, reconstruct};
use speakrs::segment::{merge_segments, to_rttm, to_segments};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: diarize <path/to/audio.wav>");
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");

    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        2.5,
    )
    .expect("failed to load segmentation model");

    let mut emb_model = EmbeddingModel::new(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
    )
    .expect("failed to load embedding model");

    let (samples, sr) = load_wav_samples(wav_path);
    assert_eq!(sr, 16000, "expected 16kHz WAV, got {sr}Hz");

    let rttm = run_diarization(&mut seg_model, &mut emb_model, &samples);
    print!("{rttm}");
}

fn run_diarization(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
) -> String {
    let pm = PowersetMapping::new(3, 2);
    let step_frames = 147;

    let windows = seg_model.run(audio).expect("segmentation failed");
    if windows.is_empty() {
        return String::new();
    }

    let decoded_windows: Vec<Array2<f32>> = windows.iter().map(|w| pm.soft_decode(w)).collect();

    let frames_per_window = decoded_windows[0].nrows();
    let num_speakers = decoded_windows[0].ncols();
    let mut windows_3d =
        Array3::<f32>::zeros((decoded_windows.len(), frames_per_window, num_speakers));
    for (i, w) in decoded_windows.iter().enumerate() {
        windows_3d.slice_mut(s![i, .., ..]).assign(w);
    }

    let aggregated = overlap_add(&windows_3d, step_frames, 0);

    let config = BinarizeConfig::default();
    let binary = binarize(&aggregated, &config);
    let speaker_counts = count_speakers(&binary);

    let sample_rate = seg_model.sample_rate();
    let frame_duration = seg_model.step_samples() as f64 / sample_rate as f64 / step_frames as f64;

    let mut embeddings = Vec::new();
    for speaker_idx in 0..num_speakers {
        let active_frames: Vec<usize> = (0..binary.nrows())
            .filter(|&f| binary[[f, speaker_idx]] == 1.0)
            .collect();

        if active_frames.is_empty() {
            embeddings.push(ndarray::Array1::zeros(256));
            continue;
        }

        let mid = active_frames[active_frames.len() / 2];
        let start_sample = (mid as f64 * frame_duration * sample_rate as f64) as usize;
        let end_sample = (start_sample + sample_rate).min(audio.len());

        if end_sample <= start_sample || end_sample - start_sample < sample_rate / 4 {
            embeddings.push(ndarray::Array1::zeros(256));
            continue;
        }

        let emb = emb_model
            .embed(&audio[start_sample..end_sample])
            .expect("embedding failed");
        embeddings.push(emb);
    }

    let emb_dim = 256;
    let mut emb_matrix = Array2::<f32>::zeros((num_speakers, emb_dim));
    for (i, emb) in embeddings.iter().enumerate() {
        emb_matrix.row_mut(i).assign(emb);
    }

    let vbx_config = VbxConfig {
        fa: 0.07,
        fb: 0.8,
        ..Default::default()
    };
    let cluster_result = cluster(&emb_matrix.view(), num_speakers, None, &vbx_config);

    let reconstructed = reconstruct(
        &aggregated,
        &cluster_result.labels,
        cluster_result.num_clusters,
        Some(&speaker_counts),
    );

    let final_binary = binarize(&reconstructed, &config);
    let segments = to_segments(&final_binary, frame_duration);
    let merged = merge_segments(&segments, 0.5);
    to_rttm(&merged, "file1")
}

fn load_wav_samples(path: &str) -> (Vec<f32>, u32) {
    let data = fs::read(path).expect("failed to read WAV file");

    let sample_rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(data[34..36].try_into().unwrap());
    assert_eq!(bits_per_sample, 16, "expected 16-bit PCM WAV");

    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            let samples: Vec<f32> = data[pos + 8..pos + 8 + chunk_size]
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            return (samples, sample_rate);
        }
        pos += 8 + chunk_size;
    }
    panic!("no data chunk found in WAV");
}
