use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;

use ndarray::{Array2, Array3, ArrayView2, s};
use ort::ep;
use ort::memory::Allocator;
use ort::session::{OutputSelector, RunOptions, Session};
use ort::value::{Tensor, TensorRef};
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::SEGMENTATION_STEP_SECONDS;
use speakrs::powerset::PowersetMapping;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        eprintln!(
            "Usage: profile_ort_embedding <borrow|owned|prealloc|stream-borrow|stream-owned|stream-prealloc|stream-batched> <path/to/audio.wav> [iterations_or_log_every] [log_every]"
        );
        std::process::exit(1);
    }

    let mode = &args[1];
    let wav_path = &args[2];
    let iterations = if mode.starts_with("stream-") {
        0
    } else {
        args.get(3)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5000)
    };
    let log_every = if mode.starts_with("stream-") {
        args.get(3)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(200)
    } else {
        args.get(4)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(250)
    };

    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/models");
    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .expect("failed to load segmentation model");
    let powerset = PowersetMapping::new(3, 2);

    let (samples, sample_rate) = load_wav_samples(wav_path);
    assert_eq!(
        sample_rate, 16_000,
        "expected 16kHz WAV, got {sample_rate}Hz"
    );

    let raw_windows = seg_model.run(&samples).expect("segmentation failed");
    let segmentations = decode_windows(raw_windows, &powerset);
    let model_path = std::env::var_os("SPEAKRS_PROFILE_ORT_MODEL_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| models_dir.join("wespeaker-voxceleb-resnet34.onnx"));
    let mut session = build_embedding_session(&model_path);

    eprintln!("start rss_mb={:.1}", rss_mb());
    let output_name = session.outputs()[0].name().to_owned();
    let run_options = RunOptions::new()
        .expect("failed to create run options")
        .with_outputs(
            OutputSelector::no_default().with(&output_name).preallocate(
                &output_name,
                Tensor::<f32>::new(&Allocator::default(), [1_usize, 256])
                    .expect("failed to allocate output tensor"),
            ),
        );

    if mode == "borrow" || mode == "owned" || mode == "prealloc" {
        let chunk_audio = chunk_audio(&samples, &seg_model, 0);
        let chunk_segmentations = segmentations.slice(s![0, .., ..]);
        let clean_masks = clean_masks(&chunk_segmentations);
        let local_idx = (0..chunk_segmentations.ncols())
            .find(|&speaker_idx| chunk_segmentations.column(speaker_idx).sum() > 0.0)
            .unwrap_or(0);
        let mask_column = chunk_segmentations.column(local_idx).to_owned();
        let clean_mask_column = clean_masks.column(local_idx).to_owned();
        let weights = select_mask(
            mask_column.as_slice().unwrap(),
            Some(clean_mask_column.as_slice().unwrap()),
            chunk_audio.len(),
            400,
        );

        let mut waveform_buffer = ndarray::Array3::<f32>::zeros((1, 1, 160_000));
        waveform_buffer
            .slice_mut(s![0, 0, ..chunk_audio.len()])
            .assign(&ndarray::ArrayView1::from(chunk_audio));
        let mut weights_buffer = ndarray::Array2::<f32>::zeros((1, 589));
        weights_buffer
            .slice_mut(s![0, ..weights.len()])
            .assign(&ndarray::ArrayView1::from(weights));

        for iteration in 0..iterations {
            run_one_embedding(
                mode,
                &mut session,
                &waveform_buffer,
                &weights_buffer,
                Some(&run_options),
            );
            if (iteration + 1) % log_every == 0 || iteration + 1 == iterations {
                eprintln!(
                    "mode={mode} iter={}/{} rss_mb={:.1}",
                    iteration + 1,
                    iterations,
                    rss_mb()
                );
            }
        }
    } else if mode == "stream-borrow" || mode == "stream-owned" || mode == "stream-prealloc" {
        let num_chunks = segmentations.shape()[0];
        let num_speakers = segmentations.shape()[2];
        let mut waveform_buffer = ndarray::Array3::<f32>::zeros((1, 1, 160_000));
        let mut weights_buffer = ndarray::Array2::<f32>::zeros((1, 589));

        for chunk_idx in 0..num_chunks {
            let chunk_audio = chunk_audio(&samples, &seg_model, chunk_idx);
            let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);
            waveform_buffer.fill(0.0);
            waveform_buffer
                .slice_mut(s![0, 0, ..chunk_audio.len()])
                .assign(&ndarray::ArrayView1::from(chunk_audio));

            for speaker_idx in 0..num_speakers {
                let mask = chunk_segmentations.column(speaker_idx).to_owned();
                let clean_mask = clean_masks.column(speaker_idx).to_owned();
                let selected_mask = select_mask(
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                    chunk_audio.len(),
                    400,
                );
                weights_buffer.fill(0.0);
                weights_buffer
                    .slice_mut(s![0, ..selected_mask.len()])
                    .assign(&ndarray::ArrayView1::from(selected_mask));
                run_one_embedding(
                    mode,
                    &mut session,
                    &waveform_buffer,
                    &weights_buffer,
                    Some(&run_options),
                );
            }

            if (chunk_idx + 1) % log_every == 0 || chunk_idx + 1 == num_chunks {
                eprintln!(
                    "mode={mode} chunk={}/{} rss_mb={:.1}",
                    chunk_idx + 1,
                    num_chunks,
                    rss_mb()
                );
            }
        }
    } else if mode == "stream-batched" {
        let num_chunks = segmentations.shape()[0];
        let num_speakers = segmentations.shape()[2];
        let batch_size = std::env::var("SPEAKRS_PROFILE_ORT_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16);
        let mut waveform_buffer = ndarray::Array3::<f32>::zeros((batch_size, 1, 160_000));
        let mut weights_buffer = ndarray::Array2::<f32>::zeros((batch_size, 589));
        let mut batch_fill = 0usize;

        for chunk_idx in 0..num_chunks {
            let chunk_audio = chunk_audio(&samples, &seg_model, chunk_idx);
            let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let mask = chunk_segmentations.column(speaker_idx).to_owned();
                let clean_mask = clean_masks.column(speaker_idx).to_owned();
                let selected_mask = select_mask(
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                    chunk_audio.len(),
                    400,
                );

                waveform_buffer.slice_mut(s![batch_fill, 0, ..]).fill(0.0);
                waveform_buffer
                    .slice_mut(s![batch_fill, 0, ..chunk_audio.len()])
                    .assign(&ndarray::ArrayView1::from(chunk_audio));
                weights_buffer.slice_mut(s![batch_fill, ..]).fill(0.0);
                weights_buffer
                    .slice_mut(s![batch_fill, ..selected_mask.len()])
                    .assign(&ndarray::ArrayView1::from(selected_mask));
                batch_fill += 1;

                if batch_fill == batch_size {
                    run_batched_embedding(
                        &mut session,
                        &waveform_buffer.slice(s![..batch_fill, .., ..]).to_owned(),
                        &weights_buffer.slice(s![..batch_fill, ..]).to_owned(),
                    );
                    batch_fill = 0;
                }
            }

            if (chunk_idx + 1) % log_every == 0 || chunk_idx + 1 == num_chunks {
                if batch_fill > 0 {
                    run_batched_embedding(
                        &mut session,
                        &waveform_buffer.slice(s![..batch_fill, .., ..]).to_owned(),
                        &weights_buffer.slice(s![..batch_fill, ..]).to_owned(),
                    );
                    batch_fill = 0;
                }
                eprintln!(
                    "mode={mode} chunk={}/{} rss_mb={:.1}",
                    chunk_idx + 1,
                    num_chunks,
                    rss_mb()
                );
            }
        }
    } else {
        eprintln!("Unknown mode: {mode}");
        std::process::exit(1);
    }
}

fn build_embedding_session(model_path: &Path) -> Session {
    if std::env::var_os("SPEAKRS_PROFILE_ORT_DEFAULTS").is_some() {
        return Session::builder()
            .expect("failed to create session builder")
            .commit_from_file(model_path)
            .expect("failed to load embedding session");
    }

    let builder = Session::builder()
        .expect("failed to create session builder")
        .with_independent_thread_pool()
        .expect("failed to configure thread pool")
        .with_intra_threads(1)
        .expect("failed to configure intra threads")
        .with_inter_threads(1)
        .expect("failed to configure inter threads")
        .with_memory_pattern(false)
        .expect("failed to disable memory pattern")
        .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
        .expect("failed to configure execution provider");
    let mut builder = builder;
    builder
        .commit_from_file(model_path)
        .expect("failed to load embedding session")
}

fn run_one_embedding(
    mode: &str,
    session: &mut Session,
    waveform_buffer: &ndarray::Array3<f32>,
    weights_buffer: &ndarray::Array2<f32>,
    run_options: Option<&RunOptions<ort::session::HasSelectedOutputs>>,
) {
    match mode {
        "borrow" | "stream-borrow" => {
            let waveform = TensorRef::from_array_view(waveform_buffer.view())
                .expect("failed to build waveform tensor");
            let weights = TensorRef::from_array_view(weights_buffer.view())
                .expect("failed to build weights tensor");
            let outputs = session
                .run(ort::inputs!["waveform" => waveform, "weights" => weights])
                .expect("embedding inference failed");
            let _ = outputs[0]
                .try_extract_tensor::<f32>()
                .expect("failed to extract output");
        }
        "owned" | "stream-owned" => {
            let waveform = Tensor::from_array(waveform_buffer.clone())
                .expect("failed to build waveform tensor");
            let weights =
                Tensor::from_array(weights_buffer.clone()).expect("failed to build weights tensor");
            let outputs = session
                .run(ort::inputs!["waveform" => waveform, "weights" => weights])
                .expect("embedding inference failed");
            let _ = outputs[0]
                .try_extract_tensor::<f32>()
                .expect("failed to extract output");
        }
        "prealloc" | "stream-prealloc" => {
            let waveform = TensorRef::from_array_view(waveform_buffer.view())
                .expect("failed to build waveform tensor");
            let weights = TensorRef::from_array_view(weights_buffer.view())
                .expect("failed to build weights tensor");
            let outputs = session
                .run_with_options(
                    ort::inputs!["waveform" => waveform, "weights" => weights],
                    run_options.expect("missing run options"),
                )
                .expect("embedding inference failed");
            let _ = outputs[0]
                .try_extract_tensor::<f32>()
                .expect("failed to extract output");
        }
        _ => unreachable!("unknown mode"),
    }
}

fn run_batched_embedding(
    session: &mut Session,
    waveform_buffer: &ndarray::Array3<f32>,
    weights_buffer: &ndarray::Array2<f32>,
) {
    let waveform = TensorRef::from_array_view(waveform_buffer.view())
        .expect("failed to build waveform tensor");
    let weights =
        TensorRef::from_array_view(weights_buffer.view()).expect("failed to build weights tensor");
    let outputs = session
        .run(ort::inputs!["waveform" => waveform, "weights" => weights])
        .expect("embedding inference failed");
    let _ = outputs[0]
        .try_extract_tensor::<f32>()
        .expect("failed to extract output");
}

fn decode_windows(raw_windows: Vec<Array2<f32>>, powerset: &PowersetMapping) -> Array3<f32> {
    let num_windows = raw_windows.len();
    let mut windows = raw_windows.into_iter();
    let first = powerset.hard_decode(&windows.next().unwrap());
    let mut stacked = Array3::<f32>::zeros((num_windows, first.nrows(), first.ncols()));
    stacked.slice_mut(s![0, .., ..]).assign(&first);

    for (window_idx, window) in windows.enumerate() {
        let decoded = powerset.hard_decode(&window);
        stacked
            .slice_mut(s![window_idx + 1, .., ..])
            .assign(&decoded);
    }

    stacked
}

fn clean_masks(segmentations: &ArrayView2<f32>) -> Array2<f32> {
    let single_active: Vec<bool> = segmentations
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().sum::<f32>() < 2.0)
        .collect();
    let mut clean = Array2::<f32>::zeros(segmentations.raw_dim());
    for (frame_idx, is_single_active) in single_active.iter().enumerate() {
        if !*is_single_active {
            continue;
        }

        clean
            .slice_mut(s![frame_idx, ..])
            .assign(&segmentations.slice(s![frame_idx, ..]));
    }
    clean
}

fn chunk_audio<'a>(audio: &'a [f32], seg_model: &SegmentationModel, chunk_idx: usize) -> &'a [f32] {
    let step_samples = (SEGMENTATION_STEP_SECONDS * seg_model.sample_rate() as f64) as usize;
    let start = chunk_idx * step_samples;
    let end = (start + seg_model.window_samples()).min(audio.len());
    if start < audio.len() {
        &audio[start..end]
    } else {
        &[]
    }
}

fn select_mask<'a>(
    mask: &'a [f32],
    clean_mask: Option<&'a [f32]>,
    num_samples: usize,
    min_num_samples: usize,
) -> &'a [f32] {
    let Some(clean_mask) = clean_mask else {
        return mask;
    };

    if clean_mask.len() != mask.len() || num_samples == 0 {
        return mask;
    }

    let min_mask_frames = (mask.len() * min_num_samples).div_ceil(num_samples) as f32;
    let clean_weight: f32 = clean_mask.iter().copied().sum();
    if clean_weight > min_mask_frames {
        clean_mask
    } else {
        mask
    }
}

fn rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .expect("failed to read rss from ps");
    let rss_kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0);
    rss_kb / 1024.0
}

fn load_wav_samples(path: &str) -> (Vec<f32>, u32) {
    let file = File::open(path).expect("failed to open WAV file");
    let mut reader = BufReader::new(file);
    let mut riff_header = [0u8; 12];
    reader
        .read_exact(&mut riff_header)
        .expect("failed to read WAV header");
    assert_eq!(&riff_header[0..4], b"RIFF", "expected RIFF WAV");
    assert_eq!(&riff_header[8..12], b"WAVE", "expected WAVE file");

    let mut sample_rate = None;
    let mut channels = None;
    let mut bits_per_sample = None;

    loop {
        let mut chunk_header = [0u8; 8];
        if reader.read_exact(&mut chunk_header).is_err() {
            break;
        }

        let chunk_id = &chunk_header[0..4];
        let chunk_size = u32::from_le_bytes(chunk_header[4..8].try_into().unwrap()) as usize;

        match chunk_id {
            b"fmt " => {
                let mut fmt = vec![0u8; chunk_size];
                reader
                    .read_exact(&mut fmt)
                    .expect("failed to read fmt chunk");
                let audio_format = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
                let chunk_channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
                let chunk_sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
                let chunk_bits_per_sample = u16::from_le_bytes(fmt[14..16].try_into().unwrap());

                assert_eq!(audio_format, 1, "expected PCM WAV");
                channels = Some(chunk_channels);
                sample_rate = Some(chunk_sample_rate);
                bits_per_sample = Some(chunk_bits_per_sample);
            }
            b"data" => {
                let sample_rate = sample_rate.expect("fmt chunk must appear before data chunk");
                let channels = channels.expect("missing channel count");
                let bits_per_sample = bits_per_sample.expect("missing bits per sample");
                assert_eq!(channels, 1, "expected mono WAV");
                assert_eq!(bits_per_sample, 16, "expected 16-bit PCM WAV");

                let mut samples = Vec::with_capacity(chunk_size / 2);
                let mut remaining = chunk_size;
                let mut buffer = [0u8; 8192];

                while remaining > 0 {
                    let to_read = remaining.min(buffer.len());
                    reader
                        .read_exact(&mut buffer[..to_read])
                        .expect("failed to read WAV samples");
                    for bytes in buffer[..to_read].chunks_exact(2) {
                        samples.push(i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0);
                    }
                    remaining -= to_read;
                }

                if chunk_size % 2 == 1 {
                    reader
                        .seek(SeekFrom::Current(1))
                        .expect("failed to skip WAV padding");
                }

                return (samples, sample_rate);
            }
            _ => {
                reader
                    .seek(SeekFrom::Current(chunk_size as i64))
                    .expect("failed to skip WAV chunk");
            }
        }

        if chunk_size % 2 == 1 {
            reader
                .seek(SeekFrom::Current(1))
                .expect("failed to skip WAV padding");
        }
    }

    panic!("no data chunk found in WAV");
}
