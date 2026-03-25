use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::{Result, bail, ensure};
use ndarray::{Array2, Array3, ArrayView2, s};
use ort::ep;
use ort::memory::Allocator;
use ort::session::{OutputSelector, RunOptions, Session};
use ort::value::{Tensor, TensorRef};
use speakrs::PowersetMapping;
use speakrs::inference::SegmentationModel;
use speakrs::pipeline::SEGMENTATION_STEP_SECONDS;

use crate::wav;

pub fn run(
    mode: &str,
    wav_path: &str,
    iterations: usize,
    log_every: usize,
    model_path: Option<PathBuf>,
    batch_size: Option<usize>,
    ort_defaults: bool,
) -> Result<()> {
    let iterations = if mode.starts_with("stream-") {
        0
    } else if iterations == 0 {
        5000
    } else {
        iterations
    };
    let log_every = if log_every == 0 {
        if mode.starts_with("stream-") {
            200
        } else {
            250
        }
    } else {
        log_every
    };

    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/models");
    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )?;
    let powerset = PowersetMapping::new(3, 2);

    let (samples, sample_rate) = wav::load_wav_samples(wav_path)?;
    ensure!(
        sample_rate == 16_000,
        "expected 16kHz WAV, got {sample_rate}Hz"
    );

    let raw_windows = seg_model.run(&samples)?;
    let segmentations = decode_windows(raw_windows, &powerset);
    let resolved_model_path =
        model_path.unwrap_or_else(|| models_dir.join("wespeaker-voxceleb-resnet34.onnx"));
    let mut session = build_embedding_session(&resolved_model_path, ort_defaults)?;

    eprintln!("start rss_mb={:.1}", rss_mb());
    let output_name = session.outputs()[0].name().to_owned();
    let run_options = RunOptions::new()
        .map_err(|e| color_eyre::eyre::eyre!("{e}"))?
        .with_outputs(
            OutputSelector::no_default().with(&output_name).preallocate(
                &output_name,
                Tensor::<f32>::new(&Allocator::default(), [1_usize, 256])
                    .map_err(|e| color_eyre::eyre::eyre!("{e}"))?,
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
        let batch_size = batch_size.unwrap_or(16);
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
        bail!("unknown mode: {mode}");
    }

    Ok(())
}

fn ort_err(e: impl std::fmt::Display) -> color_eyre::eyre::Report {
    color_eyre::eyre::eyre!("{e}")
}

fn build_embedding_session(model_path: &Path, ort_defaults: bool) -> Result<Session> {
    if ort_defaults {
        return Session::builder()
            .map_err(ort_err)?
            .commit_from_file(model_path)
            .map_err(ort_err);
    }

    let mut builder = Session::builder()
        .map_err(ort_err)?
        .with_independent_thread_pool()
        .map_err(ort_err)?
        .with_intra_threads(1)
        .map_err(ort_err)?
        .with_inter_threads(1)
        .map_err(ort_err)?
        .with_memory_pattern(false)
        .map_err(ort_err)?
        .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
        .map_err(ort_err)?;

    builder.commit_from_file(model_path).map_err(ort_err)
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
