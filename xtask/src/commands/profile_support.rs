use std::process::Command;

use color_eyre::eyre::{Result, eyre};
use ndarray::{Array1, Array2, Array3, ArrayView2, s};
use speakrs::PowersetMapping;
use speakrs::inference::SegmentationModel;
use speakrs::pipeline::SEGMENTATION_STEP_SECONDS;

pub(crate) fn decode_windows(
    raw_windows: Vec<Array2<f32>>,
    powerset: &PowersetMapping,
) -> Array3<f32> {
    let mut windows = raw_windows.into_iter();
    let Some(first_window) = windows.next() else {
        return Array3::zeros((0, 0, 0));
    };

    let first = powerset.hard_decode(&first_window);
    let mut stacked = Array3::<f32>::zeros((windows.len() + 1, first.nrows(), first.ncols()));
    stacked.slice_mut(s![0, .., ..]).assign(&first);

    for (window_idx, window) in windows.enumerate() {
        let decoded = powerset.hard_decode(&window);
        stacked
            .slice_mut(s![window_idx + 1, .., ..])
            .assign(&decoded);
    }

    stacked
}

pub(crate) fn clean_masks(segmentations: &ArrayView2<f32>) -> Array2<f32> {
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

pub(crate) fn chunk_audio<'a>(
    audio: &'a [f32],
    seg_model: &SegmentationModel,
    chunk_idx: usize,
) -> &'a [f32] {
    let step_samples = (SEGMENTATION_STEP_SECONDS * seg_model.sample_rate() as f64) as usize;
    let start = chunk_idx * step_samples;
    let end = (start + seg_model.window_samples()).min(audio.len());
    if start < audio.len() {
        &audio[start..end]
    } else {
        &[]
    }
}

pub(crate) fn array1_slice<'a>(array: &'a Array1<f32>, context: &str) -> Result<&'a [f32]> {
    array
        .as_slice()
        .ok_or_else(|| eyre!("{context}: array was not contiguous"))
}

pub(crate) fn rss_mb() -> Result<f64> {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()?;
    let rss_kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0);
    Ok(rss_kb / 1024.0)
}
