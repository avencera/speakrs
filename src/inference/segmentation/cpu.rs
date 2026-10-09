//! Native CPU segmentation adapter with useful batches and shared immutable weights

use super::SegmentationError;
use crate::inference::cpu::{
    assets::SegmentationAssets,
    segmentation::{CpuPyanNet, PyanNetWorkspace},
    workers::CpuWorkers,
};
use crate::inference::native_model::{WeightsFile, segmentation::WINDOW_SAMPLES};
use crate::inference::{CpuError, ModelLoadError, TensorShapeError};
use ndarray::Array2;
use std::path::Path;

/// Native model and private scratch for one segmentation handle
pub(super) struct CpuSegmentationBackend {
    model: CpuPyanNet,
    workers: CpuWorkers<PyanNetWorkspace>,
}

impl CpuSegmentationBackend {
    /// Loads the supported family and checks fixed input geometry
    pub(super) fn load(model_path: &Path, window_samples: usize) -> Result<Self, ModelLoadError> {
        if window_samples != WINDOW_SAMPLES {
            return Err(CpuError::Shape(TensorShapeError::LengthMismatch {
                context: "CPU segmentation window",
                expected: WINDOW_SAMPLES,
                actual: window_samples,
            })
            .into());
        }
        let assets = SegmentationAssets::resolve(model_path)?;
        let file = WeightsFile::open(assets.weights()).map_err(CpuError::from)?;
        let model = CpuPyanNet::load(&file)?;
        let workers = CpuWorkers::new(model.workspace());
        Ok(Self { model, workers })
    }

    /// Maximum useful windows in one adapter batch group
    pub(super) fn capacity(&self) -> usize {
        8
    }

    /// Runs one padded or truncated window with reset recurrent state
    pub(super) fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        self.model
            .forward(window, self.workers.first())
            .map_err(SegmentationError::from)
    }

    /// Splits arbitrary useful lengths in groups of eight, without synthetic output rows
    pub(super) fn run_batch(
        &mut self,
        windows: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        let mut outputs = Vec::with_capacity(windows.len());
        for group in windows.chunks(self.capacity()) {
            outputs.extend(self.workers.map(
                group,
                || self.model.workspace(),
                |workspace, window| self.model.forward(window, workspace),
            )?);
        }
        Ok(outputs)
    }
}

impl Clone for CpuSegmentationBackend {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            workers: CpuWorkers::new(self.model.workspace()),
        }
    }
}

#[cfg(test)]
mod tests;
