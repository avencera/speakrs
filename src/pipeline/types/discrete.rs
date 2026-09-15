use std::ops::Deref;

use ndarray::Array2;

use crate::imported_segmentation::{OutputExtent, OutputExtentPolicy};

use super::{FrameTiming, PipelineError, PipelineGeometry};

/// Frame-level binary speaker activations with their exact output timing
#[derive(Debug, Clone)]
pub struct DiscreteDiarization {
    activations: Array2<f32>,
    timing: FrameTiming,
    output_extent: OutputExtent,
    output_extent_policy: OutputExtentPolicy,
}

impl Deref for DiscreteDiarization {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.activations
    }
}

impl DiscreteDiarization {
    /// Create activations checked against a pipeline geometry
    pub fn try_new(
        activations: Array2<f32>,
        geometry: &PipelineGeometry,
    ) -> Result<Self, PipelineError> {
        if activations.nrows() != geometry.output_frames() {
            return Err(PipelineError::Invariant(format!(
                "activation frames {} do not match geometry output frames {}",
                activations.nrows(),
                geometry.output_frames()
            )));
        }

        let timing = geometry.frame_timing()?;

        Ok(Self {
            activations,
            timing,
            output_extent: *geometry.output_extent(),
            output_extent_policy: geometry.output_extent_policy(),
        })
    }

    pub(crate) fn map_activations(&self, activations: Array2<f32>) -> Result<Self, PipelineError> {
        if activations.raw_dim() != self.activations.raw_dim() {
            return Err(PipelineError::Invariant(
                "activation transform changed the diarization shape".to_owned(),
            ));
        }

        Ok(Self {
            activations,
            timing: self.timing,
            output_extent: self.output_extent,
            output_extent_policy: self.output_extent_policy,
        })
    }

    /// Return the exact frame timing owned by these activations
    pub const fn timing(&self) -> FrameTiming {
        self.timing
    }

    /// Zero out all but the highest-scoring speaker in each frame, making activations exclusive
    pub fn make_exclusive(&mut self) {
        crate::reconstruct::make_exclusive(&mut self.activations);
    }

    /// Convert frame activations to time-stamped speaker segments using owned timing
    pub fn to_segments(&self) -> Vec<crate::segment::Segment> {
        crate::segment::to_segments_with_timing(
            &self.activations,
            self.timing,
            self.output_extent,
            self.output_extent_policy,
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::imported_segmentation::{
        ChunkGeometry, OutputExtentPolicy, RationalSample, SegmentationManifest,
    };

    #[test]
    fn audio_extent_clips_segments_to_the_canonical_audio_end() {
        let mut manifest = SegmentationManifest::from_json(include_bytes!(
            "../../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        manifest.audio.sample_count = 321;
        manifest.geometry.chunks = vec![ChunkGeometry {
            index: 0,
            padding_samples: 127_679,
            start_samples: 0,
            valid_samples: 321,
        }];
        manifest.geometry.aggregate_grid.origin = RationalSample {
            numerator: 0,
            denominator: 1,
        };
        manifest.geometry.aggregate_grid.step = RationalSample {
            numerator: 320,
            denominator: 1,
        };
        manifest.geometry.aggregate_grid.support = RationalSample {
            numerator: 400,
            denominator: 1,
        };
        manifest.geometry.output_extent.end_samples = 321;
        manifest.geometry.output_extent_policy = OutputExtentPolicy::AudioExtent;
        let geometry =
            PipelineGeometry::from_imported(&manifest.audio, &manifest.geometry).unwrap();
        let diarization =
            DiscreteDiarization::try_new(ndarray::array![[1.0], [1.0]], &geometry).unwrap();

        let segments = diarization.to_segments();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].end, 321.0 / 16_000.0);
    }

    #[test]
    fn imported_diarization_uses_first_and_last_exact_frame_timing() {
        let mut manifest = SegmentationManifest::from_json(include_bytes!(
            "../../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        manifest.audio.sample_count = 1;
        manifest.geometry.chunks = vec![ChunkGeometry {
            index: 0,
            padding_samples: 127_999,
            start_samples: 0,
            valid_samples: 1,
        }];
        manifest.geometry.output_extent.end_samples = 128_400;
        let geometry =
            PipelineGeometry::from_imported(&manifest.audio, &manifest.geometry).unwrap();

        let mut activations = Array2::<f32>::zeros((geometry.output_frames(), 1));
        activations[[0, 0]] = 1.0;
        activations[[geometry.output_frames() - 1, 0]] = 1.0;
        let diarization = DiscreteDiarization::try_new(activations, &geometry).unwrap();

        let segments = diarization.to_segments();

        assert_eq!(segments.len(), 2);
        assert!((segments[0].start - 200.0 / 16_000.0).abs() < 1e-12);
        assert!((segments[0].end - 520.0 / 16_000.0).abs() < 1e-12);
        assert!((segments[1].start - 128_200.0 / 16_000.0).abs() < 1e-12);
        assert!((segments[1].end - 128_200.0 / 16_000.0).abs() < 1e-12);
    }
}
