use ndarray::{Array2, Array3, s};
use thiserror::Error;

use crate::imported_segmentation::{
    ArgmaxTie, FilterBoundary, MAX_MEDIAN_FILTER_WIDTH, SegmentationBundle,
    SegmentationBundleError, SegmentationManifest,
};
use crate::powerset::{PowersetDecodeError, PowersetMapping};

use super::types::DecodedSegmentations;

/// Errors raised while decoding a checked imported segmentation bundle
#[derive(Debug, Error)]
pub enum ImportedDecodeError {
    /// The bundle was not a valid checked artifact
    #[error(transparent)]
    Bundle(#[from] SegmentationBundleError),
    /// A powerset mapping could not be represented by the decoder
    #[error("invalid imported powerset mapping: {0}")]
    Mapping(String),
    /// A score tensor shape did not match its manifest range
    #[error("imported score tensor shape mismatch: {0}")]
    Shape(String),
    /// A powerset score row could not be hard-decoded
    #[error(transparent)]
    Powerset(#[from] PowersetDecodeError),
    /// A median filter width is not a bounded positive odd width
    #[error("median filter width must be positive, odd, and bounded, got {0}")]
    InvalidFilterWidth(usize),
}

/// Decoder for the explicit class order and filter policy in one bundle
#[derive(Clone, Debug)]
pub(crate) struct ImportedSegmentationDecoder {
    mapping: PowersetMapping,
    tie: ArgmaxTie,
    filter_enabled: bool,
    filter_boundary: FilterBoundary,
    filter_width: usize,
}

impl ImportedSegmentationDecoder {
    /// Build a decoder from a validated imported manifest
    pub(crate) fn from_manifest(
        manifest: &SegmentationManifest,
    ) -> Result<Self, ImportedDecodeError> {
        let classes = manifest
            .head
            .class_to_slot_subsets
            .iter()
            .map(|subset| {
                subset
                    .iter()
                    .copied()
                    .map(|slot| {
                        usize::try_from(slot).map_err(|_| {
                            ImportedDecodeError::Mapping(
                                "class subset slot does not fit in usize".to_owned(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mapping = PowersetMapping::from_ordered_subsets(
            usize::try_from(manifest.head.local_slots).map_err(|_| {
                ImportedDecodeError::Mapping("local slot count does not fit in usize".to_owned())
            })?,
            classes,
        )
        .map_err(|error| ImportedDecodeError::Mapping(error.to_string()))?;

        let filter_width = usize::try_from(manifest.policy.filter.width).map_err(|_| {
            ImportedDecodeError::InvalidFilterWidth(manifest.policy.filter.width as usize)
        })?;
        if manifest.policy.filter.enabled && !is_valid_filter_width(filter_width) {
            return Err(ImportedDecodeError::InvalidFilterWidth(filter_width));
        }

        Ok(Self {
            mapping,
            tie: manifest.head.argmax_tie,
            filter_enabled: manifest.policy.filter.enabled,
            filter_boundary: manifest.policy.filter.boundary,
            filter_width,
        })
    }

    /// Decode all checked score shards in their declared chunk order
    pub(crate) fn decode(
        &self,
        bundle: &SegmentationBundle,
    ) -> Result<DecodedSegmentations, ImportedDecodeError> {
        let manifest = &bundle.manifest;
        let chunks = manifest.geometry.chunks.len();
        let frames = usize::try_from(manifest.geometry.frame_grid.frame_count).map_err(|_| {
            ImportedDecodeError::Shape("frame count does not fit in usize".to_owned())
        })?;
        let classes = self.mapping.num_powerset_classes();
        let slots = usize::try_from(manifest.head.local_slots).map_err(|_| {
            ImportedDecodeError::Shape("slot count does not fit in usize".to_owned())
        })?;

        if chunks == 0 {
            return Ok(DecodedSegmentations(Array3::zeros((0, frames, slots))));
        }

        let mut decoded = Array3::<f32>::zeros((chunks, frames, slots));
        let mut next_chunk = 0_usize;
        for shard in &bundle.shards {
            let shard_start = usize::try_from(shard.chunk_start).map_err(|_| {
                ImportedDecodeError::Shape("shard chunk start does not fit in usize".to_owned())
            })?;
            let shard_end = usize::try_from(shard.chunk_end).map_err(|_| {
                ImportedDecodeError::Shape("shard chunk end does not fit in usize".to_owned())
            })?;
            if shard_start != next_chunk || shard_end < shard_start || shard_end > chunks {
                return Err(ImportedDecodeError::Shape(format!(
                    "shard {} does not continue chunk order",
                    shard.path
                )));
            }
            let shard_chunks = shard_end - shard_start;
            let expected_values = shard_chunks
                .checked_mul(frames)
                .and_then(|value| value.checked_mul(classes))
                .ok_or_else(|| {
                    ImportedDecodeError::Shape("score tensor shape overflow".to_owned())
                })?;
            if shard.shape
                != [
                    shard.chunk_end - shard.chunk_start,
                    u64::from(manifest.geometry.frame_grid.frame_count),
                    u64::try_from(classes).map_err(|_| {
                        ImportedDecodeError::Shape("class count does not fit in u64".to_owned())
                    })?,
                ]
                || shard.values.len() != expected_values
            {
                return Err(ImportedDecodeError::Shape(format!(
                    "shard {} has shape {:?} and {} values",
                    shard.path,
                    shard.shape,
                    shard.values.len()
                )));
            }

            for local_chunk in 0..shard_chunks {
                let chunk_offset = local_chunk
                    .checked_mul(frames)
                    .and_then(|value| value.checked_mul(classes))
                    .ok_or_else(|| {
                        ImportedDecodeError::Shape("score tensor offset overflow".to_owned())
                    })?;
                let chunk_values = &shard.values[chunk_offset..chunk_offset + frames * classes];
                let scores = Array2::from_shape_vec((frames, classes), chunk_values.to_vec())
                    .map_err(|error| ImportedDecodeError::Shape(error.to_string()))?;
                let hard = self.decode_scores(&scores)?;
                decoded
                    .slice_mut(s![shard_start + local_chunk, .., ..])
                    .assign(&hard);
            }
            next_chunk = shard_end;
        }

        if next_chunk != chunks {
            return Err(ImportedDecodeError::Shape(format!(
                "score shards cover {next_chunk} chunks, expected {chunks}"
            )));
        }

        if self.filter_enabled {
            match self.filter_boundary {
                FilterBoundary::Reflect => {
                    for chunk_idx in 0..chunks {
                        let mut chunk = decoded.slice_mut(s![chunk_idx, .., ..]);
                        median_filter_reflect(chunk.view_mut(), self.filter_width)?;
                    }
                }
            }
        }

        Ok(DecodedSegmentations(decoded))
    }

    /// Hard-decode one score matrix using the manifest's explicit class order
    pub(crate) fn decode_scores(
        &self,
        scores: &Array2<f32>,
    ) -> Result<Array2<f32>, ImportedDecodeError> {
        Ok(self.mapping.hard_decode_with_tie(scores, self.tie)?)
    }
}

/// Apply an odd-width median filter to each column using reflect boundaries
pub(crate) fn median_filter_reflect(
    mut values: ndarray::ArrayViewMut2<'_, f32>,
    width: usize,
) -> Result<(), ImportedDecodeError> {
    if !is_valid_filter_width(width) {
        return Err(ImportedDecodeError::InvalidFilterWidth(width));
    }
    if values.nrows() == 0 {
        return Ok(());
    }

    let radius = width / 2;
    let original = values.to_owned();
    let mut window = Vec::with_capacity(width);
    for frame in 0..values.nrows() {
        for slot in 0..values.ncols() {
            window.clear();
            for offset in 0..width {
                let source = frame as isize + offset as isize - radius as isize;
                let source = reflect_index(source, values.nrows());
                window.push(original[[source, slot]]);
            }
            window.sort_by(f32::total_cmp);
            values[[frame, slot]] = window[radius];
        }
    }
    Ok(())
}

fn is_valid_filter_width(width: usize) -> bool {
    width > 0 && !width.is_multiple_of(2) && width <= MAX_MEDIAN_FILTER_WIDTH
}

fn reflect_index(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * length as isize;
    let reflected = index.rem_euclid(period);
    if reflected < length as isize {
        reflected as usize
    } else {
        (period - 1 - reflected) as usize
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn reflect_filter_handles_a_short_signal() {
        let mut values = array![[0.0], [0.0], [0.0], [1.0], [1.0]];
        median_filter_reflect(values.view_mut(), 3).unwrap();
        assert_eq!(values, array![[0.0], [0.0], [0.0], [1.0], [1.0]]);
    }

    #[test]
    fn width_eleven_reflect_filter_repeats_edges_at_both_boundaries() {
        let mut values = Array2::<f32>::zeros((11, 1));
        for frame in 0..11 {
            values[[frame, 0]] = frame as f32;
        }
        median_filter_reflect(values.view_mut(), 11).unwrap();
        assert_eq!(values[[0, 0]], 2.0);
        assert_eq!(values[[10, 0]], 8.0);
    }

    #[test]
    fn median_filter_rejects_width_above_the_shared_limit() {
        let mut values = array![[0.0]];
        let width = MAX_MEDIAN_FILTER_WIDTH + 2;
        assert!(matches!(
            median_filter_reflect(values.view_mut(), width),
            Err(ImportedDecodeError::InvalidFilterWidth(actual)) if actual == width
        ));

        let mut manifest = crate::imported_segmentation::SegmentationManifest::from_json(
            include_bytes!("../../fixtures/wavlm_bridge/manifest.json"),
        )
        .unwrap();
        manifest.policy.filter.width = width as u32;
        assert!(matches!(
            ImportedSegmentationDecoder::from_manifest(&manifest),
            Err(ImportedDecodeError::InvalidFilterWidth(actual)) if actual == width
        ));
    }

    #[test]
    fn explicit_class_permutation_and_tie_policy_are_preserved() {
        let mut manifest = crate::imported_segmentation::SegmentationManifest::from_json(
            include_bytes!("../../fixtures/wavlm_bridge/manifest.json"),
        )
        .unwrap();
        manifest.policy.filter.enabled = false;
        manifest.head.class_to_slot_subsets.swap(1, 2);
        manifest.policy.decoder.argmax_tie = ArgmaxTie::First;
        manifest.head.argmax_tie = ArgmaxTie::First;
        let first = ImportedSegmentationDecoder::from_manifest(&manifest).unwrap();
        let scores = array![[0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        assert_eq!(
            first.decode_scores(&scores).unwrap(),
            array![[0.0, 1.0, 0.0, 0.0]]
        );

        manifest.policy.decoder.argmax_tie = ArgmaxTie::Last;
        manifest.head.argmax_tie = ArgmaxTie::Last;
        let last = ImportedSegmentationDecoder::from_manifest(&manifest).unwrap();
        let scores = array![[0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        assert_eq!(
            last.decode_scores(&scores).unwrap(),
            array![[1.0, 0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn explicit_class_maps_support_four_and_six_slot_heads() {
        let mut manifest = crate::imported_segmentation::SegmentationManifest::from_json(
            include_bytes!("../../fixtures/wavlm_bridge/manifest.json"),
        )
        .unwrap();
        manifest.policy.filter.enabled = false;

        for (slots, overlap, expected_classes) in [(4, 2, 11), (6, 2, 22), (6, 3, 42), (6, 6, 64)] {
            manifest.head.local_slots = slots;
            manifest.head.max_overlap = overlap;
            manifest.head.class_to_slot_subsets = explicit_class_map(slots, overlap);
            let decoder = ImportedSegmentationDecoder::from_manifest(&manifest).unwrap();
            assert_eq!(decoder.mapping.num_powerset_classes(), expected_classes);
        }
    }

    fn explicit_class_map(slots: u32, max_overlap: u32) -> Vec<Vec<u32>> {
        let mut classes = vec![Vec::new()];
        for size in 1..=max_overlap {
            append_combinations(slots, size, 0, &mut Vec::new(), &mut classes);
        }
        classes
    }

    fn append_combinations(
        slots: u32,
        size: u32,
        start: u32,
        current: &mut Vec<u32>,
        output: &mut Vec<Vec<u32>>,
    ) {
        if current.len() == size as usize {
            output.push(current.clone());
            return;
        }
        let remaining = size as usize - current.len();
        let max_start = slots as usize - remaining;
        for slot in start as usize..=max_start {
            current.push(slot as u32);
            append_combinations(slots, size, slot as u32 + 1, current, output);
            current.pop();
        }
    }
}
