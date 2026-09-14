//! Semantic and geometry validation for imported manifests

use std::collections::BTreeSet;
use std::path::{Component, Path};

use crate::rational::{CheckedRational, RationalError};

use super::*;

impl SegmentationManifest {
    /// Validates every structural and semantic manifest invariant
    pub fn validate(&self) -> Result<(), SegmentationBundleError> {
        if self.format_version != FORMAT_VERSION {
            return invalid(format!(
                "unsupported format_version {}; expected {FORMAT_VERSION}",
                self.format_version
            ));
        }
        if self.schema_id != SCHEMA_ID {
            return invalid(format!(
                "unsupported schema_id {}; expected {SCHEMA_ID}",
                self.schema_id
            ));
        }
        validate_identity(&self.identity)?;
        validate_audio(&self.audio)?;
        validate_head(&self.head)?;
        validate_geometry(&self.audio, &self.geometry)?;
        validate_tensors(&self.geometry, &self.head, &self.tensors)?;
        validate_policy(&self.head, &self.geometry, &self.policy)?;
        let expected_bundle_id = self.canonical_bundle_id()?;
        if expected_bundle_id != self.identity.bundle_id {
            return invalid("bundle_id does not match derived content identity".to_owned());
        }
        let computed = self.canonical_digest()?;
        if computed != self.identity.manifest_digest {
            return invalid("manifest_digest does not match canonical identity".to_owned());
        }
        Ok(())
    }
}

pub(crate) fn invalid<T>(message: String) -> Result<T, SegmentationBundleError> {
    Err(SegmentationBundleError::Invalid(message))
}

fn validate_identity(identity: &BundleIdentity) -> Result<(), SegmentationBundleError> {
    validate_text(identity.bundle_id.as_str(), "identity.bundle_id")?;
    if identity.components.is_empty() {
        return invalid("identity.components must not be empty".to_owned());
    }
    let mut names = BTreeSet::new();
    for component in &identity.components {
        validate_text(&component.id, "identity.components[].id")?;
        validate_text(&component.name, "identity.components[].name")?;
        validate_text(&component.revision, "identity.components[].revision")?;
        if !names.insert(&component.name) {
            return invalid("identity component names must be unique".to_owned());
        }
    }
    for (name, reference) in [
        ("identity.config", &identity.config),
        ("identity.environment", &identity.environment),
        ("identity.model", &identity.model),
        ("identity.producer", &identity.producer),
        ("identity.source", &identity.source),
    ] {
        validate_text(&reference.id, &format!("{name}.id"))?;
        validate_text(&reference.revision, &format!("{name}.revision"))?;
    }
    Ok(())
}

fn validate_audio(audio: &AudioIdentity) -> Result<(), SegmentationBundleError> {
    if audio.channels != 1 {
        return invalid("audio.channels must be one".to_owned());
    }
    if audio.sample_rate != 16_000 {
        return invalid("audio.sample_rate must be 16000".to_owned());
    }
    if audio.sample_count > MAX_TENSOR_ELEMENTS.saturating_mul(64) {
        return invalid("audio.sample_count exceeds the admission bound".to_owned());
    }
    validate_text(&audio.original_recording_id, "audio.original_recording_id")?;
    if let Some(parent) = &audio.parent_recording_id {
        validate_text(parent, "audio.parent_recording_id")?;
    }
    validate_text(&audio.resampling.algorithm, "audio.resampling.algorithm")?;
    validate_text(&audio.resampling.id, "audio.resampling.id")?;
    validate_text(&audio.resampling.revision, "audio.resampling.revision")?;
    if audio.channel_selection != ChannelSelection::First || audio.downmix != Downmix::None {
        return invalid("v1 admits first-channel selection without downmix".to_owned());
    }
    Ok(())
}

fn validate_head(head: &SegmentationHead) -> Result<(), SegmentationBundleError> {
    if head.local_slots == 0 || u64::from(head.local_slots) > MAX_CLASSES {
        return invalid("head.local_slots is outside the admission bound".to_owned());
    }
    if head.max_overlap > head.local_slots {
        return invalid("head.max_overlap cannot exceed local_slots".to_owned());
    }
    let expected = powerset_class_count(head.local_slots, head.max_overlap)?;
    if expected > MAX_CLASSES {
        return invalid("head class count exceeds the admission bound".to_owned());
    }
    if head.class_to_slot_subsets.len() as u64 != expected {
        return invalid(format!(
            "head class mapping has {}; expected {expected} classes",
            head.class_to_slot_subsets.len()
        ));
    }
    let mut subsets = BTreeSet::new();
    for subset in &head.class_to_slot_subsets {
        if subset.len() as u32 > head.max_overlap {
            return invalid("a class subset exceeds max_overlap".to_owned());
        }
        if subset.windows(2).any(|window| window[0] >= window[1]) {
            return invalid("class subsets must be sorted and unique".to_owned());
        }
        if subset.iter().any(|slot| *slot >= head.local_slots) {
            return invalid("class subset contains an out-of-range slot".to_owned());
        }
        if !subsets.insert(subset.clone()) {
            return invalid("class subsets must be unique".to_owned());
        }
    }
    Ok(())
}

fn powerset_class_count(
    local_slots: u32,
    max_overlap: u32,
) -> Result<u64, SegmentationBundleError> {
    let mut total = 0_u64;
    for size in 0..=max_overlap {
        let mut count = 1_u64;
        for index in 0..size {
            count = count
                .checked_mul(u64::from(local_slots - index))
                .and_then(|value| value.checked_div(u64::from(index + 1)))
                .ok_or_else(|| {
                    SegmentationBundleError::Invalid("powerset count overflow".to_owned())
                })?;
        }
        total = total.checked_add(count).ok_or_else(|| {
            SegmentationBundleError::Invalid("powerset count overflow".to_owned())
        })?;
    }
    Ok(total)
}

fn validate_geometry(
    audio: &AudioIdentity,
    geometry: &SegmentationGeometry,
) -> Result<(), SegmentationBundleError> {
    if geometry.window_samples == 0 {
        return invalid("geometry.window_samples must be positive".to_owned());
    }
    if geometry.window_samples > u64::from(u32::MAX) * 1_024 {
        return invalid("geometry.window_samples exceeds the admission bound".to_owned());
    }
    validate_grid(&geometry.frame_grid)?;
    validate_grid(&geometry.aggregate_grid)?;
    if geometry.frame_grid.step != geometry.aggregate_grid.step
        || geometry.frame_grid.support != geometry.aggregate_grid.support
    {
        return invalid("receptive and aggregate grids must share step and support".to_owned());
    }
    let expected_frame_count = frame_count_for_window(
        geometry.window_samples,
        &geometry.frame_grid.support,
        &geometry.frame_grid.step,
    )?;
    if u64::from(geometry.frame_grid.frame_count) != expected_frame_count {
        return invalid("frame_grid.frame_count does not match window geometry".to_owned());
    }
    if geometry.aggregate_grid.frame_count != geometry.frame_grid.frame_count {
        return invalid("aggregate and receptive frame counts must match".to_owned());
    }
    if geometry.frame_grid.origin == geometry.aggregate_grid.origin {
        return invalid("receptive and aggregate origins must remain distinct".to_owned());
    }
    if geometry.aggregate_grid.origin
        != (RationalSample {
            numerator: 0,
            denominator: 1,
        })
    {
        return invalid("aggregate origin must equal the first planned chunk start".to_owned());
    }
    if geometry.window_planning.kind != WindowPlanningKind::RegularFixedStepV1
        || geometry.window_planning.tail_policy != TailPolicy::PadFinal
        || geometry.window_planning.step_samples == 0
        || geometry.window_planning.step_samples > geometry.window_samples
    {
        return invalid("unsupported window planning policy".to_owned());
    }
    let expected_starts = planned_starts(
        audio.sample_count,
        geometry.window_samples,
        geometry.window_planning.step_samples,
    )?;
    if geometry.chunks.len() as u64 != expected_starts.len() as u64
        || geometry.chunks.len() as u64 > MAX_CHUNKS
    {
        return invalid("chunk count does not match fixed-step planning".to_owned());
    }
    for (index, (chunk, expected_start)) in geometry
        .chunks
        .iter()
        .zip(expected_starts.iter())
        .enumerate()
    {
        if chunk.index != index as u64 || chunk.start_samples != *expected_start {
            return invalid("chunk order or start does not match planning".to_owned());
        }
        let valid = audio
            .sample_count
            .saturating_sub(chunk.start_samples)
            .min(geometry.window_samples);
        if chunk.valid_samples != valid
            || chunk.padding_samples != geometry.window_samples - valid
            || chunk.valid_samples + chunk.padding_samples != geometry.window_samples
        {
            return invalid("chunk valid and padding samples are inconsistent".to_owned());
        }
    }

    let expected_extent = match geometry.output_extent_policy {
        OutputExtentPolicy::AggregateGrid => aggregate_grid_extent(geometry)?,
        OutputExtentPolicy::AudioExtent => OutputExtent {
            end_samples: audio.sample_count,
            start_samples: 0,
        },
    };
    if geometry.output_extent != expected_extent {
        return invalid("output extent does not match its declared policy".to_owned());
    }

    Ok(())
}

fn aggregate_grid_extent(
    geometry: &SegmentationGeometry,
) -> Result<OutputExtent, SegmentationBundleError> {
    if geometry.chunks.is_empty() {
        return Ok(OutputExtent {
            end_samples: 0,
            start_samples: 0,
        });
    }

    let last = geometry.chunks.last().ok_or_else(|| {
        SegmentationBundleError::Invalid("aggregate extent has no chunk start".to_owned())
    })?;

    let window_end = last
        .start_samples
        .checked_add(geometry.window_samples)
        .ok_or_else(|| SegmentationBundleError::Invalid("aggregate extent overflow".to_owned()))?;
    let step = rational_from_sample(&geometry.aggregate_grid.step)?;

    // the reference allocates the full aggregate grid for every planned window, including padded tails
    let raw_last_index = exact_sample(
        rational_extent(CheckedRational::integer(i128::from(window_end)).checked_div(step))?,
        "aggregate frame offset",
    )?;
    let raw_frame_count = raw_last_index.checked_add(1).ok_or_else(|| {
        SegmentationBundleError::Invalid("aggregate frame count overflow".to_owned())
    })?;
    let frame_count = raw_frame_count;
    let end = rational_extent(step.checked_mul(i128::from(frame_count - 1)))?;
    let end =
        rational_extent(end.checked_add(rational_from_sample(&geometry.aggregate_grid.support)?))?;
    let end_samples = exact_sample(end, "aggregate extent end")?;
    Ok(OutputExtent {
        end_samples,
        start_samples: 0,
    })
}

fn planned_starts(
    sample_count: u64,
    window_samples: u64,
    step_samples: u64,
) -> Result<Vec<u64>, SegmentationBundleError> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }
    let last = if sample_count <= window_samples {
        0
    } else {
        let remainder = sample_count - window_samples;
        remainder
            .checked_add(step_samples - 1)
            .ok_or_else(|| SegmentationBundleError::Invalid("window start overflow".to_owned()))?
            / step_samples
            * step_samples
    };
    let count = last / step_samples + 1;
    if count > MAX_CHUNKS {
        return invalid("planned chunk count exceeds the admission bound".to_owned());
    }
    (0..count)
        .map(|index| {
            index
                .checked_mul(step_samples)
                .ok_or_else(|| SegmentationBundleError::Invalid("window start overflow".to_owned()))
        })
        .collect()
}

fn validate_grid(grid: &FrameGrid) -> Result<(), SegmentationBundleError> {
    if grid.frame_count == 0 || u64::from(grid.frame_count) > MAX_FRAMES {
        return invalid("frame grid count is outside the admission bound".to_owned());
    }
    validate_rational(&grid.origin, false, "frame origin")?;
    validate_rational(&grid.step, true, "frame step")?;
    validate_rational(&grid.support, true, "frame support")?;
    Ok(())
}

fn validate_rational(
    value: &RationalSample,
    positive: bool,
    context: &str,
) -> Result<(), SegmentationBundleError> {
    CheckedRational::validate_parts(
        i128::from(value.numerator),
        i128::from(value.denominator),
        positive,
    )
    .map_err(|error| match error {
        RationalError::NonPositiveDenominator => {
            SegmentationBundleError::Invalid(format!("{context} denominator must be positive"))
        }
        RationalError::NonPositiveValue => {
            SegmentationBundleError::Invalid(format!("{context} must be positive"))
        }
        RationalError::NotReduced => {
            SegmentationBundleError::Invalid(format!("{context} must be reduced"))
        }
        RationalError::Overflow
        | RationalError::NonPositiveDivisor
        | RationalError::Negative
        | RationalError::NotExact => {
            SegmentationBundleError::Invalid(format!("{context} is invalid"))
        }
    })
}

fn rational_from_sample(
    value: &RationalSample,
) -> Result<CheckedRational, SegmentationBundleError> {
    CheckedRational::new(i128::from(value.numerator), i128::from(value.denominator)).map_err(
        |error| match error {
            RationalError::NonPositiveDenominator => {
                SegmentationBundleError::Invalid("rational denominator must be positive".to_owned())
            }
            RationalError::Overflow => {
                SegmentationBundleError::Invalid("rational reduction overflow".to_owned())
            }
            RationalError::NonPositiveValue
            | RationalError::NotReduced
            | RationalError::NonPositiveDivisor
            | RationalError::Negative
            | RationalError::NotExact => {
                SegmentationBundleError::Invalid("rational value is invalid".to_owned())
            }
        },
    )
}

fn rational_extent<T>(result: Result<T, RationalError>) -> Result<T, SegmentationBundleError> {
    result.map_err(|error| match error {
        RationalError::NonPositiveDivisor => {
            SegmentationBundleError::Invalid("rational divisor must be positive".to_owned())
        }
        RationalError::Overflow => {
            SegmentationBundleError::Invalid("rational extent overflow".to_owned())
        }
        RationalError::NonPositiveDenominator
        | RationalError::NonPositiveValue
        | RationalError::NotReduced
        | RationalError::Negative
        | RationalError::NotExact => {
            SegmentationBundleError::Invalid("rational extent is invalid".to_owned())
        }
    })
}

fn exact_sample(value: CheckedRational, context: &str) -> Result<u64, SegmentationBundleError> {
    value.to_u64_exact().map_err(|error| match error {
        RationalError::NotExact => SegmentationBundleError::Invalid(format!(
            "{context} is not an exact non-negative sample"
        )),
        RationalError::Overflow => SegmentationBundleError::Invalid(format!("{context} overflows")),
        RationalError::NonPositiveDenominator
        | RationalError::NonPositiveValue
        | RationalError::NotReduced
        | RationalError::NonPositiveDivisor
        | RationalError::Negative => {
            SegmentationBundleError::Invalid(format!("{context} is invalid"))
        }
    })
}

fn frame_count_for_window(
    window_samples: u64,
    support: &RationalSample,
    step: &RationalSample,
) -> Result<u64, SegmentationBundleError> {
    let window = CheckedRational::integer(i128::from(window_samples));
    let support = rational_from_sample(support)?;
    let step = rational_from_sample(step)?;
    let difference = rational_extent(window.checked_sub(support))?;
    if difference.is_negative() {
        return invalid("frame support cannot exceed the window".to_owned());
    }

    let count = rational_extent(difference.checked_div(step))?
        .floor_nonnegative()
        .map_err(|error| match error {
            RationalError::Negative => {
                SegmentationBundleError::Invalid("frame count must be non-negative".to_owned())
            }
            RationalError::Overflow => {
                SegmentationBundleError::Invalid("frame count overflows".to_owned())
            }
            RationalError::NonPositiveDenominator
            | RationalError::NonPositiveValue
            | RationalError::NotReduced
            | RationalError::NonPositiveDivisor
            | RationalError::NotExact => {
                SegmentationBundleError::Invalid("frame count is invalid".to_owned())
            }
        })?
        .checked_add(1)
        .ok_or_else(|| SegmentationBundleError::Invalid("frame count overflows".to_owned()))?;
    u64::try_from(count)
        .map_err(|_| SegmentationBundleError::Invalid("frame count overflows".to_owned()))
}

fn validate_tensors(
    geometry: &SegmentationGeometry,
    head: &SegmentationHead,
    tensors: &TensorInventory,
) -> Result<(), SegmentationBundleError> {
    let frame_count = u64::from(geometry.frame_grid.frame_count);
    let class_count = head.class_to_slot_subsets.len() as u64;
    if class_count == 0 || class_count > MAX_CLASSES {
        return invalid("tensor class extent is outside the admission bound".to_owned());
    }
    let mut next_chunk = 0_u64;
    let mut paths = BTreeSet::new();
    let mut total_elements = 0_u64;
    for shard in &tensors.shards {
        validate_member_path(&shard.path)?;
        if shard.bytes == 0 || shard.bytes > MAX_SHARD_BYTES {
            return invalid(format!("shard {} exceeds byte bounds", shard.path));
        }
        if shard.chunk_start != next_chunk
            || shard.chunk_start >= shard.chunk_end
            || shard.chunk_end > geometry.chunks.len() as u64
        {
            return invalid("tensor chunk ranges must be ordered and contiguous".to_owned());
        }
        if shard.shape[0] != shard.chunk_end - shard.chunk_start
            || shard.shape[1] != frame_count
            || shard.shape[2] != class_count
        {
            return invalid(format!(
                "shard {} shape does not match geometry/head",
                shard.path
            ));
        }
        let elements = shard
            .shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| {
                SegmentationBundleError::Invalid("tensor element count overflow".to_owned())
            })?;
        total_elements = total_elements.checked_add(elements).ok_or_else(|| {
            SegmentationBundleError::Invalid("tensor element count overflow".to_owned())
        })?;
        if total_elements > MAX_TENSOR_ELEMENTS {
            return invalid("tensor element count exceeds the admission bound".to_owned());
        }
        if !paths.insert(&shard.path) {
            return invalid("tensor shard paths must be unique".to_owned());
        }
        next_chunk = shard.chunk_end;
    }
    if next_chunk != geometry.chunks.len() as u64 {
        return invalid("tensor shard ranges do not cover all chunks".to_owned());
    }
    if geometry.chunks.is_empty() && !tensors.shards.is_empty() {
        return invalid("an empty recording cannot have score shards".to_owned());
    }
    Ok(())
}

fn validate_policy(
    head: &SegmentationHead,
    geometry: &SegmentationGeometry,
    policy: &SegmentationPolicy,
) -> Result<(), SegmentationBundleError> {
    for (name, value) in [
        ("policy.decoder.id", &policy.decoder.id),
        ("policy.decoder.revision", &policy.decoder.revision),
        ("policy.filter.id", &policy.filter.id),
        ("policy.filter.revision", &policy.filter.revision),
        ("policy.count.id", &policy.count.id),
        ("policy.count.revision", &policy.count.revision),
        ("policy.embedding.id", &policy.embedding.id),
        ("policy.embedding.revision", &policy.embedding.revision),
        (
            "policy.embedding.embedding_model.id",
            &policy.embedding.embedding_model.id,
        ),
        (
            "policy.embedding.embedding_model.revision",
            &policy.embedding.embedding_model.revision,
        ),
        ("policy.embedding.plda.id", &policy.embedding.plda.id),
        (
            "policy.embedding.plda.revision",
            &policy.embedding.plda.revision,
        ),
        ("policy.reconstruction.id", &policy.reconstruction.id),
        (
            "policy.reconstruction.revision",
            &policy.reconstruction.revision,
        ),
    ] {
        validate_text(value, name)?;
    }
    if policy.decoder.argmax_tie != head.argmax_tie
        || policy.decoder.representation != head.score_representation
    {
        return invalid("decoder policy does not match head score semantics".to_owned());
    }
    if policy.filter.enabled && (policy.filter.width == 0 || policy.filter.width.is_multiple_of(2))
    {
        return invalid("enabled median filter width must be a positive odd number".to_owned());
    }
    if policy.embedding.min_num_samples == 0 || policy.embedding.target_frames == 0 {
        return invalid("embedding policy sample/frame bounds must be positive".to_owned());
    }
    if policy.embedding.target_frames > geometry.frame_grid.frame_count {
        return invalid("embedding target frames exceed the segmentation frame grid".to_owned());
    }
    if policy.reconstruction.extent_policy != geometry.output_extent_policy {
        return invalid("reconstruction extent policy does not match geometry".to_owned());
    }
    Ok(())
}

fn validate_text(value: &str, context: &str) -> Result<(), SegmentationBundleError> {
    if value.is_empty()
        || value.len() > MAX_IDENTITY_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        return invalid(format!("{context} must be non-empty bounded text"));
    }
    Ok(())
}

pub(crate) fn validate_member_path(path: &str) -> Result<(), SegmentationBundleError> {
    if path.is_empty() || path.len() > MAX_MEMBER_PATH_BYTES || path.contains('\0') {
        return invalid("tensor member path is empty or too long".to_owned());
    }
    if !path.ends_with(".npy") {
        return invalid(format!("tensor member path must name an NPY file: {path}"));
    }
    let value = Path::new(path);
    if value.is_absolute()
        || value.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return invalid(format!("unsafe tensor member path {path}"));
    }
    if value
        .components()
        .any(|component| matches!(component, Component::CurDir))
    {
        return invalid(format!("non-canonical tensor member path {path}"));
    }
    Ok(())
}
