use crate::imported_segmentation::{
    AudioIdentity, FrameGrid, OutputExtent, OutputExtentPolicy, RationalSample,
    SegmentationGeometry,
};
use crate::inference::segmentation::{WindowSpec, segmentation_window_count};
use crate::rational::{CheckedRational, RationalError};

const LEGACY_SAMPLE_RATE: u32 = 16_000;
const LEGACY_FRAME_SUPPORT_SAMPLES: i64 = 991;
const LEGACY_FRAME_STEP_SAMPLES: i64 = 270;

/// An error returned when a pipeline geometry cannot be represented exactly
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PipelineGeometryError {
    /// A geometry field violates the checked pipeline contract
    #[error("invalid pipeline geometry: {0}")]
    Invalid(String),
    /// A geometry calculation exceeded its checked integer bounds
    #[error("pipeline geometry overflow: {0}")]
    Overflow(&'static str),
}

/// Exact sample extent for one planned input chunk
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkExtent {
    index: usize,
    start_samples: usize,
    valid_samples: usize,
    padding_samples: usize,
}

impl ChunkExtent {
    /// Returns the zero-based chunk index
    pub const fn index(self) -> usize {
        self.index
    }

    /// Returns the chunk start in canonical audio samples
    pub const fn start_samples(self) -> usize {
        self.start_samples
    }

    /// Returns the number of canonical samples copied into the chunk
    pub const fn valid_samples(self) -> usize {
        self.valid_samples
    }

    /// Returns the number of right-padding samples in the chunk
    pub const fn padding_samples(self) -> usize {
        self.padding_samples
    }
}

/// Exact frame timing used to convert frame activations to seconds
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameTiming {
    sample_rate: u32,
    origin: RationalSample,
    step: RationalSample,
    support: RationalSample,
}

impl FrameTiming {
    /// Create checked timing from a frame grid and sample rate
    pub fn from_grid(sample_rate: u32, grid: &FrameGrid) -> Result<Self, PipelineGeometryError> {
        validate_sample_rate(sample_rate)?;
        validate_rational(&grid.origin, false, "frame origin")?;
        validate_rational(&grid.step, true, "frame step")?;
        validate_rational(&grid.support, true, "frame support")?;
        Ok(Self {
            sample_rate,
            origin: grid.origin,
            step: grid.step,
            support: grid.support,
        })
    }

    /// Returns the sample rate used by this timing
    pub const fn sample_rate(self) -> u32 {
        self.sample_rate
    }

    /// Returns the exact first-frame origin in samples
    pub const fn origin(self) -> RationalSample {
        self.origin
    }

    /// Returns the exact frame-to-frame step in samples
    pub const fn step(self) -> RationalSample {
        self.step
    }

    /// Returns the exact receptive-field support in samples
    pub const fn support(self) -> RationalSample {
        self.support
    }

    /// Returns the exact frame midpoint in seconds when the index is representable
    pub fn frame_middle_seconds(self, frame_idx: usize) -> Option<f64> {
        let origin = rational_from_sample(&self.origin).ok()?;
        let step = rational_from_sample(&self.step).ok()?;
        let support = rational_from_sample(&self.support).ok()?;
        let frame_idx = i128::try_from(frame_idx).ok()?;
        let frame_offset =
            rational_result(step.checked_mul(frame_idx), "rational multiplication").ok()?;
        let midpoint =
            rational_result(origin.checked_add(frame_offset), "rational addition").ok()?;
        let support_midpoint = rational_result(
            support.checked_div(CheckedRational::integer(2)),
            "rational division",
        )
        .ok()?;
        let midpoint =
            rational_result(midpoint.checked_add(support_midpoint), "rational addition").ok()?;
        Some(midpoint.to_f64() / f64::from(self.sample_rate))
    }
}

/// Checked geometry shared by native and imported pipeline stages
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineGeometry {
    sample_rate: u32,
    sample_count: usize,
    step_samples: usize,
    window_samples: usize,
    chunks: Vec<ChunkExtent>,
    frame_grid: FrameGrid,
    aggregate_grid: FrameGrid,
    start_frames: Vec<usize>,
    output_frames: usize,
    output_extent: OutputExtent,
    output_extent_policy: OutputExtentPolicy,
}

impl PipelineGeometry {
    /// Build geometry from the existing native model window and audio length
    pub fn from_legacy(
        sample_rate: u32,
        window_samples: u64,
        step_samples: u64,
        audio_samples: u64,
    ) -> Result<Self, PipelineGeometryError> {
        validate_sample_rate(sample_rate)?;
        let window_samples = to_usize(window_samples, "window samples")?;
        let step_samples = to_usize(step_samples, "step samples")?;
        let sample_count = to_usize(audio_samples, "audio samples")?;
        validate_chunk_geometry(window_samples, step_samples)?;

        let frame_grid = legacy_frame_grid(window_samples)?;
        let aggregate_grid = frame_grid.clone();
        let chunks = planned_chunks(sample_count, window_samples, step_samples)?;
        let start_frames = derive_legacy_offsets(&chunks, &aggregate_grid)?;
        let output_frames = derive_legacy_output_frames(&chunks, &aggregate_grid)?;
        let output_extent = output_extent_for_grid(
            &aggregate_grid,
            output_frames,
            OutputExtentPolicy::AggregateGrid,
        )?;

        Ok(Self {
            sample_rate,
            sample_count,
            step_samples,
            window_samples,
            chunks,
            frame_grid,
            aggregate_grid,
            start_frames,
            output_frames,
            output_extent,
            output_extent_policy: OutputExtentPolicy::AggregateGrid,
        })
    }

    /// Build geometry from a validated imported segmentation contract
    pub fn from_imported(
        audio: &AudioIdentity,
        geometry: &SegmentationGeometry,
    ) -> Result<Self, PipelineGeometryError> {
        validate_sample_rate(audio.sample_rate)?;
        let sample_count = to_usize(audio.sample_count, "audio samples")?;
        let window_samples = to_usize(geometry.window_samples, "window samples")?;
        let step_samples = to_usize(geometry.window_planning.step_samples, "window step samples")?;
        validate_chunk_geometry(window_samples, step_samples)?;
        validate_imported_grids(geometry, window_samples)?;

        let chunks = imported_chunks(sample_count, window_samples, step_samples, geometry)?;
        let start_frames = derive_imported_offsets(&chunks, &geometry.aggregate_grid)?;
        let output_frames = derive_imported_output_frames(sample_count, &chunks, geometry)?;
        let output_extent = expected_imported_extent(sample_count, &chunks, geometry)?;

        Ok(Self {
            sample_rate: audio.sample_rate,
            sample_count,
            step_samples,
            window_samples,
            chunks,
            frame_grid: geometry.frame_grid.clone(),
            aggregate_grid: geometry.aggregate_grid.clone(),
            start_frames,
            output_frames,
            output_extent,
            output_extent_policy: geometry.output_extent_policy,
        })
    }

    /// Build legacy geometry from a segmentation model specification
    #[cfg(feature = "coreml")]
    pub(crate) fn from_spec(
        spec: WindowSpec,
        audio_samples: usize,
    ) -> Result<Self, PipelineGeometryError> {
        Self::from_legacy(
            LEGACY_SAMPLE_RATE,
            spec.window_samples() as u64,
            spec.step_samples() as u64,
            audio_samples as u64,
        )
    }

    /// Returns the sample rate of the canonical audio
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the canonical audio sample count used for planning
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the number of samples in every padded input window
    pub const fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Returns the step between planned input windows in samples
    pub const fn step_samples(&self) -> usize {
        self.step_samples
    }

    /// Returns the explicit ordered chunk extents
    pub fn chunks(&self) -> &[ChunkExtent] {
        &self.chunks
    }

    /// Returns the number of planned input chunks
    pub const fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the local receptive-field frame grid
    pub const fn frame_grid(&self) -> &FrameGrid {
        &self.frame_grid
    }

    /// Returns the aggregate/output frame grid
    pub const fn aggregate_grid(&self) -> &FrameGrid {
        &self.aggregate_grid
    }

    /// Returns the exact chunk-to-global-frame offsets
    pub fn start_frames(&self) -> &[usize] {
        &self.start_frames
    }

    /// Returns the number of frames in the aggregate/output track
    pub const fn output_frames(&self) -> usize {
        self.output_frames
    }

    /// Returns the derived output sample extent
    pub const fn output_extent(&self) -> &OutputExtent {
        &self.output_extent
    }

    /// Returns the policy used to select the output extent
    pub const fn output_extent_policy(&self) -> OutputExtentPolicy {
        self.output_extent_policy
    }

    /// Returns timing for converting aggregate frames to seconds
    pub fn frame_timing(&self) -> Result<FrameTiming, PipelineGeometryError> {
        FrameTiming::from_grid(self.sample_rate, &self.aggregate_grid)
    }

    pub(crate) fn frame_count_for_duration(&self, duration_seconds: f64) -> f32 {
        let step = rational_from_sample(&self.aggregate_grid.step)
            .expect("pipeline geometry has a valid aggregate frame step");
        (duration_seconds * f64::from(self.sample_rate) / step.to_f64()).floor() as f32
    }

    /// Returns the chunk audio available in the canonical input
    pub(crate) fn chunk_audio<'a>(&self, audio: &'a [f32], chunk_idx: usize) -> &'a [f32] {
        let Some(chunk) = self.chunks.get(chunk_idx) else {
            return &[];
        };
        let start = chunk.start_samples;
        let end = start.saturating_add(chunk.valid_samples).min(audio.len());
        if start < audio.len() {
            &audio[start..end]
        } else {
            &[]
        }
    }
}

pub(crate) fn chunk_audio_raw(
    audio: &[f32],
    step_samples: usize,
    window_samples: usize,
    chunk_idx: usize,
) -> &[f32] {
    let start = chunk_idx.saturating_mul(step_samples);
    let end = start.saturating_add(window_samples).min(audio.len());
    if start < audio.len() {
        &audio[start..end]
    } else {
        &[]
    }
}

#[cfg(test)]
pub(crate) fn chunk_start_frames(num_chunks: usize, step_seconds: f64) -> Vec<usize> {
    let step_samples = (step_seconds * f64::from(LEGACY_SAMPLE_RATE)) as usize;
    let chunks = (0..num_chunks)
        .map(|index| ChunkExtent {
            index,
            start_samples: index.saturating_mul(step_samples),
            valid_samples: 0,
            padding_samples: 0,
        })
        .collect::<Vec<_>>();
    let grid = legacy_frame_grid(160_000).expect("legacy frame constants are valid");
    derive_legacy_offsets(&chunks, &grid).expect("legacy frame offsets are representable")
}

#[cfg(test)]
pub(crate) fn total_output_frames(num_chunks: usize, step_seconds: f64) -> usize {
    let step_samples = (step_seconds * f64::from(LEGACY_SAMPLE_RATE)) as usize;
    let chunks = (0..num_chunks)
        .map(|index| ChunkExtent {
            index,
            start_samples: index.saturating_mul(step_samples),
            valid_samples: 160_000,
            padding_samples: 0,
        })
        .collect::<Vec<_>>();
    let grid = legacy_frame_grid(160_000).expect("legacy frame constants are valid");
    derive_legacy_output_frames(&chunks, &grid).expect("legacy output extent is representable")
}

fn validate_sample_rate(sample_rate: u32) -> Result<(), PipelineGeometryError> {
    if sample_rate != LEGACY_SAMPLE_RATE {
        return Err(PipelineGeometryError::Invalid(format!(
            "pipeline sample rate must be {LEGACY_SAMPLE_RATE}, got {sample_rate}"
        )));
    }
    Ok(())
}

fn validate_chunk_geometry(
    window_samples: usize,
    step_samples: usize,
) -> Result<(), PipelineGeometryError> {
    if window_samples == 0 {
        return Err(PipelineGeometryError::Invalid(
            "window samples must be greater than zero".to_owned(),
        ));
    }
    if step_samples == 0 || step_samples > window_samples {
        return Err(PipelineGeometryError::Invalid(
            "window step must be positive and no larger than the window".to_owned(),
        ));
    }
    Ok(())
}

fn planned_chunks(
    sample_count: usize,
    window_samples: usize,
    step_samples: usize,
) -> Result<Vec<ChunkExtent>, PipelineGeometryError> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }

    let spec = WindowSpec::new(window_samples, step_samples).ok_or_else(|| {
        PipelineGeometryError::Invalid("window and step samples must be non-zero".to_owned())
    })?;
    let chunk_count = segmentation_window_count(sample_count, spec);

    (0..chunk_count)
        .map(|index| {
            let start_samples = index
                .checked_mul(step_samples)
                .ok_or(PipelineGeometryError::Overflow("chunk start"))?;
            let valid_samples = sample_count
                .saturating_sub(start_samples)
                .min(window_samples);
            Ok(ChunkExtent {
                index,
                start_samples,
                valid_samples,
                padding_samples: window_samples - valid_samples,
            })
        })
        .collect()
}

fn imported_chunks(
    sample_count: usize,
    window_samples: usize,
    step_samples: usize,
    geometry: &SegmentationGeometry,
) -> Result<Vec<ChunkExtent>, PipelineGeometryError> {
    let expected = planned_chunks(sample_count, window_samples, step_samples)?;
    if expected.len() != geometry.chunks.len() {
        return Err(PipelineGeometryError::Invalid(format!(
            "imported geometry has {} chunks, expected {}",
            geometry.chunks.len(),
            expected.len()
        )));
    }

    geometry
        .chunks
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (chunk, expected))| {
            let actual_start = to_usize(chunk.start_samples, "chunk start")?;
            let actual_valid = to_usize(chunk.valid_samples, "chunk valid samples")?;
            let actual_padding = to_usize(chunk.padding_samples, "chunk padding samples")?;
            if chunk.index != index as u64
                || actual_start != expected.start_samples
                || actual_valid != expected.valid_samples
                || actual_padding != expected.padding_samples
            {
                return Err(PipelineGeometryError::Invalid(format!(
                    "imported chunk {index} does not match fixed-step audio planning"
                )));
            }
            Ok(ChunkExtent {
                index,
                start_samples: actual_start,
                valid_samples: actual_valid,
                padding_samples: actual_padding,
            })
        })
        .collect()
}

fn legacy_frame_grid(window_samples: usize) -> Result<FrameGrid, PipelineGeometryError> {
    let frame_count = frame_count_for_window(
        window_samples,
        &RationalSample {
            numerator: LEGACY_FRAME_SUPPORT_SAMPLES,
            denominator: 1,
        },
        &RationalSample {
            numerator: LEGACY_FRAME_STEP_SAMPLES,
            denominator: 1,
        },
    )?;
    let frame_count =
        u32::try_from(frame_count).map_err(|_| PipelineGeometryError::Overflow("frame count"))?;
    Ok(FrameGrid {
        frame_count,
        origin: RationalSample {
            numerator: 0,
            denominator: 1,
        },
        step: RationalSample {
            numerator: LEGACY_FRAME_STEP_SAMPLES,
            denominator: 1,
        },
        support: RationalSample {
            numerator: LEGACY_FRAME_SUPPORT_SAMPLES,
            denominator: 1,
        },
    })
}

fn validate_imported_grids(
    geometry: &SegmentationGeometry,
    window_samples: usize,
) -> Result<(), PipelineGeometryError> {
    validate_grid(&geometry.frame_grid)?;
    validate_grid(&geometry.aggregate_grid)?;
    if geometry.frame_grid.frame_count != geometry.aggregate_grid.frame_count
        || geometry.frame_grid.step != geometry.aggregate_grid.step
        || geometry.frame_grid.support != geometry.aggregate_grid.support
    {
        return Err(PipelineGeometryError::Invalid(
            "imported receptive and aggregate grids must share frame count, step, and support"
                .to_owned(),
        ));
    }
    let expected_frame_count = frame_count_for_window(
        window_samples,
        &geometry.frame_grid.support,
        &geometry.frame_grid.step,
    )?;
    if u64::from(geometry.frame_grid.frame_count) != expected_frame_count {
        return Err(PipelineGeometryError::Invalid(
            "imported frame count does not match its window geometry".to_owned(),
        ));
    }
    Ok(())
}

fn validate_grid(grid: &FrameGrid) -> Result<(), PipelineGeometryError> {
    if grid.frame_count == 0 {
        return Err(PipelineGeometryError::Invalid(
            "frame grid must contain at least one frame".to_owned(),
        ));
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
) -> Result<(), PipelineGeometryError> {
    CheckedRational::validate_parts(
        i128::from(value.numerator),
        i128::from(value.denominator),
        positive,
    )
    .map_err(|error| match error {
        RationalError::NonPositiveDenominator => {
            PipelineGeometryError::Invalid(format!("{context} denominator must be positive"))
        }
        RationalError::NonPositiveValue => {
            PipelineGeometryError::Invalid(format!("{context} must be positive"))
        }
        RationalError::NotReduced => {
            PipelineGeometryError::Invalid(format!("{context} must be reduced"))
        }
        RationalError::Overflow
        | RationalError::NonPositiveDivisor
        | RationalError::Negative
        | RationalError::NotExact => {
            PipelineGeometryError::Invalid(format!("{context} is invalid"))
        }
    })
}

fn derive_legacy_offsets(
    chunks: &[ChunkExtent],
    grid: &FrameGrid,
) -> Result<Vec<usize>, PipelineGeometryError> {
    chunks
        .iter()
        .map(|chunk| {
            let start = CheckedRational::integer(
                i128::try_from(chunk.start_samples)
                    .map_err(|_| PipelineGeometryError::Overflow("chunk start"))?,
            );
            let origin = rational_from_sample(&grid.origin)?;
            let step = rational_from_sample(&grid.step)?;
            let relative = rational_result(start.checked_sub(origin), "rational subtraction")?;
            let offset = rational_result(relative.checked_div(step), "rational division")?;
            let offset = rational_result(offset.round_ties_even(), "nearest-even rounding")?;
            usize::try_from(offset).map_err(|_| PipelineGeometryError::Overflow("frame offset"))
        })
        .collect()
}

fn derive_imported_offsets(
    chunks: &[ChunkExtent],
    grid: &FrameGrid,
) -> Result<Vec<usize>, PipelineGeometryError> {
    chunks
        .iter()
        .map(|chunk| {
            let start = CheckedRational::integer(
                i128::try_from(chunk.start_samples)
                    .map_err(|_| PipelineGeometryError::Overflow("chunk start"))?,
            );
            let origin = rational_from_sample(&grid.origin)?;
            let step = rational_from_sample(&grid.step)?;
            let relative = rational_result(start.checked_sub(origin), "rational subtraction")?;
            let offset = rational_result(relative.checked_div(step), "rational division")?;
            let offset = exact_frame_index(offset, "frame offset")?;
            Ok(offset)
        })
        .collect()
}

fn derive_legacy_output_frames(
    chunks: &[ChunkExtent],
    grid: &FrameGrid,
) -> Result<usize, PipelineGeometryError> {
    let Some(last) = chunks.last() else {
        return Ok(0);
    };
    let endpoint = i128::try_from(last.start_samples)
        .map_err(|_| PipelineGeometryError::Overflow("output extent"))?
        .checked_add(
            i128::try_from(last.valid_samples + last.padding_samples)
                .map_err(|_| PipelineGeometryError::Overflow("output extent"))?,
        )
        .ok_or(PipelineGeometryError::Overflow("output extent"))?;
    let origin = rational_from_sample(&grid.origin)?;
    let step = rational_from_sample(&grid.step)?;
    let frame_index = rational_result(
        CheckedRational::integer(endpoint).checked_sub(origin),
        "rational subtraction",
    )?;
    let frame_index = rational_result(frame_index.checked_div(step), "rational division")?;
    let frame_index = rational_result(frame_index.round_ties_even(), "nearest-even rounding")?;
    usize::try_from(
        frame_index
            .checked_add(1)
            .ok_or(PipelineGeometryError::Overflow("output frame count"))?,
    )
    .map_err(|_| PipelineGeometryError::Overflow("output frame count"))
}

fn derive_imported_output_frames(
    sample_count: usize,
    chunks: &[ChunkExtent],
    geometry: &SegmentationGeometry,
) -> Result<usize, PipelineGeometryError> {
    match geometry.output_extent_policy {
        OutputExtentPolicy::AggregateGrid => {
            let Some(last) = chunks.last() else {
                return Ok(0);
            };
            let endpoint = i128::try_from(last.start_samples)
                .map_err(|_| PipelineGeometryError::Overflow("output extent"))?
                .checked_add(
                    i128::try_from(last.valid_samples + last.padding_samples)
                        .map_err(|_| PipelineGeometryError::Overflow("output extent"))?,
                )
                .ok_or(PipelineGeometryError::Overflow("output extent"))?;
            let origin = rational_from_sample(&geometry.aggregate_grid.origin)?;
            let step = rational_from_sample(&geometry.aggregate_grid.step)?;
            let frame_index = rational_result(
                CheckedRational::integer(endpoint).checked_sub(origin),
                "rational subtraction",
            )?;
            let frame_index = rational_result(frame_index.checked_div(step), "rational division")?;
            let frame_index = exact_frame_index(frame_index, "aggregate output frame index")?;
            frame_index
                .checked_add(1)
                .ok_or(PipelineGeometryError::Overflow("output frame count"))
        }
        OutputExtentPolicy::AudioExtent => {
            if sample_count == 0 {
                return Ok(0);
            }
            let origin = rational_from_sample(&geometry.aggregate_grid.origin)?;
            let step = rational_from_sample(&geometry.aggregate_grid.step)?;
            let end = CheckedRational::integer(
                i128::try_from(sample_count)
                    .map_err(|_| PipelineGeometryError::Overflow("audio extent"))?,
            );
            let frames = rational_result(end.checked_sub(origin), "rational subtraction")?;
            let frames = rational_result(frames.checked_div(step), "rational division")?;
            let frames = rational_result(frames.ceil_nonnegative(), "audio output frame count")?;
            usize::try_from(frames)
                .map_err(|_| PipelineGeometryError::Overflow("output frame count"))
        }
    }
}

fn expected_imported_extent(
    sample_count: usize,
    chunks: &[ChunkExtent],
    geometry: &SegmentationGeometry,
) -> Result<OutputExtent, PipelineGeometryError> {
    let expected = match geometry.output_extent_policy {
        OutputExtentPolicy::AggregateGrid => {
            let output_frames = derive_imported_output_frames(sample_count, chunks, geometry)?;
            output_extent_for_grid(
                &geometry.aggregate_grid,
                output_frames,
                OutputExtentPolicy::AggregateGrid,
            )?
        }
        OutputExtentPolicy::AudioExtent => OutputExtent {
            start_samples: 0,
            end_samples: sample_count as u64,
        },
    };
    if expected != geometry.output_extent {
        return Err(PipelineGeometryError::Invalid(
            "imported output extent does not match its policy".to_owned(),
        ));
    }
    Ok(expected)
}

fn output_extent_for_grid(
    grid: &FrameGrid,
    output_frames: usize,
    policy: OutputExtentPolicy,
) -> Result<OutputExtent, PipelineGeometryError> {
    if !matches!(policy, OutputExtentPolicy::AggregateGrid) || output_frames == 0 {
        return Ok(OutputExtent {
            start_samples: 0,
            end_samples: 0,
        });
    }
    let origin = rational_from_sample(&grid.origin)?;
    let step = rational_from_sample(&grid.step)?;
    let support = rational_from_sample(&grid.support)?;
    let frame_offset = rational_result(
        step.checked_mul(
            i128::try_from(output_frames - 1)
                .map_err(|_| PipelineGeometryError::Overflow("output extent"))?,
        ),
        "rational multiplication",
    )?;
    let endpoint = rational_result(origin.checked_add(frame_offset), "rational addition")?;
    let endpoint = rational_result(endpoint.checked_add(support), "rational addition")?;
    let endpoint = exact_sample(endpoint, "output extent")?;
    Ok(OutputExtent {
        start_samples: 0,
        end_samples: endpoint,
    })
}

fn frame_count_for_window(
    window_samples: usize,
    support: &RationalSample,
    step: &RationalSample,
) -> Result<u64, PipelineGeometryError> {
    let window = CheckedRational::integer(
        i128::try_from(window_samples).map_err(|_| PipelineGeometryError::Overflow("window"))?,
    );
    let support = rational_from_sample(support)?;
    let step = rational_from_sample(step)?;
    let difference = rational_result(window.checked_sub(support), "rational subtraction")?;
    if difference.is_negative() {
        return Err(PipelineGeometryError::Invalid(
            "frame support cannot exceed the window".to_owned(),
        ));
    }
    let count = rational_result(difference.checked_div(step), "rational division")?;
    let count = rational_result(count.floor_nonnegative(), "frame count")?
        .checked_add(1)
        .ok_or(PipelineGeometryError::Overflow("frame count"))?;
    u64::try_from(count).map_err(|_| PipelineGeometryError::Overflow("frame count"))
}

fn to_usize(value: u64, context: &'static str) -> Result<usize, PipelineGeometryError> {
    usize::try_from(value).map_err(|_| PipelineGeometryError::Overflow(context))
}

fn rational_from_sample(value: &RationalSample) -> Result<CheckedRational, PipelineGeometryError> {
    CheckedRational::new(i128::from(value.numerator), i128::from(value.denominator))
        .map_err(|error| map_rational_error(error, "rational reduction"))
}

fn rational_result<T>(
    result: Result<T, RationalError>,
    context: &'static str,
) -> Result<T, PipelineGeometryError> {
    result.map_err(|error| map_rational_error(error, context))
}

fn map_rational_error(error: RationalError, context: &'static str) -> PipelineGeometryError {
    match error {
        RationalError::NonPositiveDenominator => {
            PipelineGeometryError::Invalid("rational denominator must be positive".to_owned())
        }
        RationalError::NonPositiveValue => {
            PipelineGeometryError::Invalid("rational value must be positive".to_owned())
        }
        RationalError::NotReduced => {
            PipelineGeometryError::Invalid("rational value must be reduced".to_owned())
        }
        RationalError::NonPositiveDivisor => {
            PipelineGeometryError::Invalid("rational divisor must be positive".to_owned())
        }
        RationalError::Negative => {
            PipelineGeometryError::Invalid(format!("{context} must be non-negative"))
        }
        RationalError::NotExact => {
            PipelineGeometryError::Invalid(format!("{context} is not exact"))
        }
        RationalError::Overflow => PipelineGeometryError::Overflow(context),
    }
}

fn exact_frame_index(
    value: CheckedRational,
    context: &'static str,
) -> Result<usize, PipelineGeometryError> {
    value.to_usize_exact().map_err(|error| match error {
        RationalError::NotExact => PipelineGeometryError::Invalid(format!(
            "{context} is not an exact non-negative frame index"
        )),
        error => map_rational_error(error, context),
    })
}

fn exact_sample(
    value: CheckedRational,
    context: &'static str,
) -> Result<u64, PipelineGeometryError> {
    value.to_u64_exact().map_err(|error| match error {
        RationalError::NotExact => {
            PipelineGeometryError::Invalid(format!("{context} is not an exact non-negative sample"))
        }
        error => map_rational_error(error, context),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imported_segmentation::{ChunkGeometry, SegmentationManifest};

    #[test]
    fn legacy_offsets_preserve_nearest_even_policy() {
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 207_999).unwrap();
        assert_eq!(geometry.start_frames(), &[0, 59, 119, 178]);
        assert_eq!(geometry.output_frames(), 771);
    }

    #[test]
    fn legacy_planning_keeps_the_padded_tail_at_exact_alignment() {
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 192_000).unwrap();

        assert_eq!(geometry.chunk_count(), 4);
        assert_eq!(
            geometry
                .chunks()
                .iter()
                .map(|chunk| (
                    chunk.start_samples(),
                    chunk.valid_samples(),
                    chunk.padding_samples()
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 160_000, 0),
                (16_000, 160_000, 0),
                (32_000, 160_000, 0),
                (48_000, 144_000, 16_000),
            ]
        );
    }

    #[test]
    fn nearest_even_rounding_handles_both_ties() {
        assert_eq!(CheckedRational::new(1, 2).unwrap().round_ties_even(), Ok(0));
        assert_eq!(CheckedRational::new(3, 2).unwrap().round_ties_even(), Ok(2));
        assert_eq!(CheckedRational::new(5, 2).unwrap().round_ties_even(), Ok(2));
    }

    #[test]
    fn legacy_half_frame_chunk_placement_uses_nearest_even() {
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 135, 160_404).unwrap();

        assert_eq!(geometry.start_frames(), &[0, 0, 1, 2]);
    }

    #[test]
    fn empty_short_and_padded_layouts_have_explicit_extents() {
        let empty = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 0).unwrap();
        assert_eq!(empty.chunks(), &[]);
        assert_eq!(empty.output_frames(), 0);

        let short = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 1).unwrap();
        assert_eq!(short.chunks()[0].valid_samples(), 1);
        assert_eq!(short.chunks()[0].padding_samples(), 159_999);
        assert_eq!(short.output_frames(), 594);
    }

    #[test]
    fn imported_geometry_derives_exact_overlapping_offsets_and_padded_tail() {
        let mut manifest = SegmentationManifest::from_json(include_bytes!(
            "../../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        manifest.audio.sample_count = 150_000;
        manifest.geometry.chunks = vec![
            ChunkGeometry {
                index: 0,
                padding_samples: 0,
                start_samples: 0,
                valid_samples: 128_000,
            },
            ChunkGeometry {
                index: 1,
                padding_samples: 0,
                start_samples: 12_800,
                valid_samples: 128_000,
            },
            ChunkGeometry {
                index: 2,
                padding_samples: 3_600,
                start_samples: 25_600,
                valid_samples: 124_400,
            },
        ];
        manifest.geometry.output_extent.end_samples = 154_000;

        let geometry =
            PipelineGeometry::from_imported(&manifest.audio, &manifest.geometry).unwrap();

        assert_eq!(geometry.start_frames(), &[0, 40, 80]);
        assert_eq!(geometry.output_frames(), 481);
        assert_eq!(geometry.chunks()[2].padding_samples(), 3_600);
        assert_eq!(geometry.output_extent().end_samples, 154_000);
    }

    #[test]
    fn imported_fractional_receptive_origin_is_preserved() {
        let manifest = SegmentationManifest::from_json(include_bytes!(
            "../../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        let geometry =
            PipelineGeometry::from_imported(&manifest.audio, &manifest.geometry).unwrap();

        assert_eq!(geometry.frame_grid().origin.numerator, -241);
        assert_eq!(geometry.frame_grid().origin.denominator, 2);
        assert_eq!(geometry.aggregate_grid().origin.numerator, 0);
        assert_eq!(geometry.output_frames(), 0);
    }

    #[test]
    fn imported_geometry_rejects_non_integral_chunk_placement() {
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
        manifest.geometry.aggregate_grid.origin = RationalSample {
            numerator: 1,
            denominator: 2,
        };

        let error =
            PipelineGeometry::from_imported(&manifest.audio, &manifest.geometry).unwrap_err();

        assert!(error.to_string().contains("exact non-negative frame index"));
    }

    #[test]
    fn imported_audio_extent_is_an_explicit_normalized_view() {
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
        manifest.geometry.output_extent.end_samples = 1;
        manifest.geometry.output_extent_policy = OutputExtentPolicy::AudioExtent;

        let geometry =
            PipelineGeometry::from_imported(&manifest.audio, &manifest.geometry).unwrap();

        assert_eq!(
            geometry.output_extent_policy(),
            OutputExtentPolicy::AudioExtent
        );
        assert_eq!(geometry.output_extent().end_samples, 1);
        assert_eq!(geometry.output_frames(), 1);
    }
}
