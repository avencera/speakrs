use std::collections::{HashMap, HashSet};

use color_eyre::eyre::{Result, ensure};

/// Number of paired-file bootstrap resamples in a Phase 1 result
pub(super) const BOOTSTRAP_SAMPLE_COUNT: usize = 10_000;

/// Lower bound for the dataset noise margin, in DER percentage points
pub(super) const NOISE_MARGIN_FLOOR_PERCENT: f64 = 0.1;

/// Minimum speed improvement accepted by the material-speed rule, in percent
pub(super) const MATERIAL_SPEED_MIN_PERCENT: f64 = 3.0;

/// Multiplier applied to MAD by the material-speed rule
pub(super) const MATERIAL_SPEED_MAD_MULTIPLIER: f64 = 3.0;

const CONFIDENCE_LEVEL: f64 = 0.95;
const LOWER_QUANTILE: f64 = (1.0 - CONFIDENCE_LEVEL) / 2.0;
const UPPER_QUANTILE: f64 = 1.0 - LOWER_QUANTILE;
const REFERENCE_DURATION_ABSOLUTE_TOLERANCE: f64 = 1e-9;
const REFERENCE_DURATION_RELATIVE_TOLERANCE: f64 = 1e-12;
const COMPARISON_ROUNDING_TOLERANCE_PERCENT: f64 = 1e-12;

/// A validated file identity used to pair baseline and candidate measurements
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct FileId(String);

impl FileId {
    /// Creates a file identity from a non-empty value
    pub(super) fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        ensure!(!value.trim().is_empty(), "file id must not be empty");

        Ok(Self(value))
    }

    /// Returns the original file identity
    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

/// A positive reference duration used as a DER denominator
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(super) struct ReferenceDuration(f64);

impl ReferenceDuration {
    /// Creates a positive, finite reference duration
    pub(super) fn new(seconds: f64) -> Result<Self> {
        ensure!(
            seconds.is_finite() && seconds > 0.0,
            "reference duration must be finite and greater than zero"
        );

        Ok(Self(seconds))
    }

    /// Returns the duration in seconds
    pub(super) const fn as_f64(self) -> f64 {
        self.0
    }

    fn is_equivalent(self, other: Self) -> bool {
        let tolerance = REFERENCE_DURATION_ABSOLUTE_TOLERANCE
            .max(self.0.abs().max(other.0.abs()) * REFERENCE_DURATION_RELATIVE_TOLERANCE);
        (self.0 - other.0).abs() <= tolerance
    }
}

/// A finite, non-negative DER numerator
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(super) struct DerNumerator(f64);

impl DerNumerator {
    /// Creates a finite, non-negative DER numerator
    pub(super) fn new(value: f64) -> Result<Self> {
        ensure!(
            value.is_finite() && value >= 0.0,
            "DER numerator must be finite and non-negative"
        );

        Ok(Self(value))
    }

    /// Returns the numerator value
    pub(super) const fn as_f64(self) -> f64 {
        self.0
    }
}

/// One file's DER numerator and reference denominator for a single run
#[derive(Clone, Debug, PartialEq)]
pub(super) struct FileDerObservation {
    file_id: FileId,
    reference_duration: ReferenceDuration,
    der_numerator: DerNumerator,
}

impl FileDerObservation {
    /// Creates a validated per-file DER observation
    pub(super) fn new(
        file_id: impl Into<String>,
        reference_duration: f64,
        der_numerator: f64,
    ) -> Result<Self> {
        Ok(Self {
            file_id: FileId::new(file_id)?,
            reference_duration: ReferenceDuration::new(reference_duration)?,
            der_numerator: DerNumerator::new(der_numerator)?,
        })
    }

    /// Returns the file identity
    pub(super) fn file_id(&self) -> &str {
        self.file_id.as_str()
    }
}

/// A baseline and candidate DER observation paired by one file identity
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PairedFileObservation {
    file_id: FileId,
    reference_duration: ReferenceDuration,
    baseline_der_numerator: DerNumerator,
    candidate_der_numerator: DerNumerator,
}

impl PairedFileObservation {
    /// Pairs two observations after checking identity and denominator equality
    pub(super) fn from_observations(
        baseline: FileDerObservation,
        candidate: FileDerObservation,
    ) -> Result<Self> {
        ensure!(
            baseline.file_id == candidate.file_id,
            "baseline and candidate file identities do not match: '{}' vs '{}'",
            baseline.file_id(),
            candidate.file_id()
        );
        ensure!(
            baseline
                .reference_duration
                .is_equivalent(candidate.reference_duration),
            "reference duration differs for paired file '{}'",
            baseline.file_id()
        );

        Ok(Self {
            file_id: baseline.file_id,
            reference_duration: baseline.reference_duration,
            baseline_der_numerator: baseline.der_numerator,
            candidate_der_numerator: candidate.der_numerator,
        })
    }

    /// Returns the file identity
    pub(super) fn file_id(&self) -> &str {
        self.file_id.as_str()
    }

    /// Returns the reference duration in seconds
    pub(super) const fn reference_duration(&self) -> f64 {
        self.reference_duration.as_f64()
    }

    /// Returns the baseline DER numerator
    pub(super) const fn baseline_der_numerator(&self) -> f64 {
        self.baseline_der_numerator.as_f64()
    }

    /// Returns the candidate DER numerator
    pub(super) const fn candidate_der_numerator(&self) -> f64 {
        self.candidate_der_numerator.as_f64()
    }
}

#[cfg(test)]
impl PairedFileObservation {
    fn new(
        file_id: impl Into<String>,
        reference_duration: f64,
        baseline_der_numerator: f64,
        candidate_der_numerator: f64,
    ) -> Result<Self> {
        Ok(Self {
            file_id: FileId::new(file_id)?,
            reference_duration: ReferenceDuration::new(reference_duration)?,
            baseline_der_numerator: DerNumerator::new(baseline_der_numerator)?,
            candidate_der_numerator: DerNumerator::new(candidate_der_numerator)?,
        })
    }
}

/// Pairs baseline and candidate observations by file identity
pub(super) fn pair_file_observations(
    baseline: &[FileDerObservation],
    candidate: &[FileDerObservation],
) -> Result<Vec<PairedFileObservation>> {
    ensure!(
        !baseline.is_empty(),
        "baseline observations must not be empty"
    );
    ensure!(
        !candidate.is_empty(),
        "candidate observations must not be empty"
    );
    ensure!(
        baseline.len() == candidate.len(),
        "baseline and candidate observation counts differ: {} vs {}",
        baseline.len(),
        candidate.len()
    );

    let candidate_by_id = observations_by_id(candidate, "candidate")?;
    let baseline_by_id = observations_by_id(baseline, "baseline")?;
    ensure!(
        baseline_by_id.len() == candidate_by_id.len(),
        "baseline and candidate file identity sets differ"
    );

    let mut paired = Vec::with_capacity(baseline.len());
    for baseline_observation in baseline {
        let candidate_observation = candidate_by_id
            .get(baseline_observation.file_id())
            .ok_or_else(|| {
                color_eyre::eyre::eyre!(
                    "candidate observations are missing file '{}'",
                    baseline_observation.file_id()
                )
            })?;
        paired.push(PairedFileObservation::from_observations(
            baseline_observation.clone(),
            (*candidate_observation).clone(),
        )?);
    }

    Ok(paired)
}

fn observations_by_id<'a>(
    observations: &'a [FileDerObservation],
    label: &str,
) -> Result<HashMap<&'a str, &'a FileDerObservation>> {
    let mut by_id = HashMap::with_capacity(observations.len());
    for observation in observations {
        ensure!(
            by_id.insert(observation.file_id(), observation).is_none(),
            "{label} observations contain duplicate file '{}'",
            observation.file_id()
        );
    }

    Ok(by_id)
}

/// Computes the median without changing the input slice
pub(super) fn median(values: &[f64]) -> Result<f64> {
    ensure!(!values.is_empty(), "median requires at least one value");
    ensure_finite_values(values, "median value")?;

    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let result = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };
    ensure!(result.is_finite(), "median result is not finite");

    Ok(result)
}

/// Computes the unscaled median absolute deviation
pub(super) fn median_absolute_deviation(values: &[f64]) -> Result<f64> {
    let center = median(values)?;
    let deviations: Vec<_> = values.iter().map(|value| (value - center).abs()).collect();

    median(&deviations)
}

/// Computes the speed improvement from two positive durations, in percent
pub(super) fn speed_improvement_percent(
    baseline_seconds: f64,
    candidate_seconds: f64,
) -> Result<f64> {
    ensure!(
        baseline_seconds.is_finite() && baseline_seconds > 0.0,
        "baseline duration must be finite and greater than zero"
    );
    ensure!(
        candidate_seconds.is_finite() && candidate_seconds > 0.0,
        "candidate duration must be finite and greater than zero"
    );

    let improvement = (baseline_seconds / candidate_seconds - 1.0) * 100.0;
    ensure!(improvement.is_finite(), "speed improvement is not finite");

    Ok(improvement)
}

/// Applies the Phase 1 material-speed rule in percentage points
pub(super) fn material_speed_rule(
    speed_improvement_percent: f64,
    mad_percent: f64,
) -> Result<bool> {
    ensure!(
        speed_improvement_percent.is_finite(),
        "speed improvement must be finite"
    );
    ensure!(
        mad_percent.is_finite() && mad_percent >= 0.0,
        "speed MAD must be finite and non-negative"
    );

    Ok(speed_improvement_percent > MATERIAL_SPEED_MIN_PERCENT
        && speed_improvement_percent > MATERIAL_SPEED_MAD_MULTIPLIER * mad_percent)
}

/// Computes the dataset noise margin from baseline repeat differences
pub(super) fn dataset_noise_margin(baseline_repeat_differences: &[f64]) -> Result<f64> {
    let mut largest_absolute_difference = 0.0_f64;
    for (index, difference) in baseline_repeat_differences.iter().enumerate() {
        ensure!(
            difference.is_finite(),
            "baseline repeat difference at index {index} must be finite"
        );
        largest_absolute_difference = largest_absolute_difference.max(difference.abs());
    }

    Ok(NOISE_MARGIN_FLOOR_PERCENT.max(largest_absolute_difference))
}

/// A closed interval estimated from a bootstrap distribution
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ConfidenceInterval {
    lower: f64,
    upper: f64,
}

impl ConfidenceInterval {
    /// Creates a finite ordered confidence interval
    pub(super) fn new(lower: f64, upper: f64) -> Result<Self> {
        ensure!(
            lower.is_finite() && upper.is_finite(),
            "confidence interval bounds must be finite"
        );
        ensure!(
            lower <= upper,
            "confidence interval lower bound must not exceed upper bound"
        );

        Ok(Self { lower, upper })
    }

    /// Returns the lower confidence bound
    pub(super) const fn lower(self) -> f64 {
        self.lower
    }

    /// Returns the upper confidence bound
    pub(super) const fn upper(self) -> f64 {
        self.upper
    }
}

/// Point estimates and 95 percent intervals from paired-file DER bootstrap
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PairedBootstrapResult {
    seed: u64,
    sample_count: usize,
    baseline_der_percent: f64,
    candidate_der_percent: f64,
    improvement_percent: f64,
    baseline_interval: ConfidenceInterval,
    candidate_interval: ConfidenceInterval,
    improvement_interval: ConfidenceInterval,
}

impl PairedBootstrapResult {
    /// Returns the explicit seed used for resampling
    pub(super) const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the number of bootstrap resamples
    pub(super) const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the duration-weighted baseline DER in percent
    pub(super) const fn baseline_der_percent(&self) -> f64 {
        self.baseline_der_percent
    }

    /// Returns the duration-weighted candidate DER in percent
    pub(super) const fn candidate_der_percent(&self) -> f64 {
        self.candidate_der_percent
    }

    /// Returns candidate improvement over baseline in DER percentage points
    pub(super) const fn improvement_percent(&self) -> f64 {
        self.improvement_percent
    }

    /// Returns the 95 percent baseline DER interval
    pub(super) const fn baseline_interval(&self) -> ConfidenceInterval {
        self.baseline_interval
    }

    /// Returns the 95 percent candidate DER interval
    pub(super) const fn candidate_interval(&self) -> ConfidenceInterval {
        self.candidate_interval
    }

    /// Returns the 95 percent improvement interval
    pub(super) const fn improvement_interval(&self) -> ConfidenceInterval {
        self.improvement_interval
    }

    /// Returns true when the 95 percent interval shows a clear improvement
    pub(super) fn clear_improvement(&self) -> bool {
        clear_improvement(&self.improvement_interval)
    }

    /// Returns whether the improvement is non-inferior to a noise margin
    pub(super) fn non_inferior(&self, noise_margin_percent: f64) -> Result<bool> {
        non_inferior(&self.improvement_interval, noise_margin_percent)
    }
}

/// Performs a deterministic paired-file bootstrap with 10,000 resamples
pub(super) fn paired_file_bootstrap(
    observations: &[PairedFileObservation],
    seed: u64,
) -> Result<PairedBootstrapResult> {
    validate_paired_observations(observations)?;
    let point = aggregate_der(observations)?;

    let mut rng = SplitMix64::new(seed);
    let mut baseline_samples = Vec::with_capacity(BOOTSTRAP_SAMPLE_COUNT);
    let mut candidate_samples = Vec::with_capacity(BOOTSTRAP_SAMPLE_COUNT);
    let mut improvement_samples = Vec::with_capacity(BOOTSTRAP_SAMPLE_COUNT);

    for _ in 0..BOOTSTRAP_SAMPLE_COUNT {
        let sample = resampled_der(observations, &mut rng)?;
        baseline_samples.push(sample.baseline_der_percent);
        candidate_samples.push(sample.candidate_der_percent);
        improvement_samples.push(sample.improvement_percent);
    }

    Ok(PairedBootstrapResult {
        seed,
        sample_count: BOOTSTRAP_SAMPLE_COUNT,
        baseline_der_percent: point.baseline_der_percent,
        candidate_der_percent: point.candidate_der_percent,
        improvement_percent: point.improvement_percent,
        baseline_interval: percentile_interval(&mut baseline_samples)?,
        candidate_interval: percentile_interval(&mut candidate_samples)?,
        improvement_interval: percentile_interval(&mut improvement_samples)?,
    })
}

fn validate_paired_observations(observations: &[PairedFileObservation]) -> Result<()> {
    ensure!(
        !observations.is_empty(),
        "paired observations must not be empty"
    );

    let mut ids = HashSet::with_capacity(observations.len());
    for observation in observations {
        ensure!(
            ids.insert(observation.file_id()),
            "paired observations contain duplicate file '{}'",
            observation.file_id()
        );
        ensure!(
            observation.reference_duration().is_finite() && observation.reference_duration() > 0.0,
            "reference duration for file '{}' must be finite and greater than zero",
            observation.file_id()
        );
        ensure!(
            observation.baseline_der_numerator().is_finite()
                && observation.baseline_der_numerator() >= 0.0,
            "baseline DER numerator for file '{}' must be finite and non-negative",
            observation.file_id()
        );
        ensure!(
            observation.candidate_der_numerator().is_finite()
                && observation.candidate_der_numerator() >= 0.0,
            "candidate DER numerator for file '{}' must be finite and non-negative",
            observation.file_id()
        );
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct DerEstimate {
    baseline_der_percent: f64,
    candidate_der_percent: f64,
    improvement_percent: f64,
}

fn aggregate_der(observations: &[PairedFileObservation]) -> Result<DerEstimate> {
    let mut denominator = 0.0;
    let mut baseline_numerator = 0.0;
    let mut candidate_numerator = 0.0;
    for observation in observations {
        denominator += observation.reference_duration();
        baseline_numerator += observation.baseline_der_numerator();
        candidate_numerator += observation.candidate_der_numerator();
    }
    ensure!(
        denominator.is_finite() && denominator > 0.0,
        "total reference duration must be finite and greater than zero"
    );
    ensure!(
        baseline_numerator.is_finite() && candidate_numerator.is_finite(),
        "DER numerator totals must be finite"
    );

    let baseline_der_percent = baseline_numerator / denominator * 100.0;
    let candidate_der_percent = candidate_numerator / denominator * 100.0;
    let improvement_percent = baseline_der_percent - candidate_der_percent;
    ensure!(
        baseline_der_percent.is_finite()
            && candidate_der_percent.is_finite()
            && improvement_percent.is_finite(),
        "duration-weighted DER result must be finite"
    );

    Ok(DerEstimate {
        baseline_der_percent,
        candidate_der_percent,
        improvement_percent,
    })
}

fn resampled_der(
    observations: &[PairedFileObservation],
    rng: &mut SplitMix64,
) -> Result<DerEstimate> {
    let mut denominator = 0.0;
    let mut baseline_numerator = 0.0;
    let mut candidate_numerator = 0.0;
    for _ in 0..observations.len() {
        let index = rng.index(observations.len());
        let observation = &observations[index];
        denominator += observation.reference_duration();
        baseline_numerator += observation.baseline_der_numerator();
        candidate_numerator += observation.candidate_der_numerator();
    }

    aggregate_der(&[PairedFileObservation {
        file_id: FileId("bootstrap-sample".to_owned()),
        reference_duration: ReferenceDuration(denominator),
        baseline_der_numerator: DerNumerator(baseline_numerator),
        candidate_der_numerator: DerNumerator(candidate_numerator),
    }])
}

fn percentile_interval(values: &mut [f64]) -> Result<ConfidenceInterval> {
    ensure!(
        !values.is_empty(),
        "bootstrap distribution must not be empty"
    );
    ensure_finite_values(values, "bootstrap value")?;
    values.sort_by(f64::total_cmp);

    ConfidenceInterval::new(
        linear_quantile(values, LOWER_QUANTILE),
        linear_quantile(values, UPPER_QUANTILE),
    )
}

fn linear_quantile(sorted_values: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted_values.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let weight = position - lower_index as f64;
    sorted_values[lower_index] + (sorted_values[upper_index] - sorted_values[lower_index]) * weight
}

fn ensure_finite_values(values: &[f64], label: &str) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        ensure!(value.is_finite(), "{label} at index {index} must be finite");
    }

    Ok(())
}

/// Returns true when the lower bound is strictly above zero improvement
pub(super) fn clear_improvement(interval: &ConfidenceInterval) -> bool {
    interval.lower() > 0.0
}

/// Returns true when the lower bound is at or above the negative noise margin
pub(super) fn non_inferior(
    interval: &ConfidenceInterval,
    noise_margin_percent: f64,
) -> Result<bool> {
    ensure!(
        noise_margin_percent.is_finite() && noise_margin_percent >= 0.0,
        "noise margin must be finite and non-negative"
    );

    Ok(interval.lower() + COMPARISON_ROUNDING_TOLERANCE_PERCENT >= -noise_margin_percent)
}

/// A small deterministic generator for reproducible bootstrap indices
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        debug_assert!(length > 0);
        let length = length as u64;
        let limit = u64::MAX - (u64::MAX % length);
        loop {
            let value = self.next_u64();
            if value < limit {
                return (value % length) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_and_even_inputs_without_mutation() {
        let odd = [9.0, 1.0, 4.0];
        assert_eq!(median(&odd).unwrap(), 4.0);
        assert_eq!(odd, [9.0, 1.0, 4.0]);

        let even = [4.0, 1.0, 9.0, 3.0];
        assert_eq!(median(&even).unwrap(), 3.5);
    }

    #[test]
    fn median_rejects_empty_and_non_finite_inputs() {
        assert!(median(&[]).is_err());
        assert!(median(&[1.0, f64::NAN]).is_err());
        assert!(median_absolute_deviation(&[1.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn median_absolute_deviation_uses_the_median_center() {
        let values = [1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0];

        assert_eq!(median_absolute_deviation(&values).unwrap(), 1.0);
    }

    #[test]
    fn material_speed_requires_both_strict_thresholds() {
        assert!(material_speed_rule(3.1, 1.0).unwrap());
        assert!(!material_speed_rule(3.0, 1.0).unwrap());
        assert!(!material_speed_rule(4.0, 2.0).unwrap());
        assert!(material_speed_rule(4.0, 1.0).unwrap());
        assert!(material_speed_rule(-1.0, 1.0).is_ok());
        assert!(material_speed_rule(4.0, -1.0).is_err());
    }

    #[test]
    fn dataset_noise_margin_applies_the_floor() {
        assert_eq!(dataset_noise_margin(&[0.02, -0.05]).unwrap(), 0.1);
        assert_eq!(dataset_noise_margin(&[0.02, -0.15]).unwrap(), 0.15);
        assert_eq!(dataset_noise_margin(&[]).unwrap(), 0.1);
        assert!(dataset_noise_margin(&[f64::NAN]).is_err());
    }

    #[test]
    fn pair_file_observations_matches_ids_in_baseline_order() {
        let baseline = vec![
            FileDerObservation::new("b", 2.0, 0.2).unwrap(),
            FileDerObservation::new("a", 1.0, 0.1).unwrap(),
        ];
        let candidate = vec![
            FileDerObservation::new("a", 1.0, 0.05).unwrap(),
            FileDerObservation::new("b", 2.0, 0.1).unwrap(),
        ];

        let paired = pair_file_observations(&baseline, &candidate).unwrap();

        assert_eq!(paired[0].file_id(), "b");
        assert_eq!(paired[0].candidate_der_numerator(), 0.1);
        assert_eq!(paired[1].file_id(), "a");
        assert!(pair_file_observations(&baseline, &candidate[..1]).is_err());
    }

    #[test]
    fn pairing_rejects_identity_and_denominator_mismatches() {
        let baseline = FileDerObservation::new("file", 2.0, 0.2).unwrap();
        let wrong_id = FileDerObservation::new("other", 2.0, 0.1).unwrap();
        let wrong_duration = FileDerObservation::new("file", 3.0, 0.1).unwrap();

        assert!(PairedFileObservation::from_observations(baseline.clone(), wrong_id).is_err());
        assert!(PairedFileObservation::from_observations(baseline, wrong_duration).is_err());
    }

    #[test]
    fn pairing_accepts_equivalent_floating_point_denominators() {
        let baseline = FileDerObservation::new("file", 2530.2600000000057, 0.2).unwrap();
        let candidate = FileDerObservation::new("file", 2530.2599999999993, 0.1).unwrap();

        PairedFileObservation::from_observations(baseline, candidate).unwrap();
    }

    #[test]
    fn speed_improvement_is_a_percent_change() {
        assert_eq!(speed_improvement_percent(100.0, 80.0).unwrap(), 25.0);
        assert!(speed_improvement_percent(0.0, 1.0).is_err());
    }

    #[test]
    fn paired_bootstrap_is_seeded_and_duration_weighted() {
        let observations = vec![
            PairedFileObservation::new("short", 1.0, 1.0, 0.0).unwrap(),
            PairedFileObservation::new("long", 9.0, 9.0, 8.0).unwrap(),
        ];

        let first = paired_file_bootstrap(&observations, 42).unwrap();
        let second = paired_file_bootstrap(&observations, 42).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.seed(), 42);
        assert_eq!(first.sample_count(), BOOTSTRAP_SAMPLE_COUNT);
        assert_eq!(first.baseline_der_percent(), 100.0);
        assert_eq!(first.candidate_der_percent(), 80.0);
        assert_eq!(first.improvement_percent(), 20.0);
        assert!(first.improvement_interval().lower().is_finite());
        assert!(first.improvement_interval().upper().is_finite());
        assert!(first.clear_improvement());
        assert!(first.non_inferior(0.1).unwrap());
    }

    #[test]
    fn paired_bootstrap_rejects_duplicate_ids_and_invalid_values() {
        let duplicate = vec![
            PairedFileObservation::new("same", 1.0, 0.0, 0.0).unwrap(),
            PairedFileObservation::new("same", 1.0, 0.0, 0.0).unwrap(),
        ];
        assert!(paired_file_bootstrap(&duplicate, 0).is_err());
        assert!(PairedFileObservation::new("file", f64::NAN, 0.0, 0.0).is_err());
        assert!(PairedFileObservation::new("file", 1.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn improvement_predicates_use_their_documented_bounds() {
        let interval = ConfidenceInterval::new(-0.1, 0.2).unwrap();
        assert!(!clear_improvement(&interval));
        assert!(non_inferior(&interval, 0.1).unwrap());
        assert!(non_inferior(&interval, 0.2).unwrap());
        assert!(non_inferior(&interval, 0.0).is_ok());
        let rounding_noise = ConfidenceInterval::new(-f64::EPSILON, f64::EPSILON).unwrap();
        assert!(non_inferior(&rounding_noise, 0.0).unwrap());
        assert!(non_inferior(&interval, -0.1).is_err());
    }
}
