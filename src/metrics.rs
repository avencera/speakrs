use std::collections::{BTreeSet, HashMap};

use crate::segment::Segment;

/// Diarization error rate (DER) breakdown
#[derive(Debug, Clone)]
pub struct DerResult {
    /// Total missed speech duration in seconds
    pub missed: f64,
    /// Total false alarm duration in seconds
    pub false_alarm: f64,
    /// Total speaker confusion duration in seconds
    pub confusion: f64,
    /// Total reference speech duration in seconds
    pub total: f64,
}

impl DerResult {
    /// Compute the overall DER as (missed + false_alarm + confusion) / total
    pub fn der(&self) -> f64 {
        if self.total == 0.0 {
            return 0.0;
        }
        (self.missed + self.false_alarm + self.confusion) / self.total
    }
}

/// Parse RTTM text into segments, ignoring file_id
pub fn parse_rttm(text: &str) -> Vec<Segment> {
    text.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.first() != Some(&"SPEAKER") || parts.len() < 8 {
                return None;
            }
            let start: f64 = parts[3].parse().ok()?;
            let duration: f64 = parts[4].parse().ok()?;
            let speaker = parts[7].to_string();
            Some(Segment::new(start, start + duration, speaker))
        })
        .collect()
}

/// Compute Diarization Error Rate between reference and hypothesis segments
///
/// Uses interval-based NIST standard DER with collar=0 and optimal
/// Speaker mapping via the Jonker-Volgenant algorithm
pub fn compute_der(reference: &[Segment], hypothesis: &[Segment]) -> DerResult {
    if reference.is_empty() {
        let fa: f64 = hypothesis.iter().map(|s| s.duration()).sum();
        return DerResult {
            missed: 0.0,
            false_alarm: fa,
            confusion: 0.0,
            total: 0.0,
        };
    }

    // collect all boundary times
    let mut boundaries = BTreeSet::new();
    for seg in reference.iter().chain(hypothesis.iter()) {
        boundaries.insert(OrderedF64(seg.start));
        boundaries.insert(OrderedF64(seg.end));
    }
    let boundaries: Vec<f64> = boundaries.into_iter().map(|b| b.0).collect();

    // assign integer IDs to speakers
    let ref_speakers = unique_speakers(reference);
    let hyp_speakers = unique_speakers(hypothesis);

    // build co-occurrence matrix for optimal mapping
    let mut cooccurrence = vec![vec![0.0f64; hyp_speakers.len()]; ref_speakers.len()];

    for window in boundaries.windows(2) {
        let (t_start, t_end) = (window[0], window[1]);
        let dt = t_end - t_start;
        if dt <= 0.0 {
            continue;
        }

        let active_ref = active_speakers_at(reference, t_start, t_end, &ref_speakers);
        let active_hyp = active_speakers_at(hypothesis, t_start, t_end, &hyp_speakers);

        for &ri in &active_ref {
            for &hi in &active_hyp {
                cooccurrence[ri][hi] += dt;
            }
        }
    }

    // find optimal 1-to-1 mapping (ref_idx → hyp_idx)
    let mapping = optimal_mapping(&cooccurrence, ref_speakers.len(), hyp_speakers.len());

    // compute DER components using the optimal mapping
    let mut total = 0.0;
    let mut missed = 0.0;
    let mut false_alarm = 0.0;
    let mut confusion = 0.0;

    for window in boundaries.windows(2) {
        let (t_start, t_end) = (window[0], window[1]);
        let dt = t_end - t_start;
        if dt <= 0.0 {
            continue;
        }

        let active_ref = active_speakers_at(reference, t_start, t_end, &ref_speakers);
        let active_hyp = active_speakers_at(hypothesis, t_start, t_end, &hyp_speakers);

        let n_ref = active_ref.len();
        let n_hyp = active_hyp.len();

        total += n_ref as f64 * dt;

        // count correctly mapped speakers
        let n_correct = active_ref
            .iter()
            .filter(|&&ri| mapping.get(&ri).is_some_and(|&hi| active_hyp.contains(&hi)))
            .count();

        missed += (n_ref.saturating_sub(n_hyp)) as f64 * dt;
        false_alarm += (n_hyp.saturating_sub(n_ref)) as f64 * dt;
        confusion += (n_ref.min(n_hyp) - n_correct) as f64 * dt;
    }

    DerResult {
        missed,
        false_alarm,
        confusion,
        total,
    }
}

fn unique_speakers(segments: &[Segment]) -> Vec<String> {
    let mut seen = Vec::new();
    for seg in segments {
        if !seen.contains(&seg.speaker) {
            seen.push(seg.speaker.clone());
        }
    }
    seen
}

/// Find speaker indices active during [t_start, t_end)
fn active_speakers_at(
    segments: &[Segment],
    t_start: f64,
    t_end: f64,
    speaker_list: &[String],
) -> Vec<usize> {
    let mid = (t_start + t_end) / 2.0;
    let mut active = Vec::new();
    for seg in segments {
        if seg.start <= mid
            && mid < seg.end
            && let Some(idx) = speaker_list.iter().position(|s| s == &seg.speaker)
            && !active.contains(&idx)
        {
            active.push(idx);
        }
    }
    active
}

/// Find optimal 1-to-1 mapping from ref speakers to hyp speakers
/// That maximizes total co-occurrence using the Hungarian algorithm (O(n³))
fn optimal_mapping(cooccurrence: &[Vec<f64>], n_ref: usize, n_hyp: usize) -> HashMap<usize, usize> {
    if n_ref == 0 || n_hyp == 0 {
        return HashMap::new();
    }

    // pad to square and negate for minimization
    let n = n_ref.max(n_hyp);
    let mut cost = vec![vec![0.0f64; n]; n];
    for ri in 0..n_ref {
        for hi in 0..n_hyp {
            cost[ri][hi] = -cooccurrence[ri][hi];
        }
    }

    let assignment = hungarian_algorithm(&cost);

    assignment
        .into_iter()
        .enumerate()
        .filter(|&(ri, hi)| ri < n_ref && hi < n_hyp)
        .collect()
}

struct HungarianSolver {
    dimension: usize,
    row_potentials: Vec<f64>,
    col_potentials: Vec<f64>,
    col_to_row: Vec<usize>,
    prev_col: Vec<usize>,
}

impl HungarianSolver {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            row_potentials: vec![0.0; dimension + 1],
            col_potentials: vec![0.0; dimension + 1],
            col_to_row: vec![0; dimension + 1],
            prev_col: vec![0; dimension + 1],
        }
    }

    fn assign_row(&mut self, row: usize, cost: &[Vec<f64>]) {
        let n = self.dimension;
        self.col_to_row[0] = row;
        let mut current_col = 0usize;
        let mut shortest_paths = vec![f64::INFINITY; n + 1];
        let mut visited = vec![false; n + 1];

        loop {
            visited[current_col] = true;
            let assigned_row = self.col_to_row[current_col];
            let mut min_delta = f64::INFINITY;
            let mut next_col = 0usize;

            for j in 1..=n {
                if visited[j] {
                    continue;
                }
                let reduced_cost = cost[assigned_row - 1][j - 1]
                    - self.row_potentials[assigned_row]
                    - self.col_potentials[j];
                if reduced_cost < shortest_paths[j] {
                    shortest_paths[j] = reduced_cost;
                    self.prev_col[j] = current_col;
                }
                if shortest_paths[j] < min_delta {
                    min_delta = shortest_paths[j];
                    next_col = j;
                }
            }

            for j in 0..=n {
                if visited[j] {
                    self.row_potentials[self.col_to_row[j]] += min_delta;
                    self.col_potentials[j] -= min_delta;
                } else {
                    shortest_paths[j] -= min_delta;
                }
            }

            current_col = next_col;
            if self.col_to_row[current_col] == 0 {
                break;
            }
        }

        // augmenting path
        loop {
            let prev = self.prev_col[current_col];
            self.col_to_row[current_col] = self.col_to_row[prev];
            current_col = prev;
            if current_col == 0 {
                break;
            }
        }
    }

    fn into_assignment(self) -> Vec<usize> {
        let mut result = vec![0usize; self.dimension];
        for j in 1..=self.dimension {
            result[self.col_to_row[j] - 1] = j - 1;
        }
        result
    }
}

/// Hungarian algorithm for the linear assignment problem (O(n³))
///
/// Returns a Vec where result[row] = assigned column
fn hungarian_algorithm(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    let mut solver = HungarianSolver::new(n);
    for row in 1..=n {
        solver.assign_row(row, cost);
    }
    solver.into_assignment()
}

/// Wrapper for f64 that implements Ord for use in BTreeSet
#[derive(Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::to_rttm;

    #[test]
    fn perfect_match_zero_der() {
        let segments = vec![Segment::new(0.0, 5.0, "A"), Segment::new(5.0, 10.0, "B")];
        let result = compute_der(&segments, &segments);

        assert_eq!(result.der(), 0.0);
        assert_eq!(result.missed, 0.0);
        assert_eq!(result.false_alarm, 0.0);
        assert_eq!(result.confusion, 0.0);
        assert_eq!(result.total, 10.0);
    }

    #[test]
    fn swapped_speakers_zero_confusion() {
        // optimal mapping should resolve swapped labels
        let reference = vec![Segment::new(0.0, 5.0, "A"), Segment::new(5.0, 10.0, "B")];
        let hypothesis = vec![Segment::new(0.0, 5.0, "X"), Segment::new(5.0, 10.0, "Y")];
        let result = compute_der(&reference, &hypothesis);

        assert_eq!(result.der(), 0.0);
        assert_eq!(result.confusion, 0.0);
    }

    #[test]
    fn complete_miss() {
        let reference = vec![Segment::new(0.0, 10.0, "A")];
        let hypothesis = vec![];
        let result = compute_der(&reference, &hypothesis);

        assert!((result.der() - 1.0).abs() < 1e-9);
        assert!((result.missed - 10.0).abs() < 1e-9);
        assert_eq!(result.false_alarm, 0.0);
        assert_eq!(result.confusion, 0.0);
    }

    #[test]
    fn complete_false_alarm() {
        let reference = vec![Segment::new(0.0, 5.0, "A")];
        let hypothesis = vec![Segment::new(0.0, 5.0, "X"), Segment::new(5.0, 10.0, "Y")];
        let result = compute_der(&reference, &hypothesis);

        // 5s of correct speech + 5s false alarm / 5s total = 1.0
        assert!((result.false_alarm - 5.0).abs() < 1e-9);
        assert_eq!(result.missed, 0.0);
        assert_eq!(result.confusion, 0.0);
        assert!((result.der() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn partial_overlap_with_confusion() {
        // ref: A speaks 0-10
        // hyp: X speaks 0-5, Y speaks 5-10
        // optimal: map A→X or A→Y, either way 5s is confused
        let reference = vec![Segment::new(0.0, 10.0, "A")];
        let hypothesis = vec![Segment::new(0.0, 5.0, "X"), Segment::new(5.0, 10.0, "Y")];
        let result = compute_der(&reference, &hypothesis);

        // n_ref=1, n_hyp=1 everywhere, so no miss/fa
        // mapping picks whichever has more overlap (both 5s, picks first)
        // 5s correct, 5s confusion
        assert!((result.confusion - 5.0).abs() < 1e-9);
        assert_eq!(result.missed, 0.0);
        assert_eq!(result.false_alarm, 0.0);
        assert!((result.total - 10.0).abs() < 1e-9);
    }

    #[test]
    fn overlapping_speech() {
        // ref: A speaks 0-10, B speaks 5-10 (overlap at 5-10)
        // hyp: same
        let reference = vec![Segment::new(0.0, 10.0, "A"), Segment::new(5.0, 10.0, "B")];
        let result = compute_der(&reference, &reference);

        assert_eq!(result.der(), 0.0);
        // total = 5s (1 speaker) + 5s (2 speakers) = 15s
        assert!((result.total - 15.0).abs() < 1e-9);
    }

    #[test]
    fn parse_rttm_round_trip() {
        let segments = vec![
            Segment::new(1.5, 3.0, "SPEAKER_00"),
            Segment::new(4.0, 6.5, "SPEAKER_01"),
        ];
        let rttm = to_rttm(&segments, "test");
        let parsed = parse_rttm(&rttm);

        assert_eq!(parsed.len(), 2);
        assert!((parsed[0].start - 1.5).abs() < 1e-5);
        assert!((parsed[0].end - 3.0).abs() < 1e-5);
        assert_eq!(parsed[0].speaker, "SPEAKER_00");
        assert!((parsed[1].start - 4.0).abs() < 1e-5);
        assert!((parsed[1].end - 6.5).abs() < 1e-5);
        assert_eq!(parsed[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn empty_inputs() {
        let result = compute_der(&[], &[]);
        assert_eq!(result.der(), 0.0);
        assert_eq!(result.total, 0.0);

        let result = compute_der(&[], &[Segment::new(0.0, 5.0, "A")]);
        assert!((result.false_alarm - 5.0).abs() < 1e-9);
    }

    #[test]
    fn three_speakers_with_mapping() {
        let reference = vec![
            Segment::new(0.0, 5.0, "A"),
            Segment::new(5.0, 10.0, "B"),
            Segment::new(10.0, 15.0, "C"),
        ];
        // hypothesis uses different labels in different order
        let hypothesis = vec![
            Segment::new(0.0, 5.0, "Z"),
            Segment::new(5.0, 10.0, "X"),
            Segment::new(10.0, 15.0, "Y"),
        ];
        let result = compute_der(&reference, &hypothesis);

        assert_eq!(result.der(), 0.0);
        assert_eq!(result.total, 15.0);
    }
}
