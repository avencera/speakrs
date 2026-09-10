use ndarray::Array2;

/// Frame-count activity cleanup applied after discrete reconstruction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ActivityCleanup {
    min_active_frames: usize,
    max_inactive_gap_frames: usize,
    pad_before_frames: usize,
    pad_after_frames: usize,
}

impl ActivityCleanup {
    /// Create cleanup values from frame counts
    pub const fn new(
        min_active_frames: usize,
        max_inactive_gap_frames: usize,
        pad_before_frames: usize,
        pad_after_frames: usize,
    ) -> Self {
        Self {
            min_active_frames,
            max_inactive_gap_frames,
            pad_before_frames,
            pad_after_frames,
        }
    }

    /// Minimum active run length kept after cleanup
    pub const fn min_active_frames(self) -> usize {
        self.min_active_frames
    }

    /// Maximum inactive interior gap filled between active runs
    pub const fn max_inactive_gap_frames(self) -> usize {
        self.max_inactive_gap_frames
    }

    /// Frames added before each remaining active run
    pub const fn pad_before_frames(self) -> usize {
        self.pad_before_frames
    }

    /// Frames added after each remaining active run
    pub const fn pad_after_frames(self) -> usize {
        self.pad_after_frames
    }

    /// True when cleanup leaves binary activity unchanged
    pub const fn is_identity(self) -> bool {
        self.min_active_frames == 0
            && self.max_inactive_gap_frames == 0
            && self.pad_before_frames == 0
            && self.pad_after_frames == 0
    }

    /// Apply remove-short, fill-gap, then pad order to binary activity
    pub fn apply(&self, discrete: &Array2<f32>) -> Array2<f32> {
        if self.is_identity() {
            return discrete.clone();
        }

        let (num_frames, num_speakers) = discrete.dim();
        let mut output = Array2::<f32>::zeros((num_frames, num_speakers));

        for speaker in 0..num_speakers {
            let mut active: Vec<bool> = (0..num_frames)
                .map(|frame| discrete[[frame, speaker]] > 0.5)
                .collect();

            remove_short_on(&mut active, self.min_active_frames);
            fill_short_off(&mut active, self.max_inactive_gap_frames);
            pad_regions(&mut active, self.pad_before_frames, self.pad_after_frames);

            for (frame, &value) in active.iter().enumerate() {
                output[[frame, speaker]] = if value { 1.0 } else { 0.0 };
            }
        }

        output
    }
}

fn remove_short_on(active: &mut [bool], min_duration: usize) {
    if min_duration == 0 {
        return;
    }

    let runs = find_runs(active, true);
    for (start, end) in runs {
        if end - start < min_duration {
            active[start..end].fill(false);
        }
    }
}

fn fill_short_off(active: &mut [bool], min_duration: usize) {
    if min_duration == 0 {
        return;
    }

    let runs = find_runs(active, false);
    for (start, end) in runs {
        // only fill interior gaps (between ON regions)
        if start > 0 && end < active.len() && end - start < min_duration {
            active[start..end].fill(true);
        }
    }
}

fn pad_regions(active: &mut [bool], pad_onset: usize, pad_offset: usize) {
    if pad_onset == 0 && pad_offset == 0 {
        return;
    }

    let runs = find_runs(active, true);
    for (start, end) in runs {
        let pad_start = start.saturating_sub(pad_onset);
        let pad_end = (end + pad_offset).min(active.len());
        active[pad_start..pad_end].fill(true);
    }
}

/// Find contiguous runs of the target value, returns (start, end) pairs where end is exclusive
fn find_runs(active: &[bool], target: bool) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;

    while i < active.len() {
        if active[i] == target {
            let start = i;
            while i < active.len() && active[i] == target {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }

    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn min_duration_on_removal() {
        let probs = array![[0.0], [0.8], [0.0], [0.8], [0.8], [0.8], [0.0]];
        let config = ActivityCleanup::new(3, 0, 0, 0);
        let result = config.apply(&probs);
        let expected = array![[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn min_duration_off_fill() {
        let probs = array![[0.8], [0.8], [0.0], [0.8], [0.8]];
        let config = ActivityCleanup::new(0, 2, 0, 0);
        let result = config.apply(&probs);
        let expected = array![[1.0], [1.0], [1.0], [1.0], [1.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn pad_onset_offset() {
        let probs = array![[0.0], [0.0], [0.0], [0.8], [0.8], [0.0], [0.0], [0.0]];
        let config = ActivityCleanup::new(0, 0, 2, 1);
        let result = config.apply(&probs);
        let expected = array![[0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn multi_speaker_independence() {
        let probs = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        let result = ActivityCleanup::default().apply(&probs);
        let expected = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn default_cleanup_is_identity() {
        let config = ActivityCleanup::default();
        assert!(config.is_identity());
        let probs = array![[1.0], [0.0], [1.0]];
        assert_eq!(config.apply(&probs), probs);
    }
}
