use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct BinarizeConfig {
    pub onset: f32,
    pub offset: f32,
    pub min_duration_on: usize,
    pub min_duration_off: usize,
    pub pad_onset: usize,
    pub pad_offset: usize,
}

impl Default for BinarizeConfig {
    fn default() -> Self {
        Self {
            onset: 0.5,
            offset: 0.5,
            min_duration_on: 0,
            min_duration_off: 0,
            pad_onset: 0,
            pad_offset: 0,
        }
    }
}

pub fn binarize(probs: &Array2<f32>, config: &BinarizeConfig) -> Array2<f32> {
    let (num_frames, num_speakers) = probs.dim();
    let mut output = Array2::<f32>::zeros((num_frames, num_speakers));

    for speaker in 0..num_speakers {
        let scores: Vec<f32> = (0..num_frames).map(|f| probs[[f, speaker]]).collect();
        let mut active = hysteresis(&scores, config.onset, config.offset);

        remove_short_on(&mut active, config.min_duration_on);
        fill_short_off(&mut active, config.min_duration_off);
        pad_regions(&mut active, config.pad_onset, config.pad_offset);

        for (f, &val) in active.iter().enumerate() {
            output[[f, speaker]] = if val { 1.0 } else { 0.0 };
        }
    }

    output
}

fn hysteresis(scores: &[f32], onset: f32, offset: f32) -> Vec<bool> {
    let mut state = false;
    scores
        .iter()
        .map(|&s| {
            if !state && s >= onset {
                state = true;
            } else if state && s < offset {
                state = false;
            }
            state
        })
        .collect()
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
    fn chattering_prevention() {
        let probs = array![[0.45], [0.55], [0.45], [0.55], [0.45]];
        let config = BinarizeConfig {
            onset: 0.6,
            offset: 0.4,
            ..Default::default()
        };

        let result = binarize(&probs, &config);
        let expected = array![[0.0], [0.0], [0.0], [0.0], [0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn state_holding() {
        let probs = array![[0.0], [0.7], [0.5], [0.5], [0.3], [0.0]];
        let config = BinarizeConfig {
            onset: 0.6,
            offset: 0.4,
            ..Default::default()
        };

        let result = binarize(&probs, &config);
        let expected = array![[0.0], [1.0], [1.0], [1.0], [0.0], [0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn min_duration_on_removal() {
        // short ON blip of 1 frame gets removed with min_duration_on=3
        let probs = array![[0.0], [0.8], [0.0], [0.8], [0.8], [0.8], [0.0]];
        let config = BinarizeConfig {
            onset: 0.5,
            offset: 0.5,
            min_duration_on: 3,
            ..Default::default()
        };

        let result = binarize(&probs, &config);
        let expected = array![[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn min_duration_off_fill() {
        // short OFF gap of 1 frame gets filled with min_duration_off=2
        let probs = array![[0.8], [0.8], [0.0], [0.8], [0.8]];
        let config = BinarizeConfig {
            onset: 0.5,
            offset: 0.5,
            min_duration_off: 2,
            ..Default::default()
        };

        let result = binarize(&probs, &config);
        let expected = array![[1.0], [1.0], [1.0], [1.0], [1.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn pad_onset_offset() {
        let probs = array![[0.0], [0.0], [0.0], [0.8], [0.8], [0.0], [0.0], [0.0]];
        let config = BinarizeConfig {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 2,
            pad_offset: 1,
            ..Default::default()
        };

        let result = binarize(&probs, &config);
        // ON at frames 3,4 → padded to frames 1..6
        let expected = array![[0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn multi_speaker_independence() {
        let probs = array![[0.8, 0.0], [0.8, 0.0], [0.0, 0.8], [0.0, 0.8]];
        let config = BinarizeConfig::default();

        let result = binarize(&probs, &config);
        let expected = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn all_on() {
        let probs = array![[0.9], [0.8], [0.7]];
        let config = BinarizeConfig::default();

        let result = binarize(&probs, &config);
        let expected = array![[1.0], [1.0], [1.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn all_off() {
        let probs = array![[0.1], [0.2], [0.3]];
        let config = BinarizeConfig::default();

        let result = binarize(&probs, &config);
        let expected = array![[0.0], [0.0], [0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn default_config_works() {
        let config = BinarizeConfig::default();
        assert_eq!(config.onset, 0.5);
        assert_eq!(config.offset, 0.5);
        assert_eq!(config.min_duration_on, 0);
        assert_eq!(config.min_duration_off, 0);
        assert_eq!(config.pad_onset, 0);
        assert_eq!(config.pad_offset, 0);
    }
}
