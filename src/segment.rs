use ndarray::Array2;

/// A single speaker turn with start/end times in seconds
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Speaker label (e.g. "SPEAKER_00")
    pub speaker: String,
}

impl Segment {
    /// Create a new segment
    pub fn new(start: f64, end: f64, speaker: impl Into<String>) -> Self {
        Self {
            start,
            end,
            speaker: speaker.into(),
        }
    }

    /// Duration in seconds
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

impl std::fmt::Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "SPEAKER file 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>",
            self.start,
            self.duration(),
            self.speaker
        )
    }
}

/// Convert binary activation matrix to speaker segments
pub fn to_segments(
    activations: &Array2<f32>,
    frame_step: f64,
    frame_duration: f64,
) -> Vec<Segment> {
    let (_num_frames, num_speakers) = activations.dim();
    let mut segments = Vec::new();

    for speaker_idx in 0..num_speakers {
        let label = format!("SPEAKER_{speaker_idx:02}");
        let column = activations.column(speaker_idx);

        if column.is_empty() {
            continue;
        }

        let mut start = frame_middle(0, frame_step, frame_duration);
        let mut is_active = column[0] > 0.5;
        let mut last_timestamp = start;

        for (frame_idx, &value) in column.iter().enumerate().skip(1) {
            let timestamp = frame_middle(frame_idx, frame_step, frame_duration);
            last_timestamp = timestamp;

            if is_active {
                if value < 0.5 {
                    segments.push(Segment::new(start, timestamp, &label));
                    start = timestamp;
                    is_active = false;
                }
            } else if value > 0.5 {
                start = timestamp;
                is_active = true;
            }
        }

        if is_active {
            segments.push(Segment::new(start, last_timestamp, &label));
        }
    }

    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    segments
}

fn frame_middle(frame_idx: usize, frame_step: f64, frame_duration: f64) -> f64 {
    frame_idx as f64 * frame_step + 0.5 * frame_duration
}

/// Merge consecutive same-speaker segments with gap smaller than max_gap
pub fn merge_segments(segments: &[Segment], max_gap: f64) -> Vec<Segment> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<Segment> = vec![segments[0].clone()];

    for seg in &segments[1..] {
        if let Some(last) = merged.last_mut()
            && seg.speaker == last.speaker
            && (seg.start - last.end) < max_gap
        {
            last.end = seg.end;
            continue;
        }

        merged.push(seg.clone());
    }

    merged
}

/// Format segments as RTTM output
pub fn to_rttm(segments: &[Segment], file_id: &str) -> String {
    segments
        .iter()
        .map(|s| {
            format!(
                "SPEAKER {file_id} 1 {:.6} {:.6} <NA> <NA> {} <NA> <NA>\n",
                s.start,
                s.duration(),
                s.speaker
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn single_segment_timing() {
        let activations = array![[0.0], [1.0], [1.0], [1.0], [0.0]];
        let segments = to_segments(&activations, 0.1, 0.2);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert!((segments[0].start - 0.2).abs() < 1e-9);
        assert!((segments[0].end - 0.5).abs() < 1e-9);
        assert!((segments[0].duration() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn multi_speaker_sorted_by_start() {
        let activations = array![[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0],];
        let segments = to_segments(&activations, 0.1, 0.2);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_01");
        assert!((segments[0].start - 0.1).abs() < 1e-9);
        assert_eq!(segments[1].speaker, "SPEAKER_00");
        assert!((segments[1].start - 0.3).abs() < 1e-9);
    }

    #[test]
    fn merge_close_segments() {
        let segments = vec![
            Segment::new(0.0, 1.0, "SPEAKER_00"),
            Segment::new(1.05, 2.0, "SPEAKER_00"),
        ];
        let merged = merge_segments(&segments, 0.1);

        assert_eq!(merged.len(), 1);
        assert!((merged[0].end - 2.0).abs() < 1e-9);
    }

    #[test]
    fn no_merge_far_segments() {
        let segments = vec![
            Segment::new(0.0, 1.0, "SPEAKER_00"),
            Segment::new(2.0, 3.0, "SPEAKER_00"),
        ];
        let merged = merge_segments(&segments, 0.1);

        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn rttm_format() {
        let segments = vec![Segment::new(1.5, 3.0, "SPEAKER_00")];
        let rttm = to_rttm(&segments, "meeting");

        assert_eq!(
            rttm,
            "SPEAKER meeting 1 1.500000 1.500000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        );
    }

    #[test]
    fn empty_input() {
        let activations = Array2::<f32>::zeros((0, 0));
        let segments = to_segments(&activations, 0.1, 0.2);
        assert!(segments.is_empty());

        let merged = merge_segments(&[], 0.1);
        assert!(merged.is_empty());

        let rttm = to_rttm(&[], "file");
        assert!(rttm.is_empty());
    }

    #[test]
    fn all_zeros_no_segments() {
        let activations = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let segments = to_segments(&activations, 0.1, 0.2);
        assert!(segments.is_empty());
    }

    #[test]
    fn display_trait_rttm_line() {
        let seg = Segment::new(1.0, 2.5, "SPEAKER_01");
        let display = format!("{seg}");
        assert_eq!(
            display,
            "SPEAKER file 1 1.000 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>"
        );
    }
}
