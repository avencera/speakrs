use ndarray::Array2;

/// Maps between powerset class indices and multi-speaker binary activations
pub struct PowersetMapping {
    classes: Vec<Vec<usize>>,
    num_speakers: usize,
}

/// Invalid powerset logits at the decode boundary
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PowersetDecodeError {
    /// Logits had no class columns
    #[error("powerset logits have zero class columns")]
    ZeroClassColumns,
    /// Logits had fewer class columns than the mapping
    #[error("powerset logits have {actual} class columns, expected {expected}")]
    TooFewClassColumns {
        /// Mapping class count
        expected: usize,
        /// Observed class count
        actual: usize,
    },
    /// Logits had more class columns than the mapping
    #[error("powerset logits have {actual} class columns, expected {expected}")]
    TooManyClassColumns {
        /// Mapping class count
        expected: usize,
        /// Observed class count
        actual: usize,
    },
}

impl PowersetMapping {
    /// Build the powerset mapping for a given number of speakers and max simultaneous speakers
    pub fn new(num_speakers: usize, max_set_size: usize) -> Self {
        let mut classes = Vec::new();
        for size in 0..=max_set_size {
            for combo in combinations(num_speakers, size) {
                classes.push(combo);
            }
        }
        Self {
            classes,
            num_speakers,
        }
    }

    /// Number of powerset classes (e.g. 7 for 3 speakers with max overlap 2)
    pub fn num_powerset_classes(&self) -> usize {
        self.classes.len()
    }

    #[cfg(test)]
    fn class_row(&self, class: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; self.num_speakers];
        for &speaker in &self.classes[class] {
            row[speaker] = 1.0;
        }
        row
    }

    /// Hard decode powerset logits to binary speaker activations
    pub fn hard_decode(&self, logits: &Array2<f32>) -> Result<Array2<f32>, PowersetDecodeError> {
        let num_frames = logits.nrows();
        let expected = self.num_powerset_classes();
        let actual = logits.ncols();
        if actual == 0 {
            return Err(PowersetDecodeError::ZeroClassColumns);
        }
        if actual < expected {
            return Err(PowersetDecodeError::TooFewClassColumns { expected, actual });
        }
        if actual > expected {
            return Err(PowersetDecodeError::TooManyClassColumns { expected, actual });
        }

        let mut output = Array2::zeros((num_frames, self.num_speakers));
        for frame in 0..num_frames {
            let row = logits.row(frame);
            let class = row
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            for &speaker in &self.classes[class] {
                output[[frame, speaker]] = 1.0;
            }
        }
        Ok(output)
    }
}

/// Generate all combinations of `size` items from `0..total` in lexicographic order
fn combinations(total: usize, size: usize) -> Vec<Vec<usize>> {
    if size == 0 {
        return vec![vec![]];
    }
    if size > total {
        return vec![];
    }

    let mut result = Vec::new();
    let mut combination: Vec<usize> = (0..size).collect();

    loop {
        result.push(combination.clone());

        // find rightmost element that can be incremented
        let mut pos = size;
        while pos > 0 {
            pos -= 1;
            if combination[pos] != pos + total - size {
                break;
            }
            if pos == 0 && combination[0] == total - size {
                return result;
            }
        }

        combination[pos] += 1;
        for fill_pos in (pos + 1)..size {
            combination[fill_pos] = combination[fill_pos - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use ndarray::array;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::PathBuf;

    impl PowersetMapping {
        fn encode(&self, multilabel: &Array2<f32>) -> Array2<f32> {
            let num_frames = multilabel.nrows();
            let num_classes = self.num_powerset_classes();
            let mut output = Array2::zeros((num_frames, num_classes));

            for i in 0..num_frames {
                let frame = multilabel.row(i);
                for c in 0..num_classes {
                    let mapping_row = ndarray::Array1::from(self.class_row(c));
                    if frame == mapping_row.view() {
                        output[[i, c]] = 1.0;
                        break;
                    }
                }
            }

            output
        }

        fn mapping_matrix(&self) -> Array2<f32> {
            let mut mapping = Array2::zeros((self.num_powerset_classes(), self.num_speakers));
            for (class, _) in self.classes.iter().enumerate() {
                for (speaker, value) in self.class_row(class).into_iter().enumerate() {
                    mapping[[class, speaker]] = value;
                }
            }
            mapping
        }
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn mapping_matrix_3_2() {
        let pm = PowersetMapping::new(3, 2);
        assert_eq!(pm.num_powerset_classes(), 7);

        let expected = array![
            [0.0, 0.0, 0.0], // empty set
            [1.0, 0.0, 0.0], // S0
            [0.0, 1.0, 0.0], // S1
            [0.0, 0.0, 1.0], // S2
            [1.0, 1.0, 0.0], // S0+S1
            [1.0, 0.0, 1.0], // S0+S2
            [0.0, 1.0, 1.0], // S1+S2
        ];
        assert_eq!(pm.mapping_matrix(), expected);
    }

    #[test]
    fn num_powerset_classes_count() {
        assert_eq!(PowersetMapping::new(3, 2).num_powerset_classes(), 7);
        assert_eq!(PowersetMapping::new(4, 1).num_powerset_classes(), 5);
        assert_eq!(PowersetMapping::new(2, 2).num_powerset_classes(), 4);
        assert_eq!(PowersetMapping::new(4, 2).num_powerset_classes(), 11);
    }

    #[test]
    fn hard_decode_silence() {
        let pm = PowersetMapping::new(3, 2);

        // logits with the highest value at class 0 (empty set) give all zeros
        let logits = array![[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let result = pm.hard_decode(&logits).unwrap();
        assert_eq!(result, array![[0.0, 0.0, 0.0]]);
    }

    #[test]
    fn hard_decode_single_speaker() {
        let pm = PowersetMapping::new(3, 2);

        // logits with highest value at class 2 (S1)
        let logits = array![[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]];
        let result = pm.hard_decode(&logits).unwrap();
        assert_eq!(result, array![[0.0, 1.0, 0.0]]);
    }

    #[test]
    fn hard_decode_overlap() {
        let pm = PowersetMapping::new(3, 2);

        // logits with highest value at class 4 (S0+S1)
        let logits = array![[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]];
        let result = pm.hard_decode(&logits).unwrap();
        assert_eq!(result, array![[1.0, 1.0, 0.0]]);
    }

    #[test]
    fn roundtrip_encode_hard_decode() {
        for nc in 2..5 {
            for ms in 1..=nc {
                let pm = PowersetMapping::new(nc, ms);
                let num_classes = pm.num_powerset_classes();

                // identity matrix as one-hot powerset input
                let identity = Array2::eye(num_classes);

                let decoded = pm.hard_decode(&identity).unwrap();
                let re_encoded = pm.encode(&decoded);

                assert_eq!(
                    identity, re_encoded,
                    "roundtrip failed for num_speakers={nc}, max_set_size={ms}"
                );
            }
        }
    }

    #[test]
    fn mapping_matrices_match_fixtures() {
        let cases = [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        ];

        for (nc, ms) in cases {
            let pm = PowersetMapping::new(nc, ms);
            let filename = format!("powerset_mapping_{nc}_{ms}.npy");
            let expected: Array2<f32> =
                Array2::read_npy(File::open(fixture_path(&filename)).unwrap()).unwrap();

            let mapping = pm.mapping_matrix();
            assert_eq!(
                mapping.shape(),
                expected.shape(),
                "shape mismatch for nc={nc}, ms={ms}"
            );
            for (a, b) in mapping.iter().zip(expected.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "value mismatch for nc={nc}, ms={ms}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn hard_decode_matches_fixture() {
        let logits_3d: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("powerset_input_logits.npy")).unwrap())
                .unwrap();
        let expected_3d: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("powerset_hard_output.npy")).unwrap())
                .unwrap();

        // squeeze batch dimension
        let logits = logits_3d.index_axis(ndarray::Axis(0), 0).to_owned();
        let expected = expected_3d.index_axis(ndarray::Axis(0), 0).to_owned();

        let pm = PowersetMapping::new(3, 2);
        let result = pm.hard_decode(&logits).unwrap();

        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result, expected);
    }

    #[test]
    fn hard_decode_rejects_class_column_mismatch() {
        let pm = PowersetMapping::new(3, 2);
        let empty = Array2::<f32>::zeros((1, 0));
        assert!(matches!(
            pm.hard_decode(&empty),
            Err(PowersetDecodeError::ZeroClassColumns)
        ));
        let short = Array2::<f32>::zeros((1, 3));
        assert!(matches!(
            pm.hard_decode(&short),
            Err(PowersetDecodeError::TooFewClassColumns {
                expected: 7,
                actual: 3
            })
        ));
        let long = Array2::<f32>::zeros((1, 8));
        assert!(matches!(
            pm.hard_decode(&long),
            Err(PowersetDecodeError::TooManyClassColumns {
                expected: 7,
                actual: 8
            })
        ));
    }
}
