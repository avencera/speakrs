use ndarray::Array2;

pub struct PowersetMapping {
    mapping: Array2<f32>,
}

impl PowersetMapping {
    pub fn new(num_speakers: usize, max_set_size: usize) -> Self {
        let mut rows: Vec<Vec<f32>> = Vec::new();

        for size in 0..=max_set_size {
            for combo in combinations(num_speakers, size) {
                let mut row = vec![0.0f32; num_speakers];
                for speaker in combo {
                    row[speaker] = 1.0;
                }
                rows.push(row);
            }
        }

        let num_classes = rows.len();
        let mut mapping = Array2::zeros((num_classes, num_speakers));
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                mapping[[i, j]] = val;
            }
        }

        Self { mapping }
    }

    pub fn num_powerset_classes(&self) -> usize {
        self.mapping.nrows()
    }

    /// Hard decode powerset logits to binary speaker activations
    pub fn hard_decode(&self, logits: &Array2<f32>) -> Array2<f32> {
        let num_frames = logits.nrows();
        let num_classes = self.num_powerset_classes();

        let mut one_hot = Array2::zeros((num_frames, num_classes));
        for i in 0..num_frames {
            let row = logits.row(i);
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            one_hot[[i, argmax]] = 1.0;
        }

        one_hot.dot(&self.mapping)
    }

    /// Soft decode powerset log-probabilities to speaker activations
    pub fn soft_decode(&self, log_probs: &Array2<f32>) -> Array2<f32> {
        let probs = log_probs.mapv(|x| x.exp());
        probs.dot(&self.mapping)
    }

    /// Encode binary multilabel matrix to powerset one-hot
    pub fn encode(&self, multilabel: &Array2<f32>) -> Array2<f32> {
        let num_frames = multilabel.nrows();
        let num_classes = self.num_powerset_classes();
        let mut output = Array2::zeros((num_frames, num_classes));

        for i in 0..num_frames {
            let frame = multilabel.row(i);
            for c in 0..num_classes {
                let mapping_row = self.mapping.row(c);
                if frame == mapping_row {
                    output[[i, c]] = 1.0;
                    break;
                }
            }
        }

        output
    }
}

/// Generate all combinations of `k` items from `0..n` in lexicographic order
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut combo: Vec<usize> = (0..k).collect();

    loop {
        result.push(combo.clone());

        // find rightmost element that can be incremented
        let mut i = k;
        while i > 0 {
            i -= 1;
            if combo[i] != i + n - k {
                break;
            }
            if i == 0 && combo[0] == n - k {
                return result;
            }
        }

        combo[i] += 1;
        for j in (i + 1)..k {
            combo[j] = combo[j - 1] + 1;
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
        assert_eq!(pm.mapping, expected);
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

        // logits with highest value at class 0 (empty set) → all zeros
        let logits = array![[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let result = pm.hard_decode(&logits);
        assert_eq!(result, array![[0.0, 0.0, 0.0]]);
    }

    #[test]
    fn hard_decode_single_speaker() {
        let pm = PowersetMapping::new(3, 2);

        // logits with highest value at class 2 (S1)
        let logits = array![[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]];
        let result = pm.hard_decode(&logits);
        assert_eq!(result, array![[0.0, 1.0, 0.0]]);
    }

    #[test]
    fn hard_decode_overlap() {
        let pm = PowersetMapping::new(3, 2);

        // logits with highest value at class 4 (S0+S1)
        let logits = array![[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]];
        let result = pm.hard_decode(&logits);
        assert_eq!(result, array![[1.0, 1.0, 0.0]]);
    }

    #[test]
    fn soft_decode_uniform() {
        let pm = PowersetMapping::new(3, 2);

        // uniform log-probs of 0 → exp(0) = 1 for all classes
        // each speaker appears in 3 of 7 classes, so activation = 3.0
        let logits = array![[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let result = pm.soft_decode(&logits);

        for &val in result.iter() {
            assert!((val - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn roundtrip_encode_hard_decode() {
        for nc in 2..5 {
            for ms in 1..=nc {
                let pm = PowersetMapping::new(nc, ms);
                let num_classes = pm.num_powerset_classes();

                // identity matrix as one-hot powerset input
                let identity = Array2::eye(num_classes);

                let decoded = pm.hard_decode(&identity);
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

            assert_eq!(
                pm.mapping.shape(),
                expected.shape(),
                "shape mismatch for nc={nc}, ms={ms}"
            );
            for (a, b) in pm.mapping.iter().zip(expected.iter()) {
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
        let result = pm.hard_decode(&logits);

        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result, expected);
    }

    #[test]
    fn soft_decode_matches_fixture() {
        let logits_3d: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("powerset_input_logits.npy")).unwrap())
                .unwrap();
        let expected_3d: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("powerset_soft_output.npy")).unwrap())
                .unwrap();

        let logits = logits_3d.index_axis(ndarray::Axis(0), 0).to_owned();
        let expected = expected_3d.index_axis(ndarray::Axis(0), 0).to_owned();

        let pm = PowersetMapping::new(3, 2);
        let result = pm.soft_decode(&logits);

        assert_eq!(result.shape(), expected.shape());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "soft_decode mismatch: {a} vs {b}");
        }
    }
}
