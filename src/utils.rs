use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub fn l2_normalize(v: &ArrayView1<f32>) -> Array1<f32> {
    let norm = v.dot(v).sqrt();
    if norm == 0.0 {
        return Array1::zeros(v.len());
    }
    v / norm
}

pub fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let a_norm = l2_normalize(a);
    let b_norm = l2_normalize(b);
    a_norm.dot(&b_norm)
}

pub fn cosine_distance_matrix(embeddings: &ArrayView2<f32>) -> Array2<f32> {
    let n = embeddings.nrows();
    let mut normed = Array2::zeros(embeddings.raw_dim());
    for i in 0..n {
        normed.row_mut(i).assign(&l2_normalize(&embeddings.row(i)));
    }

    let similarity = normed.dot(&normed.t());
    let ones = Array2::from_elem((n, n), 1.0f32);
    ones - similarity
}

pub fn logsumexp(a: &ArrayView1<f32>) -> f32 {
    let max = a.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    if max.is_infinite() {
        return max;
    }

    let sum_exp = a.mapv(|x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

pub fn centroid(embeddings: &ArrayView2<f32>) -> Array1<f32> {
    embeddings.mean_axis(Axis(0)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = array![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v.view(), &v.view());
        assert_abs_diff_eq!(sim, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        let sim = cosine_similarity(&a.view(), &b.view());
        assert_abs_diff_eq!(sim, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a.view(), &b.view());
        assert_abs_diff_eq!(sim, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn l2_normalize_has_unit_norm() {
        let v = array![3.0, 4.0];
        let normed = l2_normalize(&v.view());
        let norm = normed.dot(&normed).sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_stays_zero() {
        let v = array![0.0, 0.0, 0.0];
        let normed = l2_normalize(&v.view());
        assert_eq!(normed, array![0.0, 0.0, 0.0]);
    }

    #[test]
    fn logsumexp_matches_naive() {
        let a = array![1.0, 2.0, 3.0];
        let naive = a.mapv(|x: f32| x.exp()).sum().ln();
        let stable = logsumexp(&a.view());
        assert_abs_diff_eq!(stable, naive, epsilon = 1e-5);
    }

    #[test]
    fn logsumexp_large_values_stability() {
        // naive exp would overflow, but logsumexp should handle it
        let a = array![1000.0, 1001.0, 1002.0];
        let result = logsumexp(&a.view());
        assert!(result.is_finite());

        let shifted = array![0.0, 1.0, 2.0];
        let expected = logsumexp(&shifted.view()) + 1000.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-3);
    }

    #[test]
    fn centroid_known_case() {
        let embeddings =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]).unwrap();
        let c = centroid(&embeddings.view());
        assert_eq!(c, array![2.0, 3.0, 4.0]);
    }

    #[test]
    fn cosine_distance_matrix_diagonal_is_zero() {
        let embeddings =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let dist = cosine_distance_matrix(&embeddings.view());

        for i in 0..3 {
            assert_abs_diff_eq!(dist[[i, i]], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn cosine_distance_matrix_is_symmetric() {
        let embeddings =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let dist = cosine_distance_matrix(&embeddings.view());

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(dist[[i, j]], dist[[j, i]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn cosine_distance_matrix_matches_fixture() {
        let input: Array2<f32> =
            Array2::read_npy(File::open(fixture_path("cosine_sim_input.npy")).unwrap()).unwrap();
        let expected_sim: Array2<f32> =
            Array2::read_npy(File::open(fixture_path("cosine_sim_expected.npy")).unwrap()).unwrap();

        let result = cosine_distance_matrix(&input.view());
        let expected_dist = 1.0 - expected_sim;

        assert_eq!(result.shape(), expected_dist.shape());
        for (a, b) in result.iter().zip(expected_dist.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }
}
