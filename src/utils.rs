use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub fn l2_normalize(v: &ArrayView1<f32>) -> Array1<f32> {
    let norm = v.dot(v).sqrt();
    if norm == 0.0 {
        return Array1::zeros(v.len());
    }
    v / norm
}

pub fn l2_normalize_rows(embeddings: &ArrayView2<f32>) -> Array2<f32> {
    let mut normalized = embeddings.to_owned();
    for mut row in normalized.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
    normalized
}

pub fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let a_norm = l2_normalize(a);
    let b_norm = l2_normalize(b);
    a_norm.dot(&b_norm)
}

pub fn logsumexp(a: &ArrayView1<f32>) -> f32 {
    let max = a.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    if max.is_infinite() {
        return max;
    }

    let sum_exp = a.mapv(|x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

pub fn logsumexp_f64(a: &ArrayView1<f64>) -> f64 {
    let max = a.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    if max.is_infinite() {
        return max;
    }

    let sum_exp = a.mapv(|x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
        let a = array![1000.0, 1001.0, 1002.0];
        let result = logsumexp(&a.view());
        assert!(result.is_finite());

        let shifted = array![0.0, 1.0, 2.0];
        let expected = logsumexp(&shifted.view()) + 1000.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-3);
    }
}
