use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub fn l2_normalize(vector: &ArrayView1<f32>) -> Array1<f32> {
    let norm = vector.dot(vector).sqrt();
    if norm == 0.0 {
        return Array1::zeros(vector.len());
    }
    vector / norm
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

pub fn l2_normalize_rows_f64(embeddings: &ArrayView2<f64>) -> Array2<f64> {
    let mut normalized = embeddings.to_owned();
    for mut row in normalized.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
    normalized
}

pub fn cosine_similarity(lhs: &ArrayView1<f32>, rhs: &ArrayView1<f32>) -> f32 {
    let lhs_norm = l2_normalize(lhs);
    let rhs_norm = l2_normalize(rhs);
    lhs_norm.dot(&rhs_norm)
}

pub fn logsumexp_f64(values: &ArrayView1<f64>) -> f64 {
    let max = values.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    if max.is_infinite() {
        return max;
    }

    let sum_exp = values.mapv(|x| (x - max).exp()).sum();
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
}
