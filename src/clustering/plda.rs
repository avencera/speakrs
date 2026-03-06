use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Debug, Clone)]
pub struct PldaTransform {
    pub mu: Array1<f32>,
    pub phi: Array2<f32>,
}

impl PldaTransform {
    pub fn new(mu: Array1<f32>, phi: Array2<f32>) -> Self {
        Self { mu, phi }
    }

    /// Transform a batch of embeddings: (X - mu) @ phi.T
    pub fn transform(&self, embeddings: &ArrayView2<f32>) -> Array2<f32> {
        let centered = embeddings - &self.mu;
        centered.dot(&self.phi.t())
    }

    pub fn transform_one(&self, embedding: &ArrayView1<f32>) -> Array1<f32> {
        let centered = embedding - &self.mu;
        centered.dot(&self.phi.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_transform() {
        let mu = Array1::zeros(3);
        let phi = Array2::eye(3);
        let plda = PldaTransform::new(mu, phi);

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = plda.transform(&input.view());
        assert_eq!(result, input);
    }

    #[test]
    fn mean_subtraction() {
        let mu = array![1.0, 1.0, 1.0];
        let phi = Array2::eye(3);
        let plda = PldaTransform::new(mu, phi);

        let input = array![[2.0, 3.0, 4.0]];
        let result = plda.transform(&input.view());
        assert_eq!(result, array![[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn dimensionality_reduction() {
        let mu = Array1::zeros(4);
        // phi is (2, 4) to reduce 4-dim to 2-dim
        let phi = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let plda = PldaTransform::new(mu, phi);

        let input = array![[1.0, 2.0, 3.0, 4.0]];
        let result = plda.transform(&input.view());
        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(result, array![[1.0, 2.0]]);
    }

    #[test]
    fn transform_one_consistency() {
        let mu = array![1.0, 0.0];
        let phi = Array2::eye(2);
        let plda = PldaTransform::new(mu, phi);

        let input = array![3.0, 5.0];
        let result = plda.transform_one(&input.view());
        assert_eq!(result, array![2.0, 5.0]);
    }

    #[test]
    fn batch_matches_single() {
        let mu = array![0.5, 0.5, 0.5];
        let phi = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let plda = PldaTransform::new(mu, phi);

        let batch = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let batch_result = plda.transform(&batch.view());

        for i in 0..2 {
            let single = plda.transform_one(&batch.row(i));
            assert_eq!(batch_result.row(i), single.view());
        }
    }
}
