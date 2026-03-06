pub mod plda;
pub mod vbx;

use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub labels: Vec<usize>,
    pub num_clusters: usize,
    pub centroids: Vec<Array1<f32>>,
}
