//! Shared host weights and deterministic recipes for native inference

pub(crate) mod fbank;
pub(crate) mod segmentation;
mod weights;

pub use weights::NativeWeightsError;
pub(crate) use weights::WeightsFile;

#[cfg(all(test, feature = "_cuda-libraries"))]
pub(crate) mod test_support {
    pub(crate) use super::weights::test_support::TestFile;
}
