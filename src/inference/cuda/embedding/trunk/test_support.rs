//! Per-layer choices used only by qualification

use super::{ConvLayer, ConvShape};
use crate::inference::cuda::implementation::Choice;

impl ConvLayer {
    /// Whether this is one of the 14 qualified convolutions
    pub(crate) fn eligible(&self) -> bool {
        eligible(&self.plan.name, self.shape)
    }

    pub(crate) fn select(&mut self, name: &str, choice: Choice) -> bool {
        if self.plan.name != name || !self.eligible() {
            return false;
        }

        self.plan.override_choice = Some(choice);
        true
    }
}

/// Whether a layer is one of the 14 qualified convolutions: the 3x3 layers of the 32-
/// and 64-channel stages, not the stem and not the later stages
fn eligible(name: &str, shape: ConvShape) -> bool {
    name.starts_with("resnet.layer")
        && shape.kernel == 3
        && [32, 64].contains(&shape.in_channels)
        && [32, 64].contains(&shape.out_channels)
}
