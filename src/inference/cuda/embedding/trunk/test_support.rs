//! Per-layer choices used only by qualification

use super::{ConvLayer, ConvShape};
use crate::inference::cuda::implementation::Choice;

impl ConvLayer {
    /// Whether this is one of the 14 qualified convolutions
    pub(crate) fn eligible(&self) -> bool {
        eligible(self.plan.boundary.name(), self.shape)
    }

    pub(crate) fn select(&mut self, name: &str, choice: Choice) -> bool {
        if self.plan.boundary.name() != name || !self.eligible() {
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

impl super::super::ResNetEmbedding {
    /// Requests `choice` for every trunk convolution, eligible or not, before a batch
    /// shares the model; development checks of the driver-only trunk use it
    pub(crate) fn select_every_conv(&mut self, choice: Choice) -> bool {
        let Some(model) = std::sync::Arc::get_mut(&mut self.0) else {
            return false;
        };

        let trunk = &mut model.trunk;
        trunk.stem.plan.override_choice = Some(choice);
        for block in &mut trunk.blocks {
            for layer in [&mut block.conv1, &mut block.conv2]
                .into_iter()
                .chain(block.shortcut.as_mut())
            {
                layer.plan.override_choice = Some(choice);
            }
        }

        true
    }
}

impl super::super::EmbeddingBatch {
    /// Trunk convolutions this batch runs on a library plan
    pub(crate) fn library_convs(&self) -> Vec<&str> {
        self.plans
            .iter()
            .filter(|(_, plan)| plan.choice() == Choice::Library)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}
