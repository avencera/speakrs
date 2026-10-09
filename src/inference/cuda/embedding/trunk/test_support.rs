//! Explicit choices for direct convolution development checks

use crate::inference::cuda::implementation::Choice;

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
