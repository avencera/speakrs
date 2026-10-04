//! Locked dispatch for the 14 qualified ResNet convolutions
//!
//! The harness traces, times and sanitizes exactly what this file runs, so a candidate
//! never chooses its own scope, timing or fallback. An eligible layer with the `Oxide`
//! choice runs its candidate plan only at the batch sizes the candidate's coverage
//! declares, and the Library path at every other batch size

use cudarc::driver::{CudaView, CudaViewMut};

use super::super::candidate::{ConvCandidate, ConvInputs, ConvLayerSpec, ConvOxide, Phases};
use super::super::dnn::Residual;
use super::super::implementation::Choice;
use super::super::{CudaMath, CudaRuntime};
use super::trunk::{ConvLayer, Trunk};
use super::{Convs, CudaError};

impl Convs<'_> {
    /// `y = relu(conv(x, layer.weight) + residual + layer.bias)` through the layer's choice
    pub(super) fn conv_bias_relu(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        residual: Residual<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        match layer.choice(self.chunks, self.math) {
            Choice::Library => self.library(layer, x, residual, y),
            Choice::Oxide => self.oxide(layer, x, residual, y),
            #[cfg(test)]
            Choice::Mutant(mutant) => {
                super::super::test_support::poison(self.runtime)?;
                {
                    let _scope =
                        super::super::test_support::candidate(self.runtime.stream(), layer.name());
                    if !mutant.skips() {
                        super::test_support::mutant_conv(self, layer, x, residual, y, mutant)?;
                    }
                }
                super::super::test_support::unscoped(self.runtime, mutant)
            }
        }
    }

    /// The deterministic cuDNN plan with its fused epilogue, in one call
    fn library(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        residual: Residual<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        #[cfg(test)]
        let _scope = layer
            .eligible()
            .then(|| super::super::test_support::library(self.runtime.stream(), layer.name()));
        self.plan(layer).forward_bias_relu(
            &mut self.workspace.as_view_mut(),
            x,
            &layer.weight().data().as_view(),
            &layer.bias().data().as_view(),
            residual,
            y,
        )?;
        #[cfg(test)]
        super::super::test_support::perturb(self.runtime, layer.name(), y)?;
        Ok(())
    }

    fn oxide(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        residual: Residual<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let candidates = self.candidates;
        let plan = candidates
            .iter()
            .find(|(name, _)| name == layer.name())
            .map(|(_, plan)| plan);
        // an undeclared pair runs the Library path
        let Some(plan) = plan else {
            return self.library(layer, x, residual, y);
        };

        let residual = match residual {
            Residual::Add(value) => Some(value),
            Residual::None { .. } => None,
        };
        let stream = self.runtime.stream();
        #[cfg(test)]
        super::super::test_support::poison(self.runtime)?;
        #[cfg(test)]
        let _scope = super::super::test_support::candidate(stream, layer.name());
        plan.enqueue(
            ConvInputs {
                x,
                residual,
                weight: &layer.weight().data().as_view(),
                bias: &layer.bias().data().as_view(),
            },
            y,
            &Phases::new(),
            stream,
        )
    }
}

/// Plans the candidate for every eligible `Oxide` layer its coverage declares at `batch`
///
/// Runs when a batch class is set up, outside every timed and traced interval
pub(super) fn plan_candidates(
    runtime: &CudaRuntime,
    trunk: &Trunk,
    batch: usize,
    math: CudaMath,
) -> Result<Vec<(String, ConvOxide)>, CudaError> {
    let layers = trunk
        .blocks
        .iter()
        .flat_map(|block| [(&block.conv1, false), (&block.conv2, true)]);
    let mut plans = Vec::new();
    for (layer, residual) in layers {
        if layer.choice(batch, math) != Choice::Oxide
            || !ConvOxide::COVERAGE.covers(layer.name(), batch, math)
        {
            continue;
        }

        let spec = ConvLayerSpec {
            name: layer.name(),
            conv: layer.conv(batch, math),
            residual,
            weight: layer.weight().data(),
            bias: layer.bias().data(),
        };
        #[cfg(test)]
        let _scope = super::super::test_support::plan(layer.name());
        plans.push((layer.name().to_owned(), ConvOxide::plan(runtime, spec)?));
    }

    Ok(plans)
}
