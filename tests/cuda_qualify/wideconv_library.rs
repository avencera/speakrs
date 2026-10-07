//! Library implementation of every wide-convolution epilogue

use crate::inference::cuda::candidate::{
    ConvCandidate, ConvInputs, ConvLayerSpec, Coverage, Epilogue, FiniteContract, InfinityContract,
    NanContract, Phases, PlanError, SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::dnn::{ConvPlan, ConvPlanner};
use crate::inference::cuda::geometry::{Conv2d, Residual};
use crate::inference::cuda::{CudaError, CudaRuntime, KernelModule, LoadedKernels};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg,
};
use std::cell::RefCell;

/// NCHW Library plan with a bias-only shortcut or fused ReLU epilogue
pub(crate) struct Library {
    plan: ConvPlan,
    workspace: RefCell<CudaSlice<u8>>,
    scratch: CudaSlice<f32>,
    bias: CudaFunction,
    epilogue: Epilogue,
    spec: Conv2d,
}

impl Library {
    /// Build the same cuDNN geometry as the model, without a candidate artifact
    pub(crate) fn new(runtime: &CudaRuntime, layer: ConvLayerSpec<'_>) -> Result<Self, PlanError> {
        let plan = ConvPlanner::new(runtime)?.plan(layer.conv)?;
        let workspace = RefCell::new(
            runtime
                .stream()
                .alloc_zeros(plan.workspace_bytes().max(1))?,
        );
        Ok(Self {
            scratch: runtime
                .stream()
                .alloc_zeros(layer.conv.output_shape().iter().product())?,
            bias: runtime
                .load_kernels(KernelModule::Embedding)?
                .function("embedding_bias")?,
            workspace,
            plan,
            epilogue: layer.epilogue,
            spec: layer.conv,
        })
    }
}

impl ConvCandidate for Library {
    type Pin = ();
    const COVERAGE: Coverage = Coverage::NONE;
    const SPECIAL_VALUES: SpecialValues = SpecialValues {
        finite: FiniteContract::AbsoluteSum { headroom: 2 },
        nan: NanContract::Unspecified,
        infinity: InfinityContract::Unspecified,
        signed_zero: SignedZeroContract::Unspecified,
    };
    fn implemented_pin(_layer: &ConvLayerSpec<'_>) -> Result<(), PlanError> {
        Ok(())
    }
    fn plan(
        runtime: &CudaRuntime,
        _kernels: &LoadedKernels,
        layer: ConvLayerSpec<'_>,
        (): (),
    ) -> Result<Self, PlanError> {
        Self::new(runtime, layer)
    }
    fn enqueue(
        &self,
        inputs: ConvInputs<'_, '_>,
        output: &mut CudaViewMut<'_, f32>,
        _phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        if inputs.residual.is_some() != (self.epilogue == Epilogue::BiasReluResidual) {
            return Err(CudaError::Unsupported {
                context: "wideconv Library epilogue",
                reason: "residual does not match epilogue".to_owned(),
            });
        }
        let mut workspace = self.workspace.borrow_mut();
        if self.epilogue != Epilogue::Bias {
            let scratch = self.scratch.as_view();
            let residual = inputs
                .residual
                .map_or(Residual::None { scratch: &scratch }, Residual::Add);
            return self.plan.forward_bias_relu(
                &mut workspace.as_view_mut(),
                inputs.x,
                inputs.weight,
                inputs.bias,
                residual,
                output,
            );
        }
        self.plan.forward(
            &mut workspace.as_view_mut(),
            inputs.x,
            inputs.weight,
            output,
        )?;
        let [h, w] = self.spec.output();
        let plane = (h * w) as u32;
        let channels = self.spec.out_channels as u32;
        let bias_len = inputs.bias.len() as u64;
        let output_len = output.len() as u64;
        let _fixed = crate::inference::cuda::test_support::fixed("embedding_bias");
        let mut launch = stream.launch_builder(&self.bias);
        launch
            .arg(inputs.bias)
            .arg(&bias_len)
            .arg(&channels)
            .arg(&plane)
            .arg(output)
            .arg(&output_len);
        // SAFETY: the checked cuDNN output is NCHW, with one bias per channel; this is the locked embedding_bias ABI
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    plane.div_ceil(256 * 4),
                    (self.spec.batch * self.spec.out_channels) as u32,
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }
}
