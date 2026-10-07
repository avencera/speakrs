//! Library plans and stage owners for segmentation dense boundaries

use super::kernels::SegmentationKernels;
use super::shape::LEAKY_SLOPE;
use crate::inference::cuda::candidate::{DenseEpilogue, DenseSpec, Phases, PlanError, SegConvSpec};
use crate::inference::cuda::dnn::{Conv2d, ConvPlan, ConvPlanner};
use crate::inference::cuda::{CudaError, CudaRuntime, Sgemm};
use cudarc::driver::CudaSlice;
use std::cell::RefCell;

/// Same GEMM and epilogue as the full model Library path
pub(crate) struct DenseLibrary {
    spec: DenseSpec,
    kernels: SegmentationKernels,
    head: Option<crate::inference::cuda::embedding::test_support::HeadBias>,
}

impl DenseLibrary {
    /// Prepare Library functions before capture, without a candidate artifact
    pub(crate) fn new(runtime: &CudaRuntime, spec: DenseSpec) -> Result<Self, PlanError> {
        Ok(Self {
            spec,
            kernels: SegmentationKernels::load(runtime)?,
            head: if spec.epilogue() == DenseEpilogue::Bias {
                Some(crate::inference::cuda::embedding::test_support::HeadBias::new(runtime)?)
            } else {
                None
            },
        })
    }
    /// Execute the original Library operation without an output copy
    pub(crate) fn enqueue(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        _phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        let (rows, n, k) = self.spec.dimensions();
        if self.spec.epilogue() == DenseEpilogue::Bias {
            self.head
                .as_ref()
                .expect("prepared embedding head")
                .run(runtime, bias, output)?;
        }
        runtime.sgemm(
            Sgemm {
                b_transposed: self.spec.transposed_weights(),
                beta: self.spec.beta(),
                math: self.spec.math(),
                ..Sgemm::new(self.spec.batch() * rows, n, k)
            },
            x,
            weight,
            output,
        )?;
        match self.spec.epilogue() {
            DenseEpilogue::Bias => Ok(()),
            DenseEpilogue::BiasLeakyRelu => {
                self.kernels.bias_leaky(runtime, bias, LEAKY_SLOPE, output)
            }
            DenseEpilogue::BiasLogSoftmax => self.kernels.bias_log_softmax(runtime, bias, output),
        }
    }
}

/// Valid five-tap Library convolution; bias stays in the shared pool consumer
pub(crate) struct SegConvLibrary {
    plan: ConvPlan,
    workspace: RefCell<CudaSlice<u8>>,
}

impl SegConvLibrary {
    /// Prepare the unchanged Library convolution without candidate modules
    pub(crate) fn new(runtime: &CudaRuntime, spec: SegConvSpec) -> Result<Self, PlanError> {
        let plan = ConvPlanner::new(runtime)?.plan(Conv2d {
            batch: spec.batch(),
            in_channels: spec.in_channels(),
            out_channels: spec.out_channels(),
            input: [1, spec.input_steps()],
            kernel: [1, spec.kernel()],
            padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            math: spec.math(),
        })?;
        let workspace = RefCell::new(
            runtime
                .stream()
                .alloc_zeros(plan.workspace_bytes().max(1))?,
        );
        Ok(Self { plan, workspace })
    }
    /// Execute the original Library operation without an output copy
    pub(crate) fn enqueue(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        _phases: &Phases,
        _runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        self.plan.forward(
            &mut self.workspace.borrow_mut().as_view_mut(),
            &x.as_view(),
            &weight.as_view(),
            &mut output.as_view_mut(),
        )
    }
}

/// Run one linear Library operation through its optional test-only route
pub(super) fn dense(
    owner: Option<&crate::inference::cuda::test_support::boundaries::Owner>,
    runtime: &CudaRuntime,
    spec: Sgemm,
    site: crate::inference::cuda::candidate::DenseSite,
    inputs: crate::inference::cuda::test_support::candidate_seam::Slices<'_>,
    output: &mut CudaSlice<f32>,
    mut epilogue: impl FnMut(&mut CudaSlice<f32>) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    use crate::inference::cuda::test_support::{boundaries, candidate_seam::Operation};
    let input = inputs.input;
    let weight = inputs.weight;
    // a stage without a test owner keeps the original Library path for all model shapes
    let Some(owner) = owner else {
        runtime.sgemm(spec, input, weight, output)?;
        return epilogue(output);
    };

    let batch = input.len()
        / DenseSpec::new(site, 1, spec.math)
            .map_err(|e| CudaError::Unsupported {
                context: "stage dense spec",
                reason: e.to_string(),
            })?
            .input_len();
    let operation = Operation::Dense(DenseSpec::new(site, batch, spec.math).map_err(|e| {
        CudaError::Unsupported {
            context: "stage dense spec",
            reason: e.to_string(),
        }
    })?);
    boundaries::run_slices(Some(owner), runtime, operation, inputs, output, |output| {
        runtime.sgemm(spec, input, weight, output)?;
        epilogue(output)
    })
}

/// Export exact host weights through the existing checked ONNX loader
pub(crate) fn weights(
    file: &crate::inference::cuda::SafetensorsFile,
    site: &str,
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let weights = super::weights::SegmentationWeights::load(file)?;
    let layer = match site {
        "conv1" => &weights.conv1,
        "conv2" => &weights.conv2,
        "linear0" => &weights.linear[0],
        "linear1" => &weights.linear[1],
        "classifier" => &weights.linear[2],
        _ => unreachable!("fixed segmentation collection"),
    };
    Ok((
        layer.weight.clone(),
        if site.starts_with("conv") {
            Vec::new()
        } else {
            layer.bias.clone()
        },
    ))
}
