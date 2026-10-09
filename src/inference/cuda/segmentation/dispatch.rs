//! Dispatch for the Sinc producer and the LSTM stack
//!
//! A candidate enqueues only its own boundary: for Sinc, the producer, after which this file runs
//! the unchanged shared `segmentation_pool_norm` consumer with the parameters that
//! match the candidate's declared output. Undeclared batch and math pairs run the
//! Library path

use cudarc::driver::{CudaSlice, CudaViewMut};

use super::super::candidate::{
    LstmCandidate, LstmLayerWeights, LstmPhases, LstmSpec, Phases, Projection, SincCandidate,
    SincInputs, SincOutput, SincOxide, SincSpec,
};
use super::super::implementation::{AreaTarget, BoundaryId, LibraryNeed, Selected};
use super::super::{CudaLibrary, KernelModule};
#[cfg(feature = "_cuda-libraries")]
use super::super::{CudaLstmAlgorithm, dnn::ConvPlanner};
use super::shape::{LEAKY_SLOPE, NORM_EPSILON, POOL, SINC_CHANNELS, SegmentationShape};
use super::{CudaError, CudaRuntime, LstmStage, Network, PoolNorm, RowLayout, SincPlan};

/// The boundary names the coverage and the production table use
pub(super) const SINC_LAYER: &str = "sincnet.conv0.abs_pool";
pub(super) const LSTM_LAYER: &str = "lstm.stack";
pub(super) const SINC: BoundaryId = BoundaryId::named(SINC_LAYER);
pub(super) const LSTM: BoundaryId = BoundaryId::named(LSTM_LAYER);

/// The buffers of one Sinc call: the normalized waveform in, the stage-0 activation out
pub(super) struct SincIo<'a> {
    pub(super) input: &'a CudaSlice<f32>,
    /// `[batch, 80, sinc]` raw convolution
    pub(super) raw: &'a mut CudaSlice<f32>,
    /// `[batch, 80, pool0]` pooled boundary, present when a pooling candidate may run
    pub(super) pooled: Option<&'a mut CudaSlice<f32>>,
    /// `[batch, 80, pool0]` after normalization and LeakyReLU
    pub(super) stage0: &'a mut CudaSlice<f32>,
}

impl Network {
    /// The Sinc producer through its choice, then the shared consumer
    pub(super) fn sinc_forward(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        plan: &SincPlan,
        workspace: &mut CudaViewMut<'_, u8>,
        io: SincIo<'_>,
    ) -> Result<(), CudaError> {
        #[cfg(not(feature = "_cuda-libraries"))]
        let _ = workspace;
        match plan {
            #[cfg(feature = "_cuda-libraries")]
            SincPlan::Library(library) => self.sinc_library(runtime, shape, library, workspace, io),
            SincPlan::Oxide(candidate) => self.sinc_oxide(runtime, shape, candidate, io),
        }
    }

    /// The cuDNN convolution, then `abs`, pool by 3 and normalize in one kernel
    #[cfg(feature = "_cuda-libraries")]
    fn sinc_library(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        plan: &super::ConvPlan,
        workspace: &mut CudaViewMut<'_, u8>,
        io: SincIo<'_>,
    ) -> Result<(), CudaError> {
        plan.forward(
            workspace,
            &io.input.as_view(),
            &self.sinc_filters.as_view(),
            &mut io.raw.as_view_mut(),
        )?;
        self.sinc_consumer(runtime, shape, false, io.raw, io.stage0)
    }

    /// The unchanged shared consumer: instance norm and LeakyReLU, pooling first when
    /// its input is the raw convolution
    fn sinc_consumer(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        pooled: bool,
        input: &CudaSlice<f32>,
        stage0: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let (in_len, pool) = if pooled {
            (shape.pool0, 1)
        } else {
            (shape.sinc, POOL)
        };
        let spec = PoolNorm {
            batch: shape.batch,
            channels: SINC_CHANNELS,
            in_len,
            pool,
            abs_input: !pooled,
            slope: LEAKY_SLOPE,
            epsilon: NORM_EPSILON,
            layout: RowLayout::channels_first(SINC_CHANNELS, shape.pool0),
        };
        let [gamma, beta] = &self.norms[0];
        self.kernels
            .pool_norm(runtime, spec, input, &self.zero_bias, gamma, beta, stage0)
    }

    fn sinc_oxide(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        candidate: &SincOxide,
        io: SincIo<'_>,
    ) -> Result<(), CudaError> {
        let pooled = SincOxide::OUTPUT == SincOutput::Pooled;
        let SincIo {
            input,
            raw,
            pooled: pooled_buffer,
            stage0,
        } = io;
        let output = match (pooled, pooled_buffer) {
            (false, _) => raw,
            (true, Some(buffer)) => buffer,
            (true, None) => {
                return Err(CudaError::Unsupported {
                    context: "Sinc candidate",
                    reason: "no pooled buffer for a pooling candidate".to_owned(),
                });
            }
        };
        candidate.enqueue(
            SincInputs {
                waveform: &input.as_view(),
                filters: &self.sinc_filters.as_view(),
            },
            &mut output.as_view_mut(),
            &Phases::new(),
            runtime.stream(),
        )?;
        self.sinc_consumer(runtime, shape, pooled, output, stage0)
    }

    /// Build one Sinc owner from the selected token
    pub(super) fn plan_sinc(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        selected: Selected,
    ) -> Result<SincPlan, CudaError> {
        let _selected = match selected {
            Selected::Oxide(token) => {
                let spec = SincSpec {
                    batch: shape.batch,
                    samples: shape.samples,
                    sinc: shape.sinc,
                    pooled: shape.pool0,
                    math: self.options.math,
                    filters: &self.sinc_filters,
                };
                if let Some(candidate) = token.sinc(runtime, spec)? {
                    return Ok(SincPlan::Oxide(candidate));
                }
                Selected::Library
            }
            other => other,
        };
        LibraryNeed::new(
            SINC,
            shape.batch,
            self.options.math,
            AreaTarget::for_area(runtime, KernelModule::Sincnet)?,
            CudaLibrary::Cudnn,
        )
        .prepare(runtime)?;
        #[cfg(feature = "_cuda-libraries")]
        {
            let spec = super::Conv2d {
                batch: shape.batch,
                in_channels: 1,
                out_channels: SINC_CHANNELS,
                input: [1, shape.samples],
                kernel: [1, super::SINC_KERNEL],
                padding: [0, 0],
                stride: [1, super::SINC_STRIDE],
                dilation: [1, 1],
                math: self.options.math,
            };
            let library = ConvPlanner::new(runtime)?.plan(spec)?;
            Ok(SincPlan::Library(library))
        }
        #[cfg(not(feature = "_cuda-libraries"))]
        {
            let _ = _selected;
            Err(CudaError::LibraryUnavailable {
                library: CudaLibrary::Cudnn,
            })
        }
    }

    /// The stack through its choice
    pub(super) fn lstm_forward(
        &self,
        runtime: &CudaRuntime,
        stage: &mut LstmStage,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        match stage {
            #[cfg(feature = "_cuda-libraries")]
            LstmStage::Library(plan) => self.lstm_library(runtime, plan, input, output),
            LstmStage::Projected { candidate, rows } => {
                let phases = LstmPhases::new(Projection::new(runtime, *rows, self.options.math));
                candidate.enqueue(
                    &input.as_view(),
                    &mut output.as_view_mut(),
                    &phases,
                    runtime.stream(),
                )
            }
            LstmStage::Oxide { candidate, rows } => {
                let stream = runtime.stream();
                let phases = LstmPhases::new(Projection::new(runtime, *rows, self.options.math));
                candidate.enqueue(&input.as_view(), &mut output.as_view_mut(), &phases, stream)
            }
        }
    }

    #[cfg(feature = "_cuda-libraries")]
    fn lstm_library(
        &self,
        runtime: &CudaRuntime,
        plan: &mut super::LstmPlan,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        self.lstm
            .as_ref()
            .expect("Library LSTM initialized by planning")
            .forward(runtime, plan, input, output)?;
        Ok(())
    }

    /// Build one stack owner; Library descriptors and weight packing are deferred
    pub(super) fn plan_lstm(
        &mut self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        selected: Selected,
    ) -> Result<LstmStage, CudaError> {
        let _selected = match selected {
            Selected::Oxide(token) => {
                let layer = |index: usize| {
                    let layer = &self.lstm_weights[index];
                    LstmLayerWeights {
                        input: layer.input,
                        w: &layer.w,
                        r: &layer.r,
                        b: &layer.b,
                    }
                };
                let spec = LstmSpec {
                    batch: shape.batch,
                    frames: shape.frames,
                    math: self.options.math,
                    layers: [layer(0), layer(1), layer(2), layer(3)],
                };
                if token.area() == KernelModule::LstmProj {
                    if let Some(candidate) = token.projected_lstm(runtime, spec)? {
                        return Ok(LstmStage::Projected {
                            candidate: Box::new(candidate),
                            rows: shape.batch * shape.frames,
                        });
                    }
                } else if let Some(candidate) = token.lstm(runtime, spec)? {
                    return Ok(LstmStage::Oxide {
                        candidate: Box::new(candidate),
                        rows: shape.batch * shape.frames,
                    });
                }
                Selected::Library
            }
            other => other,
        };
        LibraryNeed::new(
            LSTM,
            shape.batch,
            self.options.math,
            AreaTarget::for_area(runtime, KernelModule::Lstm)?,
            CudaLibrary::Cudnn,
        )
        .prepare(runtime)?;
        #[cfg(feature = "_cuda-libraries")]
        {
            // the NVRTC library is a dependency of the selected dynamic Library plan only
            if self.options.lstm_algo == CudaLstmAlgorithm::PersistDynamic {
                LibraryNeed::new(
                    LSTM,
                    shape.batch,
                    self.options.math,
                    AreaTarget::for_area(runtime, KernelModule::Lstm)?,
                    CudaLibrary::Nvrtc,
                )
                .prepare(runtime)?;
            }
            if self.lstm.is_none() {
                let layers = self.lstm_weights.as_slice().try_into().map_err(|_| {
                    CudaError::Unsupported {
                        context: "LSTM weights",
                        reason: "expected four bidirectional layers".to_owned(),
                    }
                })?;
                self.lstm = Some(super::CudnnLstm::new(
                    runtime,
                    layers,
                    self.options.math,
                    self.options.lstm_algo,
                )?);
            }
            let library = self.lstm.as_ref().expect("Library LSTM initialized").plan(
                runtime,
                shape.batch,
                shape.frames,
            )?;
            Ok(LstmStage::Library(library))
        }
        #[cfg(not(feature = "_cuda-libraries"))]
        {
            let _ = _selected;
            Err(CudaError::LibraryUnavailable {
                library: CudaLibrary::Cudnn,
            })
        }
    }
}
