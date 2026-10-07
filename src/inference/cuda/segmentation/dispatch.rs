//! Locked dispatch for the Sinc producer and the LSTM stack
//!
//! The harness traces, times and sanitizes exactly what this file runs. A candidate
//! enqueues only its own boundary: for Sinc, the producer, after which this file runs
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
#[cfg(feature = "cuda")]
use super::super::{CudaLstmAlgorithm, dnn::ConvPlanner};
use super::shape::{LEAKY_SLOPE, NORM_EPSILON, POOL, SINC_CHANNELS, SegmentationShape};
use super::{CudaError, CudaRuntime, LstmStage, Network, PoolNorm, RowLayout, SincPlan};

/// The boundary names the harness, the coverage and the production table use
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
        #[cfg(not(feature = "cuda"))]
        let _ = workspace;
        match plan {
            #[cfg(feature = "cuda")]
            SincPlan::Library(library) => self.sinc_library(runtime, shape, library, workspace, io),
            SincPlan::Oxide(candidate) => self.sinc_oxide(runtime, shape, candidate, io),
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            SincPlan::Mutant { library, mutant } => {
                let mutant = *mutant;
                super::super::test_support::poison(runtime)?;
                {
                    let _scope = super::super::test_support::mutant_scope(
                        runtime.stream(),
                        SINC_LAYER,
                        mutant,
                    );
                    if !mutant.skips() {
                        super::test_support::mutant_sinc(
                            self, runtime, shape, library, workspace, io.input, io.raw, mutant,
                        )?;
                    }
                }
                super::super::test_support::unscoped(runtime, mutant)?;
                self.sinc_consumer(runtime, shape, false, io.raw, io.stage0)
            }
        }
    }

    /// The cuDNN convolution, then `abs`, pool by 3 and normalize in one kernel
    #[cfg(feature = "cuda")]
    fn sinc_library(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        plan: &super::ConvPlan,
        workspace: &mut CudaViewMut<'_, u8>,
        io: SincIo<'_>,
    ) -> Result<(), CudaError> {
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        let _scope = super::super::test_support::library(runtime.stream(), SINC_LAYER);
        plan.forward(
            workspace,
            &io.input.as_view(),
            &self.sinc_filters.as_view(),
            &mut io.raw.as_view_mut(),
        )?;
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        super::super::test_support::perturb(runtime, SINC_LAYER, io.raw)?;
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
        let stream = runtime.stream();
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        super::super::test_support::poison(runtime)?;
        {
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            let _scope = super::super::test_support::candidate(stream, SINC_LAYER);
            candidate.enqueue(
                SincInputs {
                    waveform: &input.as_view(),
                    filters: &self.sinc_filters.as_view(),
                },
                &mut output.as_view_mut(),
                &Phases::new(),
                stream,
            )?;
        }

        self.sinc_consumer(runtime, shape, pooled, output, stage0)
    }

    /// Build one Sinc owner from the selected qualification token
    pub(super) fn plan_sinc(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        selected: Selected,
    ) -> Result<SincPlan, CudaError> {
        let selected = match selected {
            Selected::Oxide(token) => {
                let spec = SincSpec {
                    batch: shape.batch,
                    samples: shape.samples,
                    sinc: shape.sinc,
                    pooled: shape.pool0,
                    math: self.options.math,
                    filters: &self.sinc_filters,
                };
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                let _scope = super::super::test_support::plan(SINC_LAYER);
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
        #[cfg(feature = "cuda")]
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
            Ok(match selected {
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                Selected::Mutant(mutant) => SincPlan::Mutant { library, mutant },
                _ => SincPlan::Library(library),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = selected;
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
            #[cfg(feature = "cuda")]
            LstmStage::Library(plan) => self.lstm_library(runtime, plan, input, output),
            LstmStage::Oxide { candidate, rows } => {
                let stream = runtime.stream();
                let phases = LstmPhases::new(Projection::new(runtime, *rows, self.options.math));
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                super::super::test_support::poison(runtime)?;
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                let _scope = super::super::test_support::candidate(stream, LSTM_LAYER);
                candidate.enqueue(&input.as_view(), &mut output.as_view_mut(), &phases, stream)
            }
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            LstmStage::Mutant { library, mutant } => {
                let mutant = *mutant;
                super::super::test_support::poison(runtime)?;
                {
                    let _scope = super::super::test_support::mutant_scope(
                        runtime.stream(),
                        LSTM_LAYER,
                        mutant,
                    );
                    if !mutant.skips() {
                        super::test_support::mutant_lstm(
                            self, runtime, library, input, output, mutant,
                        )?;
                    }
                }
                super::super::test_support::unscoped(runtime, mutant)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn lstm_library(
        &self,
        runtime: &CudaRuntime,
        plan: &mut super::LstmPlan,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        let _scope = super::super::test_support::library(runtime.stream(), LSTM_LAYER);
        self.lstm
            .as_ref()
            .expect("Library LSTM initialized by planning")
            .forward(runtime, plan, input, output)?;
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        super::super::test_support::perturb(runtime, LSTM_LAYER, output)?;
        Ok(())
    }

    /// Build one stack owner; Library descriptors and weight packing are deferred
    pub(super) fn plan_lstm(
        &mut self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        selected: Selected,
    ) -> Result<LstmStage, CudaError> {
        let selected = match selected {
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
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                let _scope = super::super::test_support::plan(LSTM_LAYER);
                if let Some(candidate) = token.lstm(runtime, spec)? {
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
        #[cfg(feature = "cuda")]
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
            Ok(match selected {
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                Selected::Mutant(mutant) => LstmStage::Mutant { library, mutant },
                _ => LstmStage::Library(library),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = selected;
            Err(CudaError::LibraryUnavailable {
                library: CudaLibrary::Cudnn,
            })
        }
    }
}
