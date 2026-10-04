//! Locked dispatch for the Sinc producer and the LSTM stack
//!
//! The harness traces, times and sanitizes exactly what this file runs. A candidate
//! enqueues only its own boundary: for Sinc, the producer, after which this file runs
//! the unchanged shared `segmentation_pool_norm` consumer with the parameters that
//! match the candidate's declared output. Undeclared batch and math pairs run the
//! Library path

use cudarc::driver::{CudaSlice, CudaViewMut};

use super::super::candidate::{
    Coverage, LstmCandidate, LstmLayerWeights, LstmOxide, LstmPhases, LstmSpec, Phases, Projection,
    SincCandidate, SincInputs, SincOutput, SincOxide, SincSpec,
};
use super::super::implementation::Choice;
use super::shape::{LEAKY_SLOPE, NORM_EPSILON, POOL, SINC_CHANNELS, SegmentationShape};
use super::{CudaError, CudaRuntime, LstmStage, Network, PoolNorm, RowLayout, SincPlan};

/// The boundary names the harness, the coverage and the production table use
pub(super) const SINC_LAYER: &str = "sincnet.conv0.abs_pool";
pub(super) const LSTM_LAYER: &str = "lstm.stack";

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
        match plan.choice {
            Choice::Library => self.sinc_library(runtime, shape, plan, workspace, io),
            Choice::Oxide(_) => self.sinc_oxide(runtime, shape, plan, workspace, io),
            #[cfg(test)]
            Choice::Mutant(mutant) => {
                super::super::test_support::poison(runtime)?;
                {
                    let _scope =
                        super::super::test_support::candidate(runtime.stream(), SINC_LAYER);
                    if !mutant.skips() {
                        super::test_support::mutant_sinc(
                            self, runtime, shape, plan, workspace, io.input, io.raw, mutant,
                        )?;
                    }
                }
                super::super::test_support::unscoped(runtime, mutant)?;
                self.sinc_consumer(runtime, shape, false, io.raw, io.stage0)
            }
        }
    }

    /// The cuDNN convolution, then `abs`, pool by 3 and normalize in one kernel
    fn sinc_library(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        plan: &SincPlan,
        workspace: &mut CudaViewMut<'_, u8>,
        io: SincIo<'_>,
    ) -> Result<(), CudaError> {
        #[cfg(test)]
        let _scope = super::super::test_support::library(runtime.stream(), SINC_LAYER);
        plan.library.forward(
            workspace,
            &io.input.as_view(),
            &self.sinc_filters.as_view(),
            &mut io.raw.as_view_mut(),
        )?;
        #[cfg(test)]
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
        plan: &SincPlan,
        workspace: &mut CudaViewMut<'_, u8>,
        io: SincIo<'_>,
    ) -> Result<(), CudaError> {
        // an undeclared pair runs the Library path
        let Some(candidate) = &plan.candidate else {
            return self.sinc_library(runtime, shape, plan, workspace, io);
        };

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
        #[cfg(test)]
        super::super::test_support::poison(runtime)?;
        {
            #[cfg(test)]
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

    /// Plans the Sinc candidate when `choice` is `Oxide` and its coverage declares this
    /// batch and mode
    pub(super) fn plan_sinc(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        choice: Choice,
    ) -> Result<Option<SincOxide>, CudaError> {
        if !declared(choice, SincOxide::COVERAGE, SINC_LAYER, shape.batch, self) {
            return Ok(None);
        }

        let spec = SincSpec {
            batch: shape.batch,
            samples: shape.samples,
            sinc: shape.sinc,
            pooled: shape.pool0,
            math: self.options.math,
            filters: &self.sinc_filters,
        };
        #[cfg(test)]
        let _scope = super::super::test_support::plan(SINC_LAYER);
        super::super::dispatch::candidate_plan(
            choice,
            SINC_LAYER,
            shape.batch,
            SincOxide::plan(runtime, spec),
        )
    }

    /// The stack through its choice
    pub(super) fn lstm_forward(
        &self,
        runtime: &CudaRuntime,
        stage: &mut LstmStage,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        match stage.choice {
            Choice::Library => self.lstm_library(runtime, stage, input, output),
            Choice::Oxide(_) => {
                // an undeclared pair runs the Library path
                let Some(candidate) = &stage.candidate else {
                    return self.lstm_library(runtime, stage, input, output);
                };

                let stream = runtime.stream();
                let rows = stage.library.batch() * stage.library.seq_len();
                let phases = LstmPhases::new(Projection::new(runtime, rows, self.options.math));
                #[cfg(test)]
                super::super::test_support::poison(runtime)?;
                #[cfg(test)]
                let _scope = super::super::test_support::candidate(stream, LSTM_LAYER);
                candidate.enqueue(&input.as_view(), &mut output.as_view_mut(), &phases, stream)
            }
            #[cfg(test)]
            Choice::Mutant(mutant) => {
                super::super::test_support::poison(runtime)?;
                {
                    let _scope =
                        super::super::test_support::candidate(runtime.stream(), LSTM_LAYER);
                    if !mutant.skips() {
                        super::test_support::mutant_lstm(
                            self,
                            runtime,
                            &mut stage.library,
                            input,
                            output,
                            mutant,
                        )?;
                    }
                }
                super::super::test_support::unscoped(runtime, mutant)
            }
        }
    }

    fn lstm_library(
        &self,
        runtime: &CudaRuntime,
        stage: &mut LstmStage,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        #[cfg(test)]
        let _scope = super::super::test_support::library(runtime.stream(), LSTM_LAYER);
        self.lstm
            .forward(runtime, &mut stage.library, input, output)?;
        #[cfg(test)]
        super::super::test_support::perturb(runtime, LSTM_LAYER, output)?;
        Ok(())
    }

    /// Plans the stack candidate when `choice` is `Oxide` and its coverage declares this
    /// batch and mode
    pub(super) fn plan_lstm(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
        choice: Choice,
    ) -> Result<Option<LstmOxide>, CudaError> {
        if !declared(choice, LstmOxide::COVERAGE, LSTM_LAYER, shape.batch, self) {
            return Ok(None);
        }

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
        #[cfg(test)]
        let _scope = super::super::test_support::plan(LSTM_LAYER);
        super::super::dispatch::candidate_plan(
            choice,
            LSTM_LAYER,
            shape.batch,
            LstmOxide::plan(runtime, spec),
        )
    }
}

fn declared(
    choice: Choice,
    coverage: Coverage,
    layer: &str,
    batch: usize,
    network: &Network,
) -> bool {
    choice.is_candidate() && coverage.covers(layer, batch, network.options.math)
}
