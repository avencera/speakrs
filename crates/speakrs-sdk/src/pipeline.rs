//! The binding-facing pipeline: single files, samples, batches, and progress

use std::path::{Path, PathBuf};
use std::time::Instant;

use speakrs::ExecutionMode;
use speakrs::pipeline::{
    OwnedDiarizationPipeline, PipelineBuilder, PipelineConfig, QueueConfig, RuntimeConfig,
};
use speakrs::segment::Segment;

use crate::audio::{SDK_SAMPLE_RATE_HZ, decode_file_to_mono, resample_mono_to_16khz};
use crate::batch::DecodePool;
use crate::cancel::CancelToken;
use crate::error::SdkError;
use crate::models::PreparedModels;
use crate::queue::SdkQueue;

/// Something an operation reports while it runs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent {
    /// A model file is about to be downloaded
    DownloadingModel {
        /// File name within the model repository
        file: String,
    },
    /// An audio file is being decoded
    DecodingAudio,
    /// Decoded audio is being resampled to 16 kHz
    ResamplingAudio,
    /// Inference and clustering are running
    RunningPipeline,
    /// A batch started inference on the file at `index`
    BatchFile {
        /// Position of the file in the batch, from 0
        index: usize,
        /// File id of that file
        file_id: String,
    },
    /// The operation finished successfully
    Completed,
}

/// Receives [`ProgressEvent`]s on the thread that runs the operation
pub type ProgressCallback<'a> = dyn Fn(ProgressEvent) + 'a;

/// Per-call options shared by the diarize methods
#[derive(Clone, Copy, Default)]
pub struct RunOptions<'a> {
    /// Overrides the pipeline's default config for this call
    pub pipeline_config: Option<&'a PipelineConfig>,
    /// Checked between stages, and between files in a batch
    pub cancel_token: Option<&'a CancelToken>,
    /// Receives progress events
    pub progress: Option<&'a ProgressCallback<'a>>,
}

impl RunOptions<'_> {
    fn emit(&self, event: ProgressEvent) {
        if let Some(progress) = self.progress {
            progress(event);
        }
    }

    fn check_cancelled(&self) -> Result<(), SdkError> {
        CancelToken::check(self.cancel_token)
    }

    /// Report a stage, then check the token
    ///
    /// The check follows the event because a binding's callback may cancel, for
    /// example on a Ctrl+C it can only observe while handling an event
    fn begin(&self, event: ProgressEvent) -> Result<(), SdkError> {
        self.emit(event);
        self.check_cancelled()
    }
}

/// Wall-clock time spent in each stage of a single-file call, in milliseconds
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TimingStats {
    /// Decoding the audio file (zero for in-memory samples)
    pub audio_decode_ms: f64,
    /// Resampling to 16 kHz (zero when the input was already 16 kHz)
    pub audio_resample_ms: f64,
    /// Inference, clustering, and reconstruction
    pub pipeline_ms: f64,
    /// The whole call
    pub total_ms: f64,
}

/// Speaker turns for one input
#[derive(Debug, Clone, PartialEq)]
pub struct Diarization {
    /// Identifier used in RTTM output
    pub file_id: String,
    /// Input duration in seconds
    pub duration: f64,
    /// Speaker turns; overlapping speech yields overlapping segments
    pub segments: Vec<Segment>,
    /// Speaker turns with at most one speaker at a time; overlaps go to the
    /// speaker with the highest activation
    pub exclusive_segments: Vec<Segment>,
    /// Mode that produced the result
    pub mode: ExecutionMode,
    /// Model revision, when the models were prepared from a known revision
    pub model_revision: Option<String>,
    /// Stage timings for single-file calls; `None` for batch and queue results
    pub timing: Option<TimingStats>,
}

impl Diarization {
    /// Render RTTM lines, using `file_id` instead of [`Self::file_id`] when given
    pub fn rttm(&self, file_id: Option<&str>, exclusive: bool) -> String {
        let segments = if exclusive {
            &self.exclusive_segments
        } else {
            &self.segments
        };
        speakrs::segment::to_rttm(segments, file_id.unwrap_or(&self.file_id))
    }
}

/// A loaded diarization pipeline for one execution mode
pub struct SdkPipeline {
    pipeline: OwnedDiarizationPipeline,
    mode: ExecutionMode,
    model_revision: Option<String>,
}

impl SdkPipeline {
    /// Load the models in `prepared` for `mode`
    ///
    /// `pipeline_config` becomes the default for every call; without it the mode's
    /// defaults from [`PipelineConfig::for_mode`] apply
    pub fn from_prepared(
        prepared: &PreparedModels,
        mode: ExecutionMode,
        pipeline_config: Option<PipelineConfig>,
        runtime_config: Option<RuntimeConfig>,
    ) -> Result<Self, SdkError> {
        let mut builder = PipelineBuilder::from_dir(&prepared.model_dir, mode)
            .map_err(SdkError::from_load_error)?;
        if let Some(config) = pipeline_config {
            builder = builder.pipeline(config);
        }
        if let Some(config) = runtime_config {
            builder = builder.runtime(config);
        }

        Ok(Self {
            pipeline: builder.build().map_err(SdkError::from_load_error)?,
            mode,
            model_revision: prepared.revision.clone(),
        })
    }

    /// Mode the models were loaded for
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Config used when a call does not pass one
    pub fn pipeline_config(&self) -> PipelineConfig {
        self.pipeline.pipeline_config()
    }

    /// Diarize 16 kHz mono samples
    pub fn diarize_samples(
        &mut self,
        samples: &[f32],
        file_id: &str,
        options: RunOptions<'_>,
    ) -> Result<Diarization, SdkError> {
        let start = Instant::now();
        options.begin(ProgressEvent::RunningPipeline)?;
        let mut diarization = self.run(samples, file_id, &options)?;
        let pipeline_ms = elapsed_ms(start);
        diarization.timing = Some(TimingStats {
            pipeline_ms,
            total_ms: pipeline_ms,
            ..TimingStats::default()
        });
        options.emit(ProgressEvent::Completed);
        Ok(diarization)
    }

    /// Decode, resample, and diarize an audio file
    pub fn diarize_file(
        &mut self,
        path: &Path,
        file_id: &str,
        options: RunOptions<'_>,
    ) -> Result<Diarization, SdkError> {
        let start = Instant::now();
        options.begin(ProgressEvent::DecodingAudio)?;
        let audio = decode_file_to_mono(path).map_err(SdkError::from_audio_error)?;
        let audio_decode_ms = elapsed_ms(start);

        // timed only when it runs, so 16 kHz input reports exactly zero as documented
        let (samples, audio_resample_ms) = if audio.sample_rate_hz == SDK_SAMPLE_RATE_HZ {
            (audio.samples, 0.0)
        } else {
            options.begin(ProgressEvent::ResamplingAudio)?;
            let resample_start = Instant::now();
            let samples = resample_mono_to_16khz(audio.samples, audio.sample_rate_hz)
                .map_err(SdkError::from_audio_error)?;
            (samples, elapsed_ms(resample_start))
        };

        options.begin(ProgressEvent::RunningPipeline)?;
        let pipeline_start = Instant::now();
        let mut diarization = self.run(&samples, file_id, &options)?;
        diarization.timing = Some(TimingStats {
            audio_decode_ms,
            audio_resample_ms,
            pipeline_ms: elapsed_ms(pipeline_start),
            total_ms: elapsed_ms(start),
        });
        options.emit(ProgressEvent::Completed);
        Ok(diarization)
    }

    /// Diarize many audio files, overlapping decoding, inference, and clustering
    ///
    /// Up to four workers decode files ahead of the one in inference, and one file's
    /// clustering runs while the next file is in inference. Results keep input order.
    /// The first failure stops the batch and no partial results are returned
    pub fn diarize_files(
        &mut self,
        files: Vec<(String, PathBuf)>,
        options: RunOptions<'_>,
    ) -> Result<Vec<Diarization>, SdkError> {
        let config = options
            .pipeline_config
            .cloned()
            .unwrap_or_else(|| self.pipeline.pipeline_config());
        let inputs = DecodePool::new(files, options.cancel_token.cloned(), options.progress)?;
        let outputs = self
            .pipeline
            .run_batch_stream_with_config(inputs, &config)
            .map_err(|error| match error {
                speakrs::pipeline::BatchStreamError::Input(error) => error,
                speakrs::pipeline::BatchStreamError::Pipeline(error) => {
                    SdkError::from_pipeline_error(error)
                }
            })?;

        options.emit(ProgressEvent::Completed);
        Ok(outputs
            .into_iter()
            .map(|output| self.diarization(output.result, output.file_id, output.audio_secs))
            .collect())
    }

    /// Move the pipeline onto a background worker fed by a bounded queue
    ///
    /// Takes a validated [`QueueConfig`] so a bad capacity is rejected before the
    /// pipeline is consumed; only a failure to start the worker thread loses it
    pub fn into_queue(
        self,
        pipeline_config: Option<PipelineConfig>,
        queue_config: QueueConfig,
    ) -> Result<SdkQueue, SdkError> {
        let config = pipeline_config.unwrap_or_else(|| self.pipeline.pipeline_config());
        let (sender, receiver) = self
            .pipeline
            .into_queued_with_configs(config, queue_config)
            .map_err(SdkQueue::map_error)?;

        Ok(SdkQueue::new(
            sender,
            receiver,
            self.mode,
            self.model_revision,
        ))
    }

    fn run(
        &mut self,
        samples: &[f32],
        file_id: &str,
        options: &RunOptions<'_>,
    ) -> Result<Diarization, SdkError> {
        let config = match options.pipeline_config {
            Some(config) => config.clone(),
            None => self.pipeline.pipeline_config(),
        };
        let result = self
            .pipeline
            .run_with_config(samples, file_id, &config)
            .map_err(SdkError::from_pipeline_error)?;
        options.check_cancelled()?;

        Ok(self.diarization(
            result,
            file_id.to_string(),
            samples.len() as f64 / f64::from(SDK_SAMPLE_RATE_HZ),
        ))
    }

    fn diarization(
        &self,
        result: speakrs::pipeline::DiarizationResult,
        file_id: String,
        duration: f64,
    ) -> Diarization {
        to_diarization(
            result,
            file_id,
            duration,
            self.mode,
            self.model_revision.clone(),
            None,
        )
    }
}

pub(crate) fn to_diarization(
    result: speakrs::pipeline::DiarizationResult,
    file_id: String,
    duration: f64,
    mode: ExecutionMode,
    model_revision: Option<String>,
    timing: Option<TimingStats>,
) -> Diarization {
    Diarization {
        file_id,
        duration,
        segments: result.segments,
        exclusive_segments: result.exclusive_segments,
        mode,
        model_revision,
        timing,
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}
