//! Native CUDA fbank front end: what `wespeaker-fbank.onnx` and
//! `wespeaker-fbank-b32.onnx` compute, batched on the GPU
//!
//! A batch of 10 s waveform rows becomes `[rows, FBANK_FRAMES, FBANK_MEL_BINS]` log-mel
//! features with temporal mean normalization, the `fbank` input layout of the
//! multi-mask embedding tail. A selected FFT/mel candidate writes the energies
//! directly, then uses the same log/CMN consumer. The Library route runs these steps:
//!
//! 1. `fbank_frame_window` scales to 16-bit range, frames (400 samples, hop 160),
//!    removes each frame's mean, pre-emphasizes (0.97) and applies the Hamming window
//! 2. one SGEMM against a `[400, 512]` cos/sin basis gives every real and imaginary
//!    part of the 512-point one-sided DFT; the zero padding from 400 to 512 samples
//!    contributes nothing, so the GEMM skips it
//! 3. the mel projection, either [`MelProjection::Sparse`] (one kernel that squares
//!    and sums each filter's run of bins) or [`MelProjection::Gemm`] (a power kernel and
//!    an SGEMM against the dense `[256, 80]` filters)
//! 4. `fbank_log_cmn` floors at f32 epsilon, takes the log and subtracts each mel bin's
//!    mean over the row's frames
//!
//! Both GEMMs run in the [`CudaMath`] the caller picks, which should be the embedding
//! stage's

mod constants;

use cudarc::driver::{
    CudaEvent, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};

use super::candidate::{FbankCandidate, FbankOxide, FbankSpec, Phases};
use super::error::check_len;
use super::implementation::{Selected, plan_selection};
use super::{CudaError, CudaMath, CudaRuntime, KernelModule, PtxTier, Sgemm};

use constants::{
    ENERGY_FLOOR, FBANK_FRAME_LENGTH, FBANK_FRAME_SHIFT, MEL_GEMM_BINS, PREEMPHASIS,
    SPECTRUM_COLUMNS, WAVEFORM_SCALE,
};
pub use constants::{FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankConstants};

const FBANK_FRAME_WINDOW: &str = "fbank_frame_window";
const FBANK_POWER: &str = "fbank_power";
const FBANK_MEL_SPARSE: &str = "fbank_mel_sparse";
const FBANK_LOG_CMN: &str = "fbank_log_cmn";

/// Kernel entries loaded by this host plan
#[cfg(test)]
pub(crate) const REQUIRED_KERNELS: [&str; 4] = [
    FBANK_FRAME_WINDOW,
    FBANK_POWER,
    FBANK_MEL_SPARSE,
    FBANK_LOG_CMN,
];

/// Threads per block of `fbank_frame_window`, which handles one frame per warp
const FRAME_WINDOW_THREADS: u32 = 256;
/// Frames per block of `fbank_frame_window`
const FRAMES_PER_BLOCK: usize = FRAME_WINDOW_THREADS as usize / 32;
/// Threads per block of `fbank_log_cmn`
const LOG_CMN_THREADS: u32 = 256;
/// Mel bins per block of `fbank_log_cmn`
const LOG_CMN_MEL_TILE: usize = 16;

// the CMN kernel tiles the mel axis without a remainder
const _: () = assert!(FBANK_MEL_BINS.is_multiple_of(LOG_CMN_MEL_TILE));
// frames read past the end of a row would mix two waveforms
const _: () =
    assert!((FBANK_FRAMES - 1) * FBANK_FRAME_SHIFT + FBANK_FRAME_LENGTH <= FBANK_WINDOW_SAMPLES);

/// How the power spectrum is projected onto the mel filters
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MelProjection {
    /// One kernel squares each filter's contiguous run of DFT bins and sums it with
    /// the filter weights; it skips the zero weights and the power buffer
    #[default]
    Sparse,
    /// A power-spectrum kernel, then an SGEMM against the dense `[256, 80]` filters
    /// in the front end's [`CudaMath`]
    ///
    /// The measured alternative to the default; only the parity tests and the benchmark
    /// select it
    // the dense control is not used by production or driver-only tests
    #[cfg_attr(not(all(test, feature = "_cuda-libraries")), allow(dead_code))]
    Gemm,
}

/// The fbank front end on one [`CudaRuntime`]: loaded kernels and the fixed tensors on
/// the device
///
/// Create it once per runtime and reuse it; per-batch memory lives in
/// [`FbankBuffers`]
#[derive(Debug)]
pub struct CudaFbank {
    plans: Vec<Option<FbankOxide>>,
    tier: PtxTier,
    math: CudaMath,
    projection: MelProjection,
    frame_window: CudaFunction,
    power: CudaFunction,
    mel_sparse: CudaFunction,
    log_cmn: CudaFunction,
    window: CudaSlice<f32>,
    dft_basis: CudaSlice<f32>,
    /// `[256, 80]`: the ONNX filters without their all-zero Nyquist row
    mel_dense: CudaSlice<f32>,
    mel_first: CudaSlice<u32>,
    mel_count: CudaSlice<u32>,
    mel_weights: CudaSlice<f32>,
    mel_width: u32,
}

/// The DFT GEMM, a Library boundary at every filterbank batch
const DFT: super::implementation::BoundaryId =
    super::implementation::BoundaryId::named("fbank.dft");

impl CudaFbank {
    /// Loads the kernels and uploads the window, DFT basis and mel filters, with the
    /// default [`MelProjection`]
    pub fn new(runtime: &CudaRuntime, math: CudaMath) -> Result<Self, CudaError> {
        Self::with_projection(runtime, math, MelProjection::default())
    }

    /// Like [`Self::new`], with an explicit mel projection
    pub fn with_projection(
        runtime: &CudaRuntime,
        math: CudaMath,
        projection: MelProjection,
    ) -> Result<Self, CudaError> {
        let kernels = runtime.load_kernels(KernelModule::Fbank)?;
        let mut plans = Vec::new();
        for batch in 1..=32 {
            let selected = plan_selection(
                runtime,
                DFT,
                batch,
                math,
                #[cfg(all(test, feature = "_cuda-libraries"))]
                None,
            )?;
            let plan = if let Selected::Oxide(token) = selected {
                let spec = FbankSpec::new(batch, math).map_err(|error| CudaError::Unsupported {
                    context: "fbank shape",
                    reason: error.to_string(),
                })?;
                token.fbank(runtime, spec)?
            } else {
                None
            };
            if plan.is_none() {
                super::implementation::LibraryNeed::new(
                    DFT,
                    batch,
                    math,
                    super::implementation::AreaTarget {
                        tier: kernels.tier(),
                        device: runtime.compute_capability(),
                    },
                    super::CudaLibrary::Cublas,
                )
                .prepare(runtime)?;
            }
            plans.push(plan);
        }
        let constants = FbankConstants::new();
        let table = constants.mel_table();
        let stream = runtime.stream();
        let mel_dense = &constants.mel()[..MEL_GEMM_BINS * FBANK_MEL_BINS];

        Ok(Self {
            plans,
            tier: kernels.tier(),
            math,
            projection,
            frame_window: kernels.function(FBANK_FRAME_WINDOW)?,
            power: kernels.function(FBANK_POWER)?,
            mel_sparse: kernels.function(FBANK_MEL_SPARSE)?,
            log_cmn: kernels.function(FBANK_LOG_CMN)?,
            window: stream.clone_htod(constants.window())?,
            dft_basis: stream.clone_htod(constants.dft_basis())?,
            mel_dense: stream.clone_htod(mel_dense)?,
            mel_first: stream.clone_htod(&table.first)?,
            mel_count: stream.clone_htod(&table.count)?,
            mel_weights: stream.clone_htod(&table.weights)?,
            mel_width: launch_u32("fbank mel width", table.width)?,
        })
    }

    /// The PTX tier the kernels were loaded from
    pub fn tier(&self) -> PtxTier {
        self.tier
    }

    /// Allocates the device and pinned host memory for batches of up to `capacity`
    /// rows; reuse it for every batch
    pub fn buffers(
        &self,
        runtime: &CudaRuntime,
        capacity: usize,
    ) -> Result<FbankBuffers, CudaError> {
        FbankBuffers::new(runtime, capacity, self.projection)
    }

    /// Uploads host waveforms through `buffers` and computes their features; see
    /// [`FbankBuffers::upload`] and [`Self::compute_uploaded`]
    pub fn compute_host<'a>(
        &self,
        runtime: &CudaRuntime,
        waveforms: &[&[f32]],
        buffers: &'a mut FbankBuffers,
    ) -> Result<CudaView<'a, f32>, CudaError> {
        buffers.upload(runtime, waveforms)?;
        self.compute_uploaded(runtime, buffers)
    }

    /// Uploads windows of one stretch of audio through `buffers` and computes their
    /// features; see [`FbankBuffers::upload_span`] and [`Self::compute_uploaded`]
    pub fn compute_span<'a>(
        &self,
        runtime: &CudaRuntime,
        span: &[f32],
        starts: &[usize],
        buffers: &'a mut FbankBuffers,
    ) -> Result<CudaView<'a, f32>, CudaError> {
        buffers.upload_span(runtime, span, starts)?;
        self.compute_uploaded(runtime, buffers)
    }

    /// Computes features for the rows of the last [`FbankBuffers::upload`]
    ///
    /// Returns `[rows, FBANK_FRAMES, FBANK_MEL_BINS]` on the device. The work is queued
    /// on the runtime's stream; the view is valid until the next batch overwrites it
    pub fn compute_uploaded<'a>(
        &self,
        runtime: &CudaRuntime,
        buffers: &'a mut FbankBuffers,
    ) -> Result<CudaView<'a, f32>, CudaError> {
        let rows = buffers.uploaded_rows;
        let waveform = buffers.waveform.slice(..rows * FBANK_WINDOW_SAMPLES);
        self.run(runtime, &waveform, rows, &mut buffers.work)?;
        Ok(buffers
            .work
            .features
            .slice(..rows * FBANK_FRAMES * FBANK_MEL_BINS))
    }

    #[cfg(all(test, feature = "_cuda-libraries"))]
    /// Computes features for waveform rows already on the device
    ///
    /// `waveform` is `[rows, FBANK_WINDOW_SAMPLES]` with shorter audio zero padded, at
    /// most `buffers.capacity()` rows, on the runtime's stream. Returns
    /// `[rows, FBANK_FRAMES, FBANK_MEL_BINS]`, valid until the next batch overwrites it
    pub fn compute_device<'a>(
        &self,
        runtime: &CudaRuntime,
        waveform: &CudaView<'_, f32>,
        buffers: &'a mut FbankBuffers,
    ) -> Result<CudaView<'a, f32>, CudaError> {
        let rows = waveform.len() / FBANK_WINDOW_SAMPLES;
        check_len(
            "fbank waveform rows",
            rows * FBANK_WINDOW_SAMPLES,
            waveform.len(),
        )?;
        buffers.check_rows(rows)?;

        self.run(runtime, waveform, rows, &mut buffers.work)?;
        Ok(buffers
            .work
            .features
            .slice(..rows * FBANK_FRAMES * FBANK_MEL_BINS))
    }

    fn run(
        &self,
        runtime: &CudaRuntime,
        waveform: &CudaView<'_, f32>,
        rows: usize,
        work: &mut FbankWork,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }

        runtime.record_boundary(DFT, rows, self.math, || {
            let frames = rows * FBANK_FRAMES;
            self.produce(
                runtime,
                waveform,
                rows,
                &mut work.producer,
                &mut work.energies.slice_mut(..frames * FBANK_MEL_BINS),
            )?;
            self.log_cmn(
                runtime,
                rows,
                &work.energies.slice(..frames * FBANK_MEL_BINS),
                &mut work.features.slice_mut(..frames * FBANK_MEL_BINS),
            )
        })
    }

    fn produce(
        &self,
        runtime: &CudaRuntime,
        waveform: &CudaView<'_, f32>,
        rows: usize,
        work: &mut FbankProducerWork,
        energies: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        if let Some(Some(plan)) = rows.checked_sub(1).and_then(|index| self.plans.get(index)) {
            return plan.enqueue(waveform, energies, &Phases::new(), runtime);
        }
        if super::driver_only() {
            return Err(CudaError::MissingKernel {
                boundary: DFT.name().to_owned(),
                batch: rows,
                math: self.math,
            });
        }
        let frames = rows * FBANK_FRAMES;
        self.frame_window(
            runtime,
            waveform,
            &mut work.frames.slice_mut(..frames * FBANK_FRAME_LENGTH),
        )?;

        let dft = Sgemm {
            math: self.math,
            ..Sgemm::new(frames, SPECTRUM_COLUMNS, FBANK_FRAME_LENGTH)
        };
        runtime.sgemm(
            dft,
            &work.frames.slice(..frames * FBANK_FRAME_LENGTH),
            &self.dft_basis,
            &mut work.spectrum.slice_mut(..frames * SPECTRUM_COLUMNS),
        )?;

        let spectrum = work.spectrum.slice(..frames * SPECTRUM_COLUMNS);
        match self.projection {
            MelProjection::Sparse => self.mel_sparse(runtime, &spectrum, energies)?,
            MelProjection::Gemm => {
                let power = work.power.as_mut().ok_or(CudaError::BufferLength {
                    context: "fbank power buffer",
                    expected: frames * MEL_GEMM_BINS,
                    actual: 0,
                })?;
                let mut power = power.slice_mut(..frames * MEL_GEMM_BINS);
                self.power(runtime, &spectrum, &mut power)?;

                let mel = Sgemm {
                    math: self.math,
                    ..Sgemm::new(frames, FBANK_MEL_BINS, MEL_GEMM_BINS)
                };
                runtime.sgemm(mel, &power, &self.mel_dense, energies)?;
            }
        }

        Ok(())
    }

    fn frame_window(
        &self,
        runtime: &CudaRuntime,
        waveform: &CudaView<'_, f32>,
        frames: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let frame_count = frames.len() / FBANK_FRAME_LENGTH;
        let blocks = launch_u32("fbank frame blocks", frame_count.div_ceil(FRAMES_PER_BLOCK))?;
        let samples_per_row = launch_u32("fbank samples per row", FBANK_WINDOW_SAMPLES)?;
        let frames_per_row = launch_u32("fbank frames per row", FBANK_FRAMES)?;
        let (scale, preemphasis) = (WAVEFORM_SCALE, PREEMPHASIS);
        let waveform_len = waveform.len() as u64;
        let window_len = self.window.len() as u64;
        let frames_len = frames.len() as u64;

        let mut launch = runtime.stream().launch_builder(&self.frame_window);
        launch
            .arg(&samples_per_row)
            .arg(&frames_per_row)
            .arg(&scale)
            .arg(&preemphasis)
            .arg(waveform)
            .arg(&waveform_len)
            .arg(&self.window)
            .arg(&window_len)
            .arg(frames)
            .arg(&frames_len);

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (FRAME_WINDOW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match the PTX signature of `fbank_frame_window` (two
        // u32, two f32, then pointer and length for waveform, window and frames), every
        // length is the real buffer length, and the grid gives each frame one warp
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    fn power(
        &self,
        runtime: &CudaRuntime,
        spectrum: &CudaView<'_, f32>,
        power: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let threads = launch_u32("fbank power elements", power.len())?;
        let bins = launch_u32("fbank power bins", MEL_GEMM_BINS)?;
        let spectrum_len = spectrum.len() as u64;
        let power_len = power.len() as u64;

        let mut launch = runtime.stream().launch_builder(&self.power);
        launch
            .arg(&bins)
            .arg(spectrum)
            .arg(&spectrum_len)
            .arg(power)
            .arg(&power_len);

        // SAFETY: the arguments match the PTX signature of `fbank_power` (u32, then
        // pointer and length for spectrum and power), the lengths are the real buffer
        // lengths, and the 1-D grid has one thread per power element
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;
        Ok(())
    }

    fn mel_sparse(
        &self,
        runtime: &CudaRuntime,
        spectrum: &CudaView<'_, f32>,
        energies: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let threads = launch_u32("fbank mel elements", energies.len())?;
        let spectrum_len = spectrum.len() as u64;
        let first_len = self.mel_first.len() as u64;
        let count_len = self.mel_count.len() as u64;
        let weights_len = self.mel_weights.len() as u64;
        let energies_len = energies.len() as u64;

        let mut launch = runtime.stream().launch_builder(&self.mel_sparse);
        launch
            .arg(&self.mel_width)
            .arg(spectrum)
            .arg(&spectrum_len)
            .arg(&self.mel_first)
            .arg(&first_len)
            .arg(&self.mel_count)
            .arg(&count_len)
            .arg(&self.mel_weights)
            .arg(&weights_len)
            .arg(energies)
            .arg(&energies_len);

        // SAFETY: the arguments match the PTX signature of `fbank_mel_sparse` (u32,
        // then pointer and length for spectrum, first, count, weights and energies),
        // the lengths are the real buffer lengths, and the 1-D grid has one thread per
        // energy
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;
        Ok(())
    }

    fn log_cmn(
        &self,
        runtime: &CudaRuntime,
        rows: usize,
        energies: &CudaView<'_, f32>,
        features: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        check_len("fbank features", energies.len(), features.len())?;
        let blocks = launch_u32(
            "fbank cmn blocks",
            rows * (FBANK_MEL_BINS / LOG_CMN_MEL_TILE),
        )?;
        let frames_per_row = launch_u32("fbank frames per row", FBANK_FRAMES)?;
        let mel_bins = launch_u32("fbank mel bins", FBANK_MEL_BINS)?;
        let floor = ENERGY_FLOOR;
        let energies_len = energies.len() as u64;
        let features_len = features.len() as u64;

        let mut launch = runtime.stream().launch_builder(&self.log_cmn);
        launch
            .arg(&frames_per_row)
            .arg(&mel_bins)
            .arg(&floor)
            .arg(energies)
            .arg(&energies_len)
            .arg(features)
            .arg(&features_len);

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (LOG_CMN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match the PTX signature of `fbank_log_cmn` (two u32,
        // f32, then pointer and length for energies and features), energies and
        // features have the same length (checked above), and the grid has one
        // 256-thread block per row and 16 mel bins as the kernel expects
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

/// Per-batch memory of [`CudaFbank`]: a pinned host staging buffer, the device
/// waveform and every intermediate, sized for up to [`Self::capacity`] rows
#[derive(Debug)]
pub struct FbankBuffers {
    capacity: usize,
    uploaded_rows: usize,
    /// write-combined pinned memory, so the upload is an asynchronous DMA
    staging: PinnedHostSlice<f32>,
    /// recorded after each upload; the staging buffer must not change before it fires
    staged: CudaEvent,
    /// the contiguous audio of the last [`Self::upload_span`], unfolded into `waveform`
    span: CudaSlice<f32>,
    waveform: CudaSlice<f32>,
    work: FbankWork,
}

/// Device intermediates and the output of one batch
#[derive(Debug)]
struct FbankWork {
    producer: FbankProducerWork,
    /// `[rows · FBANK_FRAMES, FBANK_MEL_BINS]`
    energies: CudaSlice<f32>,
    /// `[rows, FBANK_FRAMES, FBANK_MEL_BINS]`
    features: CudaSlice<f32>,
}

/// Scratch owned by the frame/window, DFT and mel producer
#[derive(Debug)]
struct FbankProducerWork {
    /// `[rows · FBANK_FRAMES, FBANK_FRAME_LENGTH]`
    frames: CudaSlice<f32>,
    /// `[rows · FBANK_FRAMES, SPECTRUM_COLUMNS]`
    spectrum: CudaSlice<f32>,
    /// `[rows · FBANK_FRAMES, 256]`, only for [`MelProjection::Gemm`]
    power: Option<CudaSlice<f32>>,
}

impl FbankBuffers {
    fn new(
        runtime: &CudaRuntime,
        capacity: usize,
        projection: MelProjection,
    ) -> Result<Self, CudaError> {
        let stream = runtime.stream();
        let frames = checked_mul("fbank frames", capacity, FBANK_FRAMES)?;
        let alloc = |columns: usize| -> Result<CudaSlice<f32>, CudaError> {
            Ok(stream.alloc_zeros(checked_mul("fbank buffer", frames, columns)?)?)
        };
        let samples = checked_mul("fbank waveform", capacity, FBANK_WINDOW_SAMPLES)?;

        // SAFETY: the staging memory is uninitialized; it is zero filled right below,
        // before anything reads it
        let mut staging = unsafe { runtime.context().alloc_pinned::<f32>(samples) }?;
        staging.as_mut_slice()?.fill(0.0);

        let power = match projection {
            MelProjection::Sparse => None,
            MelProjection::Gemm => Some(alloc(MEL_GEMM_BINS)?),
        };

        Ok(Self {
            capacity,
            uploaded_rows: 0,
            staging,
            staged: runtime.context().new_event(None)?,
            span: stream.alloc_zeros(samples)?,
            waveform: stream.alloc_zeros(samples)?,
            work: FbankWork {
                producer: FbankProducerWork {
                    frames: alloc(FBANK_FRAME_LENGTH)?,
                    spectrum: alloc(SPECTRUM_COLUMNS)?,
                    power,
                },
                energies: alloc(FBANK_MEL_BINS)?,
                features: alloc(FBANK_MEL_BINS)?,
            },
        })
    }

    /// Stages host waveforms and queues their upload on the runtime's stream
    ///
    /// Each waveform fills one row of [`FBANK_WINDOW_SAMPLES`] samples in [-1, 1): a
    /// shorter one is zero padded and a longer one truncated, as the ONNX path does.
    /// Waits for the previous upload to leave the staging buffer, not for the
    /// computation
    pub fn upload(&mut self, runtime: &CudaRuntime, waveforms: &[&[f32]]) -> Result<(), CudaError> {
        let rows = waveforms.len();
        self.check_rows(rows)?;
        // a failed upload leaves no rows to compute
        self.uploaded_rows = 0;
        if rows == 0 {
            return Ok(());
        }

        self.staged.synchronize()?;

        let samples = rows * FBANK_WINDOW_SAMPLES;
        let staging = &mut self.staging.as_mut_slice()?[..samples];
        let (staged_rows, _) = staging.as_chunks_mut::<FBANK_WINDOW_SAMPLES>();
        for (row, audio) in staged_rows.iter_mut().zip(waveforms) {
            let copied = audio.len().min(FBANK_WINDOW_SAMPLES);
            row[..copied].copy_from_slice(&audio[..copied]);
            row[copied..].fill(0.0);
        }

        let stream = runtime.stream();
        let (device, _record) = self.waveform.device_ptr_mut(stream);
        // SAFETY: `staging` is pinned memory owned by `self` and holds `samples`
        // values, which fit in `waveform`; the `staged` event recorded below keeps the
        // next upload, and `Drop`, from touching it until the copy has finished
        unsafe { cudarc::driver::result::memcpy_htod_async(device, staging, stream.cu_stream()) }?;
        self.staged.record(stream)?;

        self.uploaded_rows = rows;
        Ok(())
    }

    /// Stages one contiguous stretch of audio, queues its upload, and unfolds the
    /// rows `span[start..start + FBANK_WINDOW_SAMPLES]` into the device waveform
    ///
    /// Each row matches what [`Self::upload`] makes of the same window: truncated to
    /// [`FBANK_WINDOW_SAMPLES`] and zero padded past the end of `span`. Overlapping
    /// windows share their samples, so with a 1 s step over 10 s windows the host copy
    /// and the transfer are about an eighth of a per-window upload
    pub fn upload_span(
        &mut self,
        runtime: &CudaRuntime,
        span: &[f32],
        starts: &[usize],
    ) -> Result<(), CudaError> {
        let rows = starts.len();
        self.check_rows(rows)?;
        // a failed upload leaves no rows to compute
        self.uploaded_rows = 0;
        if rows == 0 {
            return Ok(());
        }
        if span.len() > self.span.len() {
            return Err(CudaError::BufferLength {
                context: "fbank span samples",
                expected: self.span.len(),
                actual: span.len(),
            });
        }

        self.staged.synchronize()?;

        let stream = runtime.stream();
        if !span.is_empty() {
            let staging = &mut self.staging.as_mut_slice()?[..span.len()];
            staging.copy_from_slice(span);
            let (device, _record) = self.span.device_ptr_mut(stream);
            // SAFETY: `staging` is pinned memory owned by `self` and holds `span.len()`
            // values, which fit in `self.span`; the `staged` event recorded below keeps
            // the next upload, and `Drop`, from touching it until the copy has finished
            unsafe {
                cudarc::driver::result::memcpy_htod_async(device, staging, stream.cu_stream())
            }?;
        }
        self.staged.record(stream)?;

        for (row, &start) in starts.iter().enumerate() {
            let copied = span.len().saturating_sub(start).min(FBANK_WINDOW_SAMPLES);
            let mut target = self
                .waveform
                .slice_mut(row * FBANK_WINDOW_SAMPLES..(row + 1) * FBANK_WINDOW_SAMPLES);
            if copied > 0 {
                let source = self.span.slice(start..start + copied);
                stream.memcpy_dtod(&source, &mut target.slice_mut(..copied))?;
            }
            if copied < FBANK_WINDOW_SAMPLES {
                stream.memset_zeros(&mut target.slice_mut(copied..))?;
            }
        }

        self.uploaded_rows = rows;
        Ok(())
    }

    fn check_rows(&self, rows: usize) -> Result<(), CudaError> {
        if rows > self.capacity {
            return Err(CudaError::BufferLength {
                context: "fbank batch rows",
                expected: self.capacity,
                actual: rows,
            });
        }

        Ok(())
    }
}

impl Drop for FbankBuffers {
    fn drop(&mut self) {
        // an upload still reading the pinned staging memory must finish before the
        // memory is freed; errors here mean the context is already gone
        let _ = self.staged.synchronize();
    }
}

fn launch_u32(context: &'static str, value: usize) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::DimensionOverflow { context, value })
}

fn checked_mul(context: &'static str, left: usize, right: usize) -> Result<usize, CudaError> {
    left.checked_mul(right).ok_or(CudaError::DimensionOverflow {
        context,
        value: left,
    })
}
