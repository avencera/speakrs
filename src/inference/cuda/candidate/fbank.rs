//! The cuda-oxide filterbank energy producer: framing, mean removal, pre-emphasis,
//! window, 512-point real DFT and sparse mel projection in one kernel, writing only
//! the `[batch, 998, 80]` energies
//!
//! The locked consumer then runs the always-on `fbank_log_cmn` pass, unchanged. Both
//! math modes run the same FP32 kernel; TF32 qualification compares it with the
//! Library's TF32 DFT GEMM

use cudarc::driver::{CudaFunction, CudaSlice, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use super::{
    Batches, Coverage, CoverageEntry, FBANK_BATCHES, FBANK_FRAMES, FBANK_MEL_BINS,
    FBANK_WINDOW_SAMPLES, FbankCandidate, FbankConstants, FbankPin, FbankSpec, FiniteContract,
    GeometryError, InfinityContract, Maths, NanContract, Phases, PlanError, SignedZeroContract,
    SpecialValues,
};
use crate::inference::cuda::error::check_len;
use crate::inference::cuda::{
    CudaError, CudaMath, CudaRuntime, KernelModule, LoadedKernels, PtxTier,
};

const FBANKDFT_FFT_MEL_ACCURATE: &str = "fbankdft_fft_mel_accurate";

/// Kernel entries loaded by this host plan
pub(crate) const REQUIRED_KERNELS: [&str; 1] = [FBANKDFT_FFT_MEL_ACCURATE];

/// The boundary name
const LAYER: &str = "fbank.dft";
/// The error context of every check
const CONTEXT: &str = "fbank.dft producer";

/// Threads of one block; one warp per frame
const THREADS: u32 = 256;
/// Frames of one block; `THREADS / 32`
const FRAMES_PER_BLOCK: usize = THREADS as usize / 32;
/// Samples of one analysis frame; the kernel's staged window
const FRAME_LENGTH: usize = 400;
/// Real DFT size; the twiddle table holds one cos/sin pair per bin
const FFT_SIZE: usize = 512;
/// Stride of the kernel's staged transposed mel weights, which reads exactly this
/// many weights of every filter
const MEL_WIDTH: usize = 16;
/// Lowest and highest bins the kernel's real post-pass writes into staged power; DC
/// and Nyquist carry no mel weight
const POWER_BINS: std::ops::RangeInclusive<u32> = 1..=255;

/// Static shared memory of one block: staged waveform, window, cos and sin, filter
/// starts and counts, transposed weights, and eight 257-word power rows
const SHARED_BYTES: usize = 4
    * (1536 + FRAME_LENGTH + 2 * FFT_SIZE + 2 * FBANK_MEL_BINS + FBANK_MEL_BINS * MEL_WIDTH)
    + 4 * FRAMES_PER_BLOCK * 257;
// every sm_75 or newer device grants 48 KiB of static shared memory per block, so the
// pin has no device-dependent choice and planning reads no device attribute
const _: () = assert!(SHARED_BYTES <= 48 * 1024);

const SAMPLES: u32 = FBANK_WINDOW_SAMPLES as u32;
const FRAMES: u32 = FBANK_FRAMES as u32;
const _: () = assert!(SAMPLES as usize == FBANK_WINDOW_SAMPLES && FRAMES as usize == FBANK_FRAMES);

/// The launch shape of one pinned plan, derived and checked before any launch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Launch {
    batch: u32,
}

impl Launch {
    /// The single shape `pin` names for `spec`'s batch
    pub(super) fn new(spec: FbankSpec, pin: FbankPin) -> Result<Self, PlanError> {
        let FbankPin::FftMelAccurate = pin;
        if !FBANK_BATCHES.contains(&spec.batch()) {
            return Err(invalid(format!(
                "batch {} is outside the pinned 1..=32 grid",
                spec.batch()
            )));
        }

        // the batch domain ends at 32, so the grid row count fits
        Ok(Self {
            batch: spec.batch() as u32,
        })
    }

    /// One block per eight frames of a row, one grid row per waveform row
    pub(super) fn config(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (FRAMES.div_ceil(FRAMES_PER_BLOCK as u32), self.batch, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// `[batch, 160000]` input samples
    pub(super) fn waveform_len(self) -> usize {
        self.batch as usize * FBANK_WINDOW_SAMPLES
    }

    /// `[batch, 998, 80]` output energies
    pub(super) fn energies_len(self) -> usize {
        self.batch as usize * FBANK_FRAMES * FBANK_MEL_BINS
    }

    /// Refuse a call whose buffers are not this plan's shape
    pub(super) fn check(self, waveform: usize, energies: usize) -> Result<(), CudaError> {
        check_len(CONTEXT, self.waveform_len(), waveform)?;
        check_len(CONTEXT, self.energies_len(), energies)
    }
}

/// The fixed host tables uploaded once per plan, in the layout the kernel stages
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Tables {
    /// The Library path's Hamming window
    pub(super) window: Vec<f32>,
    /// `[cos, sin]` of `2πk/512` for `k` in 0..512, rounded once from f64
    pub(super) twiddle: Vec<f32>,
    /// First bin of each filter's run
    pub(super) first: Vec<u32>,
    /// Bins in each filter's run
    pub(super) count: Vec<u32>,
    /// `[80, 16]` filter runs, zero past each count
    pub(super) weights: Vec<f32>,
}

impl Tables {
    /// The Library path's window and mel filter runs, plus the twiddle table
    pub(super) fn new(constants: &FbankConstants) -> Result<Self, PlanError> {
        let mel = constants.mel_table();
        Self::checked(
            constants.window().to_vec(),
            mel.first.clone(),
            mel.count.clone(),
            mel.width,
            mel.weights.clone(),
        )
    }

    /// Validate every table invariant the kernel relies on without bounds checks
    pub(super) fn checked(
        window: Vec<f32>,
        first: Vec<u32>,
        count: Vec<u32>,
        width: usize,
        weights: Vec<f32>,
    ) -> Result<Self, PlanError> {
        if window.len() != FRAME_LENGTH {
            return Err(invalid(format!(
                "window has {} samples, not {FRAME_LENGTH}",
                window.len()
            )));
        }
        if width != MEL_WIDTH {
            return Err(invalid(format!(
                "mel run stride {width} is not the staged {MEL_WIDTH}"
            )));
        }
        if first.len() != FBANK_MEL_BINS
            || count.len() != FBANK_MEL_BINS
            || weights.len() != FBANK_MEL_BINS * MEL_WIDTH
        {
            return Err(invalid(format!(
                "mel tables have {}, {} and {} entries, not {FBANK_MEL_BINS}, {FBANK_MEL_BINS} and {}",
                first.len(),
                count.len(),
                weights.len(),
                FBANK_MEL_BINS * MEL_WIDTH
            )));
        }
        for (filter, (&start, &bins)) in first.iter().zip(&count).enumerate() {
            let last = start.checked_add(bins).and_then(|end| end.checked_sub(1));
            let inside = last.is_some_and(|last| {
                (1..=MEL_WIDTH).contains(&(bins as usize))
                    && POWER_BINS.contains(&start)
                    && POWER_BINS.contains(&last)
            });
            if !inside {
                return Err(invalid(format!(
                    "mel filter {filter} covers {bins} bins from {start}, not 1 to {MEL_WIDTH} bins within {POWER_BINS:?}"
                )));
            }
        }

        Ok(Self {
            window,
            twiddle: twiddles(),
            first,
            count,
            weights,
        })
    }
}

/// `cos` and `sin` of `2πk/512`, interleaved
fn twiddles() -> Vec<f32> {
    (0..FFT_SIZE)
        .flat_map(|bin| {
            let angle = std::f64::consts::TAU * bin as f64 / FFT_SIZE as f64;
            [angle.cos() as f32, angle.sin() as f32]
        })
        .collect()
}

fn invalid(reason: String) -> PlanError {
    PlanError::Geometry(GeometryError::Invalid {
        context: CONTEXT,
        reason,
    })
}

/// The producer kernel, its uploaded tables and the launch shape of one batch size
#[derive(Debug)]
pub(crate) struct Oxide {
    produce: CudaFunction,
    launch: Launch,
    window: CudaSlice<f32>,
    twiddle: CudaSlice<f32>,
    first: CudaSlice<u32>,
    count: CudaSlice<u32>,
    weights: CudaSlice<f32>,
}

impl FbankCandidate for Oxide {
    type Pin = FbankPin;

    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &[LAYER],
        batches: Batches::Only(&FBANK_BATCHES),
        maths: Maths::All,
    }]);
    // every sum is of squared magnitudes times nonnegative weights from a +0 start, so
    // a zero energy is +0; a NaN sample reaches its frames' mean, hence every energy of
    // those frames; an infinite sample gives inf - inf in the compensated sums
    const SPECIAL_VALUES: SpecialValues = SpecialValues {
        finite: FiniteContract::UnitWaveform,
        nan: NanContract::Propagates,
        infinity: InfinityContract::Unspecified,
        signed_zero: SignedZeroContract::Positive,
    };

    fn implemented_pin(_spec: FbankSpec) -> Result<FbankPin, PlanError> {
        Ok(FbankPin::FftMelAccurate)
    }

    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: FbankSpec,
        pin: FbankPin,
    ) -> Result<Self, PlanError> {
        let launch = Launch::new(spec, pin)?;
        let tables = Tables::new(&FbankConstants::new())?;
        let stream = runtime.stream();
        let plan = Self {
            produce: kernels.function(FBANKDFT_FFT_MEL_ACCURATE)?,
            launch,
            window: stream.clone_htod(&tables.window)?,
            twiddle: stream.clone_htod(&tables.twiddle)?,
            first: stream.clone_htod(&tables.first)?,
            count: stream.clone_htod(&tables.count)?,
            weights: stream.clone_htod(&tables.weights)?,
        };
        // the producer may run on another stream than these uploads
        runtime.synchronize()?;
        Ok(plan)
    }

    fn enqueue(
        &self,
        waveform: &CudaView<'_, f32>,
        energies: &mut CudaViewMut<'_, f32>,
        _phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        self.launch.check(waveform.len(), energies.len())?;
        let width = MEL_WIDTH as u32;
        let waveform_len = waveform.len() as u64;
        let window_len = self.window.len() as u64;
        let twiddle_len = self.twiddle.len() as u64;
        let first_len = self.first.len() as u64;
        let count_len = self.count.len() as u64;
        let weights_len = self.weights.len() as u64;
        let energies_len = energies.len() as u64;

        let mut launch = runtime.stream().launch_builder(&self.produce);
        launch
            .arg(&SAMPLES)
            .arg(&FRAMES)
            .arg(&width)
            .arg(waveform)
            .arg(&waveform_len)
            .arg(&self.window)
            .arg(&window_len)
            .arg(&self.twiddle)
            .arg(&twiddle_len)
            .arg(&self.first)
            .arg(&first_len)
            .arg(&self.count)
            .arg(&count_len)
            .arg(&self.weights)
            .arg(&weights_len)
            .arg(energies)
            .arg(&energies_len);
        // SAFETY: the arguments follow the PTX signature of `fbankdft_fft_mel_accurate`
        // (three scalars, then pointer and length per slice); the plan checked the
        // table invariants the kernel indexes without bounds checks, and `check` above
        // the waveform and energy lengths of this grid
        unsafe { launch.launch(self.launch.config()) }?;
        Ok(())
    }
}

impl super::DriverCandidate for Oxide {
    const AREA: KernelModule = KernelModule::FbankDft;
    fn driver_coverage(tier: PtxTier) -> Coverage {
        Self::coverage(tier)
    }
    fn broad_evidence() -> Option<&'static crate::inference::cuda::implementation::BroadEvidence> {
        use crate::inference::cuda::implementation::{ArchitectureSpeed, BroadEvidence};
        const EVIDENCE: BroadEvidence = BroadEvidence::with_minimum(
            &[
                ArchitectureSpeed {
                    capability: crate::inference::cuda::ComputeCapability::new(12, 0),
                    minimum_speedup_milli: 1050,
                },
                ArchitectureSpeed {
                    capability: crate::inference::cuda::ComputeCapability::new(8, 9),
                    minimum_speedup_milli: 1050,
                },
            ],
            "fbank accurate FFT/mel FP32: removes dense DFT GEMM; original RTX 5060 Ti and xdev-ada accurate-producer reports, every measured case >=1.05x; p2c2-fbank Ada operator wins at all batches",
            crate::inference::cuda::ComputeCapability::new(8, 0),
        );
        Some(&EVIDENCE)
    }
    fn speed_scope(
        _boundary: crate::inference::cuda::implementation::BoundaryId,
        _batch: usize,
        math: CudaMath,
        device: &crate::inference::cuda::device::DeviceAttributes,
        _tier: PtxTier,
    ) -> Option<crate::inference::cuda::implementation::SpeedScope> {
        use crate::inference::cuda::implementation::SpeedScope;
        match math {
            CudaMath::Fp32 => Self::broad_evidence().map(SpeedScope::AllDevices),
            CudaMath::Tf32
                if device.capability() == crate::inference::cuda::ComputeCapability::new(8, 9) =>
            {
                Some(SpeedScope::MeasuredCapability {
                    capability: device.capability(),
                })
            }
            CudaMath::Tf32 => None,
        }
    }
    fn speed_summary(math: CudaMath) -> &'static str {
        match math {
            CudaMath::Fp32 => Self::broad_evidence()
                .expect("FP32 broad evidence")
                .summary(),
            CudaMath::Tf32 => {
                "p2c2-fbank Ada TF32: 1.30-1.82x, every batch 1..32; other architectures unmeasured"
            }
        }
    }
    fn driver_pin(
        _boundary: crate::inference::cuda::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        _device: &crate::inference::cuda::device::DeviceAttributes,
        _tier: PtxTier,
    ) -> Result<super::ConfigPin, PlanError> {
        Self::implemented_pin(FbankSpec::new(batch, math)?).map(super::ConfigPin::Fbank)
    }
}
