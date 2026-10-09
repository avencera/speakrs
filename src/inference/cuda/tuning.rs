//! Opt-in device tuning with production configurations and exact execution pins

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::CudaGraph;
use sha2::{Digest, Sha256};

use super::candidate::ConfigPin;
use super::device::DeviceAttributes;
use super::implementation::{self, BoundaryId};
use super::kernels::ModuleRequest;
use super::{CudaError, CudaMath, CudaRuntime, KernelModule, PtxTier};

mod accuracy;
mod bench;
mod driver_version;
mod file;

/// Inputs to the opt-in tuner; precision must match the pipeline that will use it
#[derive(Debug, Clone)]
pub struct CudaTuneOptions {
    /// Directory containing the CUDA segmentation and embedding safetensors files
    pub model_dir: PathBuf,
    /// Explicit output path; otherwise use SPEAKRS_CUDA_TUNE_FILE or user config
    pub output_path: Option<PathBuf>,
    /// CUDA device ordinal
    pub device: usize,
    /// Measure and report without writing or changing an existing tuning file
    pub dry_run: bool,
    /// Segmentation arithmetic; defaults to FP32
    pub segmentation_math: CudaMath,
    /// Embedding arithmetic; defaults to TF32, while filterbank remains FP32
    pub embedding_math: CudaMath,
}

impl Default for CudaTuneOptions {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("."),
            output_path: None,
            device: 0,
            dry_run: false,
            segmentation_math: CudaMath::Fp32,
            embedding_math: CudaMath::Tf32,
        }
    }
}

/// One approved candidate's median device time
#[derive(Debug, Clone)]
pub struct CudaTuneMeasurement {
    /// Human-readable candidate and configuration family
    pub candidate: String,
    /// Median graph replay milliseconds, including the complete boundary epilogue
    pub median_ms: f64,
}

/// The measured choices for one boundary, batch and precision
#[derive(Debug, Clone)]
pub struct CudaTuneRow {
    /// Stable model operation name
    pub boundary: String,
    /// Pipeline batch class
    pub batch: usize,
    /// Configured precision, unchanged by tuning
    pub math: CudaMath,
    /// All distinct production-policy choices timed for this tuple
    pub candidates: Vec<CudaTuneMeasurement>,
    /// The fastest measured choice
    pub selected: String,
}

/// Device measurements and the tuning-file result
#[derive(Debug, Clone)]
pub struct CudaTuneReport {
    /// Driver-reported device name
    pub device: String,
    /// Output file, or the proposed output for a dry run
    pub path: PathBuf,
    /// Whether the file was written
    pub written: bool,
    /// Boundary timing and selection table
    pub rows: Vec<CudaTuneRow>,
}

/// Errors from tuning, never from accuracy-policy changes
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaTuneError {
    /// CUDA device, library, model or launch error
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// File-system failure
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Invalid configuration, timing or tuning-file data
    #[error("CUDA tuning: {0}")]
    Invalid(String),
}

impl From<file::FileError> for CudaTuneError {
    fn from(error: file::FileError) -> Self {
        match error {
            file::FileError::Io(error) => Self::Io(error),
            other => Self::Invalid(other.to_string()),
        }
    }
}

/// Benchmark approved kernels and optional Library on real model shapes and weights
///
/// This explicitly requested operation warms each boundary, alternates candidate
/// order, and compares CUDA-event medians. It never adds configurations or
/// changes arithmetic policy. Startup only reads a matching file; it never tunes.
/// The caller must serialize GPU benchmarking against other workloads.
/// Set `dry_run` to keep the existing file unchanged.
pub fn tune_cuda(options: &CudaTuneOptions) -> Result<CudaTuneReport, CudaTuneError> {
    let runtime = CudaRuntime::for_tuning(options.device, BenchKind::Kernel)?;
    let catalogue = Catalogue::new(runtime.device(), runtime.ptx_tier())?;
    let key = device_key(runtime.device(), Some(&catalogue))?;
    let path = file::path(runtime.device(), options.output_path.as_deref())?;
    let measurements = bench::run(options, runtime)?;
    let (rows, entries) = select_winners(measurements)?;
    let tuned = file::TuneFile::new(key.clone(), entries);
    // a writer and a reader share validation, so no emitted choice can bypass it
    tuned.clone().validate(&key, &catalogue)?;
    if !options.dry_run {
        tuned.write(&path)?;
    }
    Ok(CudaTuneReport {
        device: key.device_name,
        path,
        written: !options.dry_run,
        rows,
    })
}

/// A validated tuple; boundary-specific batch restrictions cannot be omitted
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Tuple {
    boundary: BoundaryId,
    batch: usize,
    math: file::MathKey,
}

impl Tuple {
    fn new(boundary: BoundaryId, batch: usize, math: CudaMath) -> Result<Self, file::FileError> {
        if !boundary.batches().contains(batch)
            || boundary == BoundaryId::named("lstm.stack.input_proj")
        {
            return Err(file::FileError::Invalid(
                "not a tunable pipeline tuple".into(),
            ));
        }
        Ok(Self {
            boundary,
            batch,
            math: math.into(),
        })
    }

    fn parse(name: &str, batch: usize, math: CudaMath) -> Result<Self, file::FileError> {
        let boundary =
            BoundaryId::parse(name).map_err(|error| file::FileError::Invalid(error.to_string()))?;
        Self::new(boundary, batch, math)
    }
}

/// Only the catalogue constructs custom choices; input files cannot construct pins
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ApprovedChoice {
    Library,
    Kernel(ApprovedConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApprovedConfig {
    module: ModuleRequest,
    pin: ConfigPin,
    tuple: Tuple,
    family: &'static str,
    acceptance: accuracy::Approval,
}

impl ApprovedConfig {
    pub(crate) fn module(&self) -> ModuleRequest {
        self.module
    }
    pub(crate) fn pin(&self) -> ConfigPin {
        self.pin
    }
    pub(crate) fn acceptance(&self) -> &'static str {
        self.acceptance.description()
    }
    pub(crate) fn matches(&self, boundary: BoundaryId, batch: usize, math: CudaMath) -> bool {
        Tuple::new(boundary, batch, math).is_ok_and(|tuple| tuple == self.tuple)
    }
}

impl ApprovedChoice {
    pub(crate) fn label(&self) -> String {
        match self {
            Self::Library => "Library".into(),
            Self::Kernel(config) => format!("{}/{}", config.module.area().name(), config.family),
        }
    }

    fn key(&self) -> file::ChoiceKey {
        match self {
            Self::Library => file::ChoiceKey::Library,
            Self::Kernel(config) => file::ChoiceKey::Kernel {
                module: format!("{:?}", config.module),
                config_pin: format!("{:?}", config.pin),
            },
        }
    }
}

/// Approved identities, enumerated independently of speed or a tune file
#[derive(Debug)]
struct Catalogue(BTreeMap<Tuple, Vec<ApprovedChoice>>);

impl Catalogue {
    fn new(device: &DeviceAttributes, tier: PtxTier) -> Result<Self, CudaError> {
        let mut choices: BTreeMap<Tuple, Vec<ApprovedChoice>> = BTreeMap::new();
        for (boundary, batch, math, module, pin, family) in
            implementation::tuning_configurations(device, tier)?
        {
            let Some(acceptance) = accuracy::Policy::approve(boundary, math, pin) else {
                continue;
            };

            let tuple =
                Tuple::new(boundary, batch, math).map_err(|error| invalid(error.to_string()))?;
            let config = ApprovedChoice::Kernel(ApprovedConfig {
                module,
                pin,
                tuple,
                family,
                acceptance,
            });
            let entries = choices.entry(tuple).or_default();
            if !entries
                .iter()
                .any(|previous| previous.key() == config.key())
            {
                entries.push(config);
            }
        }
        for boundary in BoundaryId::all().filter(|id| {
            cfg!(feature = "_cuda-libraries") && *id != BoundaryId::named("lstm.stack.input_proj")
        }) {
            let batches = boundary.batches().iter();
            for batch in batches {
                for math in [CudaMath::Fp32, CudaMath::Tf32] {
                    let tuple = Tuple::new(boundary, batch, math)
                        .map_err(|error| invalid(error.to_string()))?;
                    choices
                        .entry(tuple)
                        .or_default()
                        .push(ApprovedChoice::Library);
                }
            }
        }
        Ok(Self(choices))
    }

    fn choices(&self, tuple: Tuple) -> &[ApprovedChoice] {
        self.0.get(&tuple).map_or(&[], Vec::as_slice)
    }
}

/// Benchmark families contain only catalogue choices, not arbitrary configurations
#[derive(Debug, Clone, Copy)]
pub(crate) enum BenchKind {
    Kernel,
    Fp32,
    Library,
}

#[derive(Debug)]
enum TuneSelection {
    File(file::ValidatedFile),
    Bench {
        kind: BenchKind,
        catalogue: Catalogue,
    },
}

#[derive(Debug)]
pub(crate) struct TuneControl {
    selection: TuneSelection,
    collecting: AtomicBool,
    graphs: Mutex<Vec<BoundaryGraph>>,
}

impl TuneControl {
    pub(crate) fn benchmark(
        kind: BenchKind,
        device: &DeviceAttributes,
        tier: PtxTier,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            selection: TuneSelection::Bench {
                kind,
                catalogue: Catalogue::new(device, tier)?,
            },
            collecting: AtomicBool::new(false),
            graphs: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn load(device: &DeviceAttributes, tier: PtxTier) -> Option<Self> {
        let load = || -> Result<Option<Self>, CudaTuneError> {
            // computing only the path does not hash artifacts or load candidate modules
            let path = file::path(device, None)?;
            let Some(file) = file::read(&path)? else {
                return Ok(None);
            };
            let catalogue = Catalogue::new(device, tier)?;
            let expected = device_key(device, Some(&catalogue))?;
            let selection = TuneSelection::File(file.validate(&expected, &catalogue)?);
            tracing::info!(path = %path.display(), "Loaded CUDA tune file");
            Ok(Some(Self {
                selection,
                collecting: AtomicBool::new(false),
                graphs: Mutex::new(Vec::new()),
            }))
        };
        match load() {
            Ok(tuning) => tuning,
            Err(error) => {
                tracing::warn!("Ignoring CUDA tune file: {error}");
                None
            }
        }
    }

    pub(crate) fn choice(
        &self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
    ) -> Option<ApprovedChoice> {
        let tuple = Tuple::new(boundary, batch, math).ok()?;
        match &self.selection {
            TuneSelection::File(file) => file.choice(tuple),
            TuneSelection::Bench { kind, catalogue } => {
                let choices = catalogue.choices(tuple);
                match kind {
                    BenchKind::Library => choices.iter().find(|choice| matches!(choice, ApprovedChoice::Library)),
                    BenchKind::Kernel => choices.first(),
                    BenchKind::Fp32 => choices.iter().find(|choice| matches!(choice, ApprovedChoice::Kernel(config) if config.family == "fp32")).or_else(|| choices.first()),
                }.cloned()
            }
        }
    }

    pub(crate) fn is_benchmark(&self) -> bool {
        matches!(self.selection, TuneSelection::Bench { .. })
    }

    pub(crate) fn begin_capture(&self) -> Result<(), CudaError> {
        if !self.is_benchmark() {
            return Err(invalid("capture is only allowed during explicit tuning"));
        }
        let graphs = self
            .graphs
            .lock()
            .map_err(|_| invalid("capture state is poisoned"))?;
        if !graphs.is_empty() || self.collecting.swap(true, Ordering::Relaxed) {
            return Err(invalid("a boundary capture is already active"));
        }
        Ok(())
    }

    pub(crate) fn take_graphs(&self) -> Result<Vec<BoundaryGraph>, CudaError> {
        self.collecting.store(false, Ordering::Relaxed);
        // drain even a poisoned collector so graphs cannot outlive model buffers
        let mut graphs = self
            .graphs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(std::mem::take(&mut *graphs))
    }

    pub(crate) fn record(
        &self,
        runtime: &CudaRuntime,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        run: impl FnOnce() -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        if !self.collecting.load(Ordering::Relaxed) {
            return run();
        }
        let choice = self
            .choice(boundary, batch, math)
            .ok_or_else(|| invalid("no production-policy choice for boundary capture"))?;
        runtime.synchronize()?;
        let context = runtime.context();
        let tracking = context.is_event_tracking();
        // all buffers in this captured boundary use this runtime's single stream
        // SAFETY: stream order provides the same dependency ordering during capture
        unsafe { context.disable_event_tracking() };
        let stream = runtime.stream();
        let captured = stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(CudaError::from)
            .and_then(|()| {
                let enqueued = run();
                let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                enqueued?;
                graph
                    .map_err(CudaError::from)?
                    .ok_or_else(|| invalid("boundary capture produced no graph"))
            });
        if tracking {
            // SAFETY: restores the tracking state observed before the capture
            unsafe { context.enable_event_tracking() };
        }
        let graph = captured?;
        // execute once so the following boundary receives the normal upstream values
        graph.launch()?;
        self.graphs
            .lock()
            .map_err(|_| invalid("capture state is poisoned"))?
            .push(BoundaryGraph {
                boundary,
                batch,
                math,
                choice,
                graph,
            });
        Ok(())
    }
}

pub(crate) struct BoundaryGraph {
    pub(crate) boundary: BoundaryId,
    pub(crate) batch: usize,
    pub(crate) math: CudaMath,
    pub(crate) choice: ApprovedChoice,
    pub(crate) graph: CudaGraph,
}

impl std::fmt::Debug for BoundaryGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundaryGraph")
            .field("boundary", &self.boundary)
            .field("batch", &self.batch)
            .field("math", &self.math)
            .field("choice", &self.choice)
            .finish_non_exhaustive()
    }
}

struct BenchmarkMeasurement {
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    choice: ApprovedChoice,
    median_ms: f64,
}

fn select_winners(
    measurements: Vec<BenchmarkMeasurement>,
) -> Result<(Vec<CudaTuneRow>, Vec<file::Entry>), CudaTuneError> {
    let mut groups: BTreeMap<Tuple, Vec<BenchmarkMeasurement>> = BTreeMap::new();
    for measurement in measurements {
        if !measurement.median_ms.is_finite() || measurement.median_ms <= 0.0 {
            return Err(CudaTuneError::Invalid("invalid timing median".into()));
        }
        let tuple = Tuple::new(measurement.boundary, measurement.batch, measurement.math)?;
        groups.entry(tuple).or_default().push(measurement);
    }
    if groups.is_empty() {
        return Err(CudaTuneError::Invalid("no boundaries were measured".into()));
    }
    let mut rows = Vec::new();
    let mut entries = Vec::new();
    for (tuple, measurements) in groups {
        let winner = measurements
            .iter()
            .min_by(|left, right| left.median_ms.total_cmp(&right.median_ms))
            .expect("a measured group is nonempty");
        entries.push(file::Entry {
            boundary: tuple.boundary.name().into(),
            batch: tuple.batch,
            math: tuple.math,
            choice: winner.choice.key(),
            median_ms: winner.median_ms,
        });
        rows.push(CudaTuneRow {
            boundary: tuple.boundary.name().into(),
            batch: tuple.batch,
            math: tuple.math.into(),
            selected: winner.choice.label(),
            candidates: measurements
                .iter()
                .map(|measurement| CudaTuneMeasurement {
                    candidate: measurement.choice.label(),
                    median_ms: measurement.median_ms,
                })
                .collect(),
        });
    }
    Ok((rows, entries))
}

fn device_key(
    device: &DeviceAttributes,
    catalogue: Option<&Catalogue>,
) -> Result<file::DeviceKey, CudaTuneError> {
    let cc = device.capability();
    let mut key = file::DeviceKey {
        device_name: device.name().into(),
        capability: [cc.major, cc.minor],
        sm_count: device.multiprocessors().get(),
        driver_version: driver_version::DriverVersion::read()?,
        speakrs_version: env!("CARGO_PKG_VERSION").into(),
        artifact_version: String::new(),
        accuracy_policy: accuracy::Policy::IDENTITY.into(),
    };
    let mut digest = Sha256::new();
    digest.update(b"speakrs-cuda-tuning-catalogue-v3");
    digest.update([u8::from(cfg!(feature = "_cuda-libraries"))]);
    for area in [
        KernelModule::Fbank,
        KernelModule::Embedding,
        KernelModule::Segmentation,
        KernelModule::Resnet,
        KernelModule::Wideconv,
        KernelModule::Lstm,
        KernelModule::LstmProj,
        KernelModule::Sincnet,
        KernelModule::Segdense,
        KernelModule::FbankDft,
    ] {
        digest.update(area.manifest().as_bytes());
        let variants = area.variants();
        for (tier, ptx) in variants.iter() {
            digest.update(format!("{tier:?}").as_bytes());
            digest.update(ptx.as_bytes());
            for cubin in variants.embedded(tier).expect("enumerated variant").cubins {
                digest.update(format!("{:?}", cubin.arch).as_bytes());
                digest.update(cubin.bytes);
            }
        }
    }
    if let Some(catalogue) = catalogue {
        for (tuple, choices) in &catalogue.0 {
            digest.update(format!("{tuple:?}").as_bytes());
            for choice in choices {
                digest.update(format!("{:?}", choice.key()).as_bytes());
            }
        }
    }
    key.artifact_version = format!("{:x}", digest.finalize());
    Ok(key)
}

pub(crate) fn invalid(reason: impl Into<String>) -> CudaError {
    CudaError::Unsupported {
        context: "CUDA tuning",
        reason: reason.into(),
    }
}

#[cfg(test)]
pub(crate) mod tests;
