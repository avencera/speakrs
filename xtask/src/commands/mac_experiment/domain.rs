use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::cmd::project_root;

const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MacExperimentSpec {
    pub schema_version: u32,
    pub experiment_id: String,
    pub dataset: DatasetSlice,
    #[serde(default)]
    pub datasets_dir: Option<PathBuf>,
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
    pub inference: InferenceVariant,
    pub post_inference: Vec<PostInferenceVariant>,
    pub performance: PerformanceProtocol,
    pub acceptance: AcceptancePolicy,
    #[serde(default)]
    pub profile: Option<ProfileProtocol>,
    #[serde(default)]
    pub baseline_run: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatasetSlice {
    pub id: String,
    #[serde(default = "default_max_files")]
    pub max_files: u32,
    #[serde(default = "default_max_minutes")]
    pub max_minutes: u32,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) enum CoreMlMode {
    #[serde(rename = "coreml")]
    CoreMl,
    #[serde(rename = "coreml_fast")]
    CoreMlFast,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InferenceLayout {
    OneSecondPhased,
    /// Historical 0.96-second control retained only for archived manifests
    AlignedS12,
    /// Historical 1.04-second control retained only for archived manifests
    AlignedS13,
    FastS25,
    #[serde(rename = "per_window_1s")]
    PerWindow1s,
    /// Historical 1.04-second control retained only for archived manifests
    #[serde(rename = "per_window_104")]
    PerWindow104,
}

impl InferenceLayout {
    pub(crate) const fn step_seconds(self) -> f64 {
        match self {
            Self::OneSecondPhased | Self::PerWindow1s => 1.0,
            Self::AlignedS12 => 0.96,
            Self::AlignedS13 | Self::PerWindow104 => 1.04,
            Self::FastS25 => 2.0,
        }
    }

    pub(crate) const fn is_archived_stride(self) -> bool {
        matches!(
            self,
            Self::AlignedS12 | Self::AlignedS13 | Self::PerWindow104
        )
    }

    fn required_model_stems(self, shape_ladder: ShapeLadder) -> &'static [&'static str] {
        match (self, shape_ladder) {
            (Self::OneSecondPhased, ShapeLadder::Reduced) => &[
                "wespeaker-chunk-emb-p1s-w21",
                "wespeaker-chunk-emb-p1s-w51",
                "wespeaker-chunk-emb-p1s-w111",
            ],
            (Self::OneSecondPhased, ShapeLadder::Full) => &[
                "wespeaker-chunk-emb-p1s-w21",
                "wespeaker-chunk-emb-p1s-w36",
                "wespeaker-chunk-emb-p1s-w51",
                "wespeaker-chunk-emb-p1s-w81",
                "wespeaker-chunk-emb-p1s-w111",
            ],
            (Self::AlignedS12, ShapeLadder::Full) => &[
                "wespeaker-chunk-emb-s12-w22",
                "wespeaker-chunk-emb-s12-w37",
                "wespeaker-chunk-emb-s12-w53",
                "wespeaker-chunk-emb-s12-w84",
                "wespeaker-chunk-emb-s12-w116",
            ],
            (Self::AlignedS13, ShapeLadder::Full) => &[
                "wespeaker-chunk-emb-s13-w20",
                "wespeaker-chunk-emb-s13-w34",
                "wespeaker-chunk-emb-s13-w49",
                "wespeaker-chunk-emb-s13-w77",
                "wespeaker-chunk-emb-s13-w106",
            ],
            (Self::FastS25, ShapeLadder::Full) => &[
                "wespeaker-chunk-emb-s25-w11",
                "wespeaker-chunk-emb-s25-w16",
                "wespeaker-chunk-emb-s25-w21",
                "wespeaker-chunk-emb-s25-w26",
                "wespeaker-chunk-emb-s25-w36",
                "wespeaker-chunk-emb-s25-w46",
                "wespeaker-chunk-emb-s25-w56",
            ],
            (Self::PerWindow1s | Self::PerWindow104, ShapeLadder::Full) => &[],
            (_, ShapeLadder::Reduced) => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ShapeLadder {
    #[default]
    Full,
    Reduced,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SegmentationWorkers {
    #[default]
    Automatic,
    Four,
    Six,
    Eight,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FbankPreparationWorkers {
    One,
    #[default]
    Two,
    Four,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FbankNormalizationScope {
    #[default]
    Chunk,
    TenSecondSegments,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EmbeddingComputeUnits {
    #[default]
    All,
    CpuOnly,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InferenceVariant {
    pub mode: CoreMlMode,
    pub layout: InferenceLayout,
    #[serde(default)]
    pub shape_ladder: ShapeLadder,
    #[serde(default)]
    pub segmentation_workers: SegmentationWorkers,
    #[serde(default)]
    pub filterbank_preparation_workers: FbankPreparationWorkers,
    #[serde(default)]
    pub filterbank_normalization_scope: FbankNormalizationScope,
    #[serde(default)]
    pub embedding_compute_units: EmbeddingComputeUnits,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PostInferenceVariant {
    pub id: String,
    #[serde(default)]
    pub vbx_max_iters: Option<usize>,
    #[serde(default)]
    pub vbx_fb: Option<f64>,
    #[serde(default)]
    pub clean_frame_seconds: Option<f64>,
    #[serde(default)]
    pub ahc_stopping: Option<ArchivedAhcStopping>,
    #[serde(default)]
    pub clustering_backend: Option<ExperimentClusteringBackend>,
    #[serde(default)]
    pub documented_der_outliers: Vec<DocumentedDerOutlier>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ExperimentClusteringBackend {
    SphereVbxPf {
        fa: f64,
        fb: f64,
        max_iters: usize,
        responsibility_tolerance: f64,
        initialization: ExperimentSphereInitialization,
        #[serde(default)]
        ahc_initialization: ExperimentSphereAhcInitialization,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ExperimentSphereInitialization {
    Hard,
    Smoothed { scale: f64 },
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExperimentSphereAhcInitialization {
    #[default]
    Cosine,
    PldaTransformed,
}

impl ExperimentClusteringBackend {
    pub(crate) fn into_pipeline(self) -> Result<speakrs::pipeline::ClusteringBackend> {
        match self {
            Self::SphereVbxPf {
                fa,
                fb,
                max_iters,
                responsibility_tolerance,
                initialization,
                ahc_initialization,
            } => {
                let initialization = match initialization {
                    ExperimentSphereInitialization::Hard => {
                        speakrs::pipeline::SphereVbxInitialization::Hard
                    }
                    ExperimentSphereInitialization::Smoothed { scale } => {
                        speakrs::pipeline::SphereVbxInitialization::Smoothed(
                            speakrs::pipeline::ResponsibilitySmoothing::new(scale)?,
                        )
                    }
                };
                let ahc_initialization = match ahc_initialization {
                    ExperimentSphereAhcInitialization::Cosine => {
                        speakrs::pipeline::SphereVbxAhcInitialization::Cosine
                    }
                    ExperimentSphereAhcInitialization::PldaTransformed => {
                        speakrs::pipeline::SphereVbxAhcInitialization::PldaTransformed
                    }
                };
                let config = speakrs::pipeline::SphereVbxPfConfig::new(
                    fa,
                    fb,
                    max_iters,
                    responsibility_tolerance,
                    initialization,
                    ahc_initialization,
                )?;

                Ok(speakrs::pipeline::ClusteringBackend::SphereVbxPf(config))
            }
        }
    }
}

/// Rejected AHC policy retained only so archived run manifests remain readable
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ArchivedAhcStopping {
    DistanceThreshold,
    EstablishedClusters { minimum_cluster_size: usize },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DocumentedDerOutlier {
    pub file_id: String,
    pub cause: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PerformanceProtocol {
    #[serde(default)]
    pub warmups: u32,
    pub repetitions: u32,
    #[serde(default)]
    pub sleep_seconds: u64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl PerformanceProtocol {
    pub(crate) const fn warmups(&self) -> u32 {
        self.warmups
    }

    pub(crate) const fn repetitions(&self) -> u32 {
        self.repetitions
    }

    pub(crate) const fn sleep_seconds(&self) -> u64 {
        self.sleep_seconds
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AcceptancePolicy {
    StandardPerformance,
    StandardDer,
    FastPareto,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProfileTemplate {
    CoreMl,
    MetalSystemTrace,
    Allocations,
    TimeProfiler,
}

impl ProfileTemplate {
    pub(crate) const fn xctrace_name(self) -> &'static str {
        match self {
            Self::CoreMl => "Core ML",
            Self::MetalSystemTrace => "Metal System Trace",
            Self::Allocations => "Allocations",
            Self::TimeProfiler => "Time Profiler",
        }
    }

    pub(crate) const fn file_stem(self) -> &'static str {
        match self {
            Self::CoreMl => "core-ml",
            Self::MetalSystemTrace => "metal-system-trace",
            Self::Allocations => "allocations",
            Self::TimeProfiler => "time-profiler",
        }
    }

    pub(crate) const fn export_schemas(self) -> &'static [&'static str] {
        match self {
            Self::CoreMl => &["coreml-os-signpost"],
            Self::MetalSystemTrace => &[
                "metal-gpu-intervals",
                "metal-application-command-buffer-submissions",
            ],
            Self::Allocations => &[],
            Self::TimeProfiler => &["time-profile"],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProfileProtocol {
    pub template: ProfileTemplate,
    #[serde(default)]
    pub file_index: usize,
    #[serde(default = "default_profile_time_limit")]
    pub time_limit_seconds: u64,
}

impl Default for ProfileProtocol {
    fn default() -> Self {
        Self {
            template: ProfileTemplate::CoreMl,
            file_index: 0,
            time_limit_seconds: default_profile_time_limit(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedExperiment {
    spec: MacExperimentSpec,
    id: ExperimentId,
    models_dir: PathBuf,
    datasets_dir: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExperimentId(String);

impl ExperimentId {
    fn parse(field: &str, value: &str) -> Result<Self> {
        ensure!(!value.is_empty(), "{field} must not be empty");
        ensure!(value.len() <= 64, "{field} must be at most 64 characters");
        ensure!(
            value
                .bytes()
                .enumerate()
                .all(|(idx, byte)| byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || (idx > 0 && matches!(byte, b'-' | b'_'))),
            "{field} '{value}' must use lowercase ASCII letters, digits, '-' or '_'"
        );
        Ok(Self(value.to_owned()))
    }
}

impl ValidatedExperiment {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .wrap_err_with(|| format!("failed to read experiment spec {}", path.display()))?;
        let spec: MacExperimentSpec = serde_json::from_str(&text)
            .wrap_err_with(|| format!("invalid experiment spec {}", path.display()))?;
        Self::validate(spec)
    }

    pub(crate) fn from_spec(spec: MacExperimentSpec) -> Result<Self> {
        Self::validate(spec)
    }

    fn validate(spec: MacExperimentSpec) -> Result<Self> {
        let id = validate_domain(&spec)?;

        let root = project_root();
        let models_dir = spec
            .models_dir
            .as_ref()
            .map(|path| resolve_path(&root, path))
            .unwrap_or_else(|| root.join("fixtures/models"));
        ensure!(
            models_dir.is_dir(),
            "models_dir does not exist: {}",
            models_dir.display()
        );
        validate_layout_assets(&models_dir, &spec.inference)?;

        let datasets_dir = spec
            .datasets_dir
            .as_ref()
            .map(|path| resolve_path(&root, path))
            .unwrap_or_else(|| root.join("fixtures/datasets"));
        ensure!(
            datasets_dir.is_dir(),
            "datasets_dir does not exist: {}",
            datasets_dir.display()
        );

        let dataset = crate::datasets::find_dataset(&spec.dataset.id)
            .ok_or_else(|| color_eyre::eyre::eyre!("unknown dataset '{}'", spec.dataset.id))?;
        let dataset_dir = dataset.dataset_dir(&datasets_dir);
        ensure!(
            dataset_dir.is_dir(),
            "dataset '{}' is not available locally at {}",
            spec.dataset.id,
            dataset_dir.display()
        );
        if let Some(baseline_run) = &spec.baseline_run {
            let baseline_run = resolve_path(&root, baseline_run);
            ensure!(
                baseline_run.join("manifest.json").is_file(),
                "baseline_run does not contain a manifest: {}",
                baseline_run.display()
            );
        }

        Ok(Self {
            spec,
            id,
            models_dir,
            datasets_dir,
        })
    }

    pub(crate) fn id(&self) -> &str {
        &self.id.0
    }

    pub(crate) const fn spec(&self) -> &MacExperimentSpec {
        &self.spec
    }

    pub(crate) fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub(crate) fn datasets_dir(&self) -> &Path {
        &self.datasets_dir
    }

    pub(crate) fn profiled_model_paths(&self) -> Vec<PathBuf> {
        let segmentation_suffix = match self.inference().mode {
            CoreMlMode::CoreMl => "",
            CoreMlMode::CoreMlFast => "-w8a16",
        };
        let mut paths = ["", "-b32", "-b64"]
            .into_iter()
            .map(|batch| {
                self.models_dir.join(format!(
                    "segmentation-3.0{batch}{segmentation_suffix}.mlmodelc"
                ))
            })
            .chain(
                ["wespeaker-fbank.mlmodelc", "wespeaker-fbank-30s.mlmodelc"]
                    .into_iter()
                    .map(|name| self.models_dir.join(name)),
            )
            .collect::<Vec<_>>();
        match self.inference().layout {
            InferenceLayout::PerWindow1s | InferenceLayout::PerWindow104 => {
                paths.extend(
                    [
                        "wespeaker-fbank-b32.mlmodelc",
                        "wespeaker-multimask-tail-b32.mlmodelc",
                    ]
                    .into_iter()
                    .map(|name| self.models_dir.join(name)),
                );
            }
            layout => {
                let suffix = match self.inference().mode {
                    CoreMlMode::CoreMl => "",
                    CoreMlMode::CoreMlFast => "-w8a16",
                };
                paths.extend(
                    layout
                        .required_model_stems(self.inference().shape_ladder)
                        .iter()
                        .map(|stem| self.models_dir.join(format!("{stem}{suffix}.mlmodelc"))),
                );
            }
        }
        paths.retain(|path| path.is_dir());
        paths.sort();
        paths.dedup();
        paths
    }

    pub(crate) fn post_inference(&self) -> &[PostInferenceVariant] {
        &self.spec.post_inference
    }

    pub(crate) const fn inference(&self) -> &InferenceVariant {
        &self.spec.inference
    }

    pub(crate) const fn performance(&self) -> &PerformanceProtocol {
        &self.spec.performance
    }

    pub(crate) fn ensure_runnable(&self) -> Result<()> {
        if self.inference().layout.is_archived_stride() {
            bail!(
                "inference layout {:?} uses a historical stride; archived runs can be summarized but new runs are restricted to 1.0-second Standard and 2.0-second Fast strides",
                self.inference().layout
            );
        }

        let Some(candidate) = self.post_inference().iter().find(|candidate| {
            matches!(
                candidate.ahc_stopping,
                Some(ArchivedAhcStopping::EstablishedClusters { .. })
            )
        }) else {
            return Ok(());
        };

        bail!(
            "post-inference candidate '{}' uses the rejected archived-only ahc_stopping policy; its run can be summarized but not executed",
            candidate.id
        )
    }

    pub(crate) fn profile(&self) -> ProfileProtocol {
        self.spec.profile.clone().unwrap_or_default()
    }
}

fn validate_domain(spec: &MacExperimentSpec) -> Result<ExperimentId> {
    ensure!(
        spec.schema_version == SCHEMA_VERSION,
        "schema_version {} is not supported; expected {SCHEMA_VERSION}",
        spec.schema_version
    );
    let id = ExperimentId::parse("experiment_id", &spec.experiment_id)?;
    ExperimentId::parse("dataset.id", &spec.dataset.id)?;
    ensure!(
        spec.dataset.max_files > 0,
        "dataset.max_files must be greater than zero"
    );
    ensure!(
        spec.dataset.max_minutes > 0,
        "dataset.max_minutes must be greater than zero"
    );
    validate_unique_ids("dataset.files", &spec.dataset.files)?;
    for file_id in &spec.dataset.files {
        ensure!(
            !file_id.is_empty()
                && file_id != "."
                && file_id != ".."
                && !file_id.contains('/')
                && !file_id.contains('\\'),
            "dataset.files contains unsafe file id '{file_id}'"
        );
    }
    ensure!(
        !spec.post_inference.is_empty(),
        "post_inference must contain at least one candidate"
    );
    let candidate_ids: Vec<_> = spec
        .post_inference
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect();
    validate_unique_ids("post_inference.id", &candidate_ids)?;
    for candidate in &spec.post_inference {
        ExperimentId::parse("post_inference.id", &candidate.id)?;
        if let Some(max_iters) = candidate.vbx_max_iters {
            ensure!(
                max_iters > 0,
                "post_inference '{}' vbx_max_iters must be greater than zero",
                candidate.id
            );
        }
        if let Some(fb) = candidate.vbx_fb {
            ensure!(
                fb.is_finite() && fb > 0.0,
                "post_inference '{}' vbx_fb must be finite and greater than zero",
                candidate.id
            );
        }
        if let Some(seconds) = candidate.clean_frame_seconds {
            speakrs::pipeline::CleanFrameDuration::new(seconds).map_err(|error| {
                color_eyre::eyre::eyre!(
                    "post_inference '{}' has invalid clean_frame_seconds: {error}",
                    candidate.id
                )
            })?;
        }
        if let Some(ArchivedAhcStopping::EstablishedClusters {
            minimum_cluster_size,
        }) = candidate.ahc_stopping
        {
            ensure!(
                minimum_cluster_size > 0,
                "post_inference '{}' ahc_stopping minimum_cluster_size must be greater than zero",
                candidate.id
            );
        }
        if let Some(ExperimentClusteringBackend::SphereVbxPf { .. }) = candidate.clustering_backend
        {
            ensure!(
                candidate.vbx_fb.is_none() && candidate.vbx_max_iters.is_none(),
                "post_inference '{}' cannot combine SphereVBx-PF with Gaussian VBx controls",
                candidate.id
            );
            candidate
                .clustering_backend
                .unwrap()
                .into_pipeline()
                .map_err(|error| {
                    color_eyre::eyre::eyre!(
                        "post_inference '{}' has invalid SphereVBx-PF configuration: {error}",
                        candidate.id
                    )
                })?;
        }
        let outlier_ids = candidate
            .documented_der_outliers
            .iter()
            .map(|outlier| outlier.file_id.clone())
            .collect::<Vec<_>>();
        validate_unique_ids(
            "post_inference.documented_der_outliers.file_id",
            &outlier_ids,
        )?;
        for outlier in &candidate.documented_der_outliers {
            ensure!(
                !outlier.file_id.is_empty()
                    && outlier.file_id != "."
                    && outlier.file_id != ".."
                    && !outlier.file_id.contains('/')
                    && !outlier.file_id.contains('\\'),
                "post_inference '{}' has unsafe documented outlier file id '{}'",
                candidate.id,
                outlier.file_id
            );
            ensure!(
                !outlier.cause.trim().is_empty(),
                "post_inference '{}' has an empty documented outlier cause for '{}'",
                candidate.id,
                outlier.file_id
            );
        }
    }
    ensure!(
        spec.performance.repetitions > 0,
        "performance.repetitions must be greater than zero"
    );
    if let Some(profile) = &spec.profile {
        ensure!(
            profile.time_limit_seconds > 0,
            "profile.time_limit_seconds must be greater than zero"
        );
    }
    if spec.baseline_run.is_some() {
        ensure!(
            spec.performance.repetitions >= 4 && spec.performance.repetitions.is_multiple_of(2),
            "baseline_run comparisons require an even repetition count of at least four"
        );
    }
    validate_inference(&spec.inference, spec.acceptance)?;
    Ok(id)
}

fn validate_inference(inference: &InferenceVariant, acceptance: AcceptancePolicy) -> Result<()> {
    let compatible = matches!(
        (inference.mode, inference.layout),
        (
            CoreMlMode::CoreMl,
            InferenceLayout::OneSecondPhased
                | InferenceLayout::AlignedS12
                | InferenceLayout::AlignedS13
                | InferenceLayout::PerWindow1s
                | InferenceLayout::PerWindow104
        ) | (CoreMlMode::CoreMlFast, InferenceLayout::FastS25)
    );
    ensure!(
        compatible,
        "inference layout {:?} is incompatible with mode {:?}",
        inference.layout,
        inference.mode
    );
    ensure!(
        inference.shape_ladder == ShapeLadder::Full
            || inference.layout == InferenceLayout::OneSecondPhased,
        "reduced shape ladder is supported only by the one_second_phased layout"
    );
    if matches!(acceptance, AcceptancePolicy::FastPareto) {
        ensure!(
            inference.mode == CoreMlMode::CoreMlFast,
            "fast_pareto acceptance requires coreml_fast mode"
        );
    } else {
        ensure!(
            inference.mode == CoreMlMode::CoreMl,
            "coreml_fast mode requires fast_pareto acceptance"
        );
    }
    Ok(())
}

fn validate_layout_assets(models_dir: &Path, inference: &InferenceVariant) -> Result<()> {
    if inference.layout.is_archived_stride() {
        return Ok(());
    }

    let segmentation_suffix = match inference.mode {
        CoreMlMode::CoreMl => "",
        CoreMlMode::CoreMlFast => "-w8a16",
    };
    for stem in [
        "segmentation-3.0",
        "segmentation-3.0-b32",
        "segmentation-3.0-b64",
    ] {
        require_model_bundle(
            models_dir,
            &format!("{stem}{segmentation_suffix}.mlmodelc"),
            inference.layout,
        )?;
    }

    for name in [
        "wespeaker-fbank.mlmodelc",
        "wespeaker-fbank-b32.mlmodelc",
        "wespeaker-voxceleb-resnet34-tail.mlmodelc",
        "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
        "wespeaker-multimask-tail-b32.mlmodelc",
    ] {
        require_model_bundle(models_dir, name, inference.layout)?;
    }

    if matches!(
        inference.layout,
        InferenceLayout::PerWindow1s | InferenceLayout::PerWindow104
    ) {
        return Ok(());
    }

    require_model_bundle(models_dir, "wespeaker-fbank-30s.mlmodelc", inference.layout)?;
    let chunk_suffix = match inference.mode {
        CoreMlMode::CoreMl => "",
        CoreMlMode::CoreMlFast => "-w8a16",
    };
    for stem in inference
        .layout
        .required_model_stems(inference.shape_ladder)
    {
        require_model_bundle(
            models_dir,
            &format!("{stem}{chunk_suffix}.mlmodelc"),
            inference.layout,
        )?;
    }

    Ok(())
}

fn require_model_bundle(models_dir: &Path, name: &str, layout: InferenceLayout) -> Result<()> {
    let path = models_dir.join(name);
    ensure!(
        path.is_dir(),
        "inference layout {:?} requires missing CoreML model asset {}",
        layout,
        path.display()
    );
    Ok(())
}

fn validate_unique_ids(field: &str, values: &[String]) -> Result<()> {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if !seen.insert(value) {
            bail!("{field} contains duplicate value '{value}'");
        }
    }
    Ok(())
}

fn resolve_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

const fn default_max_files() -> u32 {
    u32::MAX
}

const fn default_max_minutes() -> u32 {
    u32::MAX
}

const fn default_seed() -> u64 {
    42
}

const fn default_profile_time_limit() -> u64 {
    30 * 60
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_spec() -> MacExperimentSpec {
        MacExperimentSpec {
            schema_version: SCHEMA_VERSION,
            experiment_id: "mac-smoke".to_owned(),
            dataset: DatasetSlice {
                id: "ami-ihm".to_owned(),
                max_files: 1,
                max_minutes: 30,
                files: Vec::new(),
            },
            datasets_dir: None,
            models_dir: None,
            inference: InferenceVariant {
                mode: CoreMlMode::CoreMl,
                layout: InferenceLayout::OneSecondPhased,
                shape_ladder: ShapeLadder::Full,
                segmentation_workers: SegmentationWorkers::Automatic,
                filterbank_preparation_workers: FbankPreparationWorkers::Two,
                filterbank_normalization_scope: FbankNormalizationScope::Chunk,
                embedding_compute_units: EmbeddingComputeUnits::All,
            },
            post_inference: vec![PostInferenceVariant {
                id: "default".to_owned(),
                vbx_max_iters: None,
                vbx_fb: None,
                clean_frame_seconds: None,
                ahc_stopping: None,
                clustering_backend: None,
                documented_der_outliers: Vec::new(),
            }],
            performance: PerformanceProtocol {
                warmups: 0,
                repetitions: 1,
                sleep_seconds: 0,
                seed: default_seed(),
            },
            acceptance: AcceptancePolicy::StandardPerformance,
            profile: None,
            baseline_run: None,
        }
    }

    #[test]
    fn mac_experiment_accepts_current_coreml_layout() {
        let spec = valid_spec();
        let id = validate_domain(&spec).unwrap();
        assert_eq!(id.0, "mac-smoke");
        assert_eq!(spec.inference.layout.step_seconds(), 1.0);
    }

    #[test]
    fn mac_experiment_accepts_valid_sphere_vbx_configuration() {
        let mut spec = valid_spec();
        spec.post_inference[0].clustering_backend =
            Some(ExperimentClusteringBackend::SphereVbxPf {
                fa: 12.0,
                fb: 0.3,
                max_iters: 10,
                responsibility_tolerance: 1e-8,
                initialization: ExperimentSphereInitialization::Smoothed { scale: 7.0 },
                ahc_initialization: ExperimentSphereAhcInitialization::Cosine,
            });

        validate_domain(&spec).unwrap();
    }

    #[test]
    fn mac_experiment_rejects_mixed_gaussian_and_sphere_controls() {
        let mut spec = valid_spec();
        spec.post_inference[0].vbx_fb = Some(0.5);
        spec.post_inference[0].clustering_backend =
            Some(ExperimentClusteringBackend::SphereVbxPf {
                fa: 12.0,
                fb: 0.3,
                max_iters: 10,
                responsibility_tolerance: 1e-8,
                initialization: ExperimentSphereInitialization::Hard,
                ahc_initialization: ExperimentSphereAhcInitialization::Cosine,
            });

        let error = validate_domain(&spec).unwrap_err();
        assert!(error.to_string().contains("cannot combine"));
    }

    #[test]
    fn mac_experiment_rejects_invalid_sphere_initialization() {
        let mut spec = valid_spec();
        spec.post_inference[0].clustering_backend =
            Some(ExperimentClusteringBackend::SphereVbxPf {
                fa: 12.0,
                fb: 0.3,
                max_iters: 10,
                responsibility_tolerance: 1e-8,
                initialization: ExperimentSphereInitialization::Smoothed { scale: 0.0 },
                ahc_initialization: ExperimentSphereAhcInitialization::Cosine,
            });

        let error = validate_domain(&spec).unwrap_err();
        assert!(error.to_string().contains("configuration"));
    }

    #[test]
    fn archived_ahc_policy_parses_but_is_not_runnable() {
        let mut spec = valid_spec();
        spec.post_inference[0].ahc_stopping = Some(ArchivedAhcStopping::EstablishedClusters {
            minimum_cluster_size: 5,
        });
        validate_domain(&spec).unwrap();

        let experiment = ValidatedExperiment::from_spec(spec).unwrap();
        let error = experiment.ensure_runnable().unwrap_err();
        assert!(error.to_string().contains("archived-only ahc_stopping"));
    }

    #[test]
    fn historical_stride_layout_parses_but_is_not_runnable() {
        let mut spec = valid_spec();
        spec.inference.layout = InferenceLayout::AlignedS13;
        validate_domain(&spec).unwrap();

        let experiment = ValidatedExperiment::from_spec(spec).unwrap();
        let error = experiment.ensure_runnable().unwrap_err();
        assert!(error.to_string().contains("historical stride"));
    }

    #[test]
    fn archived_ahc_policy_rejects_zero_cluster_size() {
        let mut spec = valid_spec();
        spec.post_inference[0].ahc_stopping = Some(ArchivedAhcStopping::EstablishedClusters {
            minimum_cluster_size: 0,
        });

        let error = validate_domain(&spec).unwrap_err();
        assert!(error.to_string().contains("minimum_cluster_size"));
    }

    #[test]
    fn mac_experiment_rejects_incompatible_fast_layout() {
        let mut spec = valid_spec();
        spec.inference.mode = CoreMlMode::CoreMlFast;

        let err = validate_domain(&spec).unwrap_err();
        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn mac_experiment_rejects_duplicate_candidate_ids() {
        let mut spec = valid_spec();
        spec.post_inference.push(spec.post_inference[0].clone());

        let err = validate_domain(&spec).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn mac_experiment_rejects_empty_documented_outlier_causes() {
        let mut spec = valid_spec();
        spec.post_inference[0]
            .documented_der_outliers
            .push(DocumentedDerOutlier {
                file_id: "meeting".to_owned(),
                cause: "  ".to_owned(),
            });

        let err = validate_domain(&spec).unwrap_err();
        assert!(err.to_string().contains("empty documented outlier cause"));
    }

    #[test]
    fn mac_experiment_rejects_invalid_vbx_fb() {
        let mut spec = valid_spec();
        spec.post_inference[0].vbx_fb = Some(f64::NAN);

        let err = validate_domain(&spec).unwrap_err();
        assert!(err.to_string().contains("vbx_fb"));
    }

    #[test]
    fn mac_experiment_rejects_reduced_ladder_for_unrelated_layout() {
        let mut spec = valid_spec();
        spec.inference.layout = InferenceLayout::AlignedS13;
        spec.inference.shape_ladder = ShapeLadder::Reduced;

        let err = validate_domain(&spec).unwrap_err();
        assert!(err.to_string().contains("reduced shape ladder"));
    }

    #[test]
    fn inference_policy_defaults_preserve_current_execution() {
        let inference: InferenceVariant =
            serde_json::from_str(r#"{"mode":"coreml","layout":"one_second_phased"}"#).unwrap();

        assert_eq!(inference.shape_ladder, ShapeLadder::Full);
        assert_eq!(
            inference.segmentation_workers,
            SegmentationWorkers::Automatic
        );
        assert_eq!(
            inference.filterbank_preparation_workers,
            FbankPreparationWorkers::Two
        );
        assert_eq!(
            inference.filterbank_normalization_scope,
            FbankNormalizationScope::Chunk
        );
        assert_eq!(
            inference.embedding_compute_units,
            EmbeddingComputeUnits::All
        );
    }

    #[test]
    fn inference_policy_accepts_cpu_only_embedding() {
        let inference: InferenceVariant = serde_json::from_str(
            r#"{"mode":"coreml","layout":"per_window_1s","embedding_compute_units":"cpu_only"}"#,
        )
        .unwrap();

        assert_eq!(
            inference.embedding_compute_units,
            EmbeddingComputeUnits::CpuOnly
        );
    }

    #[test]
    fn time_profiler_exports_sample_table() {
        assert_eq!(
            ProfileTemplate::TimeProfiler.export_schemas(),
            &["time-profile"]
        );
    }

    #[test]
    fn per_window_validation_requires_its_native_assets() {
        let temp = tempfile::tempdir().unwrap();
        let mut inference = valid_spec().inference;
        inference.layout = InferenceLayout::PerWindow1s;
        for name in [
            "segmentation-3.0.mlmodelc",
            "segmentation-3.0-b32.mlmodelc",
            "segmentation-3.0-b64.mlmodelc",
            "wespeaker-fbank.mlmodelc",
            "wespeaker-fbank-b32.mlmodelc",
            "wespeaker-voxceleb-resnet34-tail.mlmodelc",
            "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
        ] {
            fs::create_dir(temp.path().join(name)).unwrap();
        }

        let error = validate_layout_assets(temp.path(), &inference).unwrap_err();
        assert!(error.to_string().contains("wespeaker-multimask-tail-b32"));

        fs::create_dir(temp.path().join("wespeaker-multimask-tail-b32.mlmodelc")).unwrap();
        validate_layout_assets(temp.path(), &inference).unwrap();
    }

    #[test]
    fn archived_stride_validation_does_not_require_removed_assets() {
        let temp = tempfile::tempdir().unwrap();
        let mut inference = valid_spec().inference;
        inference.layout = InferenceLayout::AlignedS13;

        validate_layout_assets(temp.path(), &inference).unwrap();
    }
}
