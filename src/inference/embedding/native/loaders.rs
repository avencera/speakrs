use std::path::{Path, PathBuf};
use std::sync::Arc;

use objc2_core_ml::MLComputeUnits;

use crate::inference::coreml::{CachedInputShape, CoreMlError, GpuPrecision, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError};
use crate::pipeline::RuntimeConfig;
#[cfg(feature = "_metrics")]
use crate::pipeline::{CoreMlChunkLayout, CoreMlShapeLadder};

use super::super::paths::{
    fp32_coreml_path, split_fbank_batched_model_path, split_fbank_model_path, split_tail_model_path,
};
use super::super::{
    CHUNK_SPEAKER_BATCH_SIZE, ChunkEmbeddingSession, ChunkSessionSpec, FBANK_FEATURES, MASK_FRAMES,
    PRIMARY_BATCH_SIZE,
};

const CHUNK_WINDOW_FBANK_FRAMES: usize = 1000;
const ONE_SECOND_FBANK_FRAMES: usize = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChunkModelLayout {
    Aligned { step_resnet_frames: usize },
    OneSecondPhased,
}

impl ChunkModelLayout {
    fn model_suffix(self) -> String {
        match self {
            Self::Aligned { step_resnet_frames } => format!("s{step_resnet_frames}"),
            Self::OneSecondPhased => "p1s".to_owned(),
        }
    }

    const fn fbank_frames(self, num_windows: usize) -> usize {
        let window_steps = num_windows - 1;
        match self {
            Self::Aligned { step_resnet_frames } => {
                window_steps * step_resnet_frames * 8 + CHUNK_WINDOW_FBANK_FRAMES
            }
            Self::OneSecondPhased => {
                window_steps * ONE_SECOND_FBANK_FRAMES + CHUNK_WINDOW_FBANK_FRAMES
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkSessionConfig {
    layout: ChunkModelLayout,
    num_windows: usize,
}

impl ChunkSessionConfig {
    const fn aligned(num_windows: usize, step_resnet_frames: usize) -> Self {
        Self {
            layout: ChunkModelLayout::Aligned { step_resnet_frames },
            num_windows,
        }
    }

    const fn one_second_phased(num_windows: usize) -> Self {
        Self {
            layout: ChunkModelLayout::OneSecondPhased,
            num_windows,
        }
    }

    fn model_stem(self) -> String {
        let suffix = self.layout.model_suffix();
        let num_windows = self.num_windows;
        format!("wespeaker-chunk-emb-{suffix}-w{num_windows}")
    }

    const fn fbank_frames(self) -> usize {
        self.layout.fbank_frames(self.num_windows)
    }

    const fn num_masks(self) -> usize {
        self.num_windows * 3
    }
}

const COREML_FAST_CHUNK_CONFIGS: &[ChunkSessionConfig] = &[
    ChunkSessionConfig::aligned(11, 25),
    ChunkSessionConfig::aligned(16, 25),
    ChunkSessionConfig::aligned(21, 25),
    ChunkSessionConfig::aligned(26, 25),
    ChunkSessionConfig::aligned(36, 25),
    ChunkSessionConfig::aligned(46, 25),
    ChunkSessionConfig::aligned(56, 25),
];

const COREML_CHUNK_CONFIGS: &[ChunkSessionConfig] = &[
    ChunkSessionConfig::one_second_phased(21),
    ChunkSessionConfig::one_second_phased(36),
    ChunkSessionConfig::one_second_phased(51),
    ChunkSessionConfig::one_second_phased(81),
    ChunkSessionConfig::one_second_phased(111),
];

#[cfg(feature = "_metrics")]
const COREML_REDUCED_CHUNK_CONFIGS: &[ChunkSessionConfig] = &[
    ChunkSessionConfig::one_second_phased(21),
    ChunkSessionConfig::one_second_phased(51),
    ChunkSessionConfig::one_second_phased(111),
];

/// Compiled CoreML embedding bundles derived from the base embedding ONNX path
///
/// Only file stems are used, so the ONNX files do not need to exist. Required bundles are
/// plain paths; optional ones are present only when the bundle exists
pub(super) struct CoreMlEmbeddingAssets {
    pub(super) fbank: PathBuf,
    pub(super) fbank_batched: PathBuf,
    pub(super) fbank_30s: Option<PathBuf>,
    pub(super) tail: PathBuf,
    pub(super) tail_batched: PathBuf,
    pub(super) tail_primary_batched: Option<PathBuf>,
    pub(super) multi_mask: PathBuf,
    pub(super) chunk_sessions: Vec<ChunkSessionSpec>,
}

impl CoreMlEmbeddingAssets {
    pub(super) fn resolve(
        model_path: &Path,
        mode: ExecutionMode,
        runtime: &RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        let fbank =
            require_native_asset(fp32_coreml_path(&split_fbank_model_path(model_path)), mode)?;
        let fbank_batched = require_native_asset(
            fp32_coreml_path(&split_fbank_batched_model_path(model_path)),
            mode,
        )?;
        let tail = require_native_asset(
            fp32_coreml_path(&split_tail_model_path(model_path, 1)),
            mode,
        )?;
        let tail_batched = require_native_asset(
            fp32_coreml_path(&split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE)),
            mode,
        )?;
        let multi_mask = require_native_asset(
            fp32_coreml_path(&model_path.with_file_name("wespeaker-multimask-tail-b32.onnx")),
            mode,
        )?;

        let fbank_30s = model_path.with_file_name("wespeaker-fbank-30s.mlmodelc");
        if runtime_uses_native_chunk_sessions(mode, runtime) {
            require_native_asset(fbank_30s.clone(), mode)?;

            for config in chunk_session_config(mode, runtime) {
                require_chunk_native_asset(model_path, *config, mode)?;
            }
        }

        Ok(Self {
            fbank,
            fbank_batched,
            fbank_30s: existing(fbank_30s),
            tail,
            tail_batched,
            tail_primary_batched: existing(fp32_coreml_path(&split_tail_model_path(
                model_path,
                PRIMARY_BATCH_SIZE,
            ))),
            multi_mask,
            chunk_sessions: chunk_session_specs(model_path, mode, runtime),
        })
    }
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.exists().then_some(path)
}

fn require_native_asset(path: PathBuf, mode: ExecutionMode) -> Result<PathBuf, ModelLoadError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(ModelLoadError::MissingNativeAsset { mode, path })
    }
}

fn chunk_session_config(
    mode: ExecutionMode,
    _runtime: &RuntimeConfig,
) -> &'static [ChunkSessionConfig] {
    #[cfg(feature = "_metrics")]
    if let Some(experiment) = _runtime.experiment {
        return match (experiment.coreml_chunk_layout(), experiment.shape_ladder()) {
            (CoreMlChunkLayout::OneSecondPhased, CoreMlShapeLadder::Full) => COREML_CHUNK_CONFIGS,
            (CoreMlChunkLayout::OneSecondPhased, CoreMlShapeLadder::Reduced) => {
                COREML_REDUCED_CHUNK_CONFIGS
            }
            (CoreMlChunkLayout::FastS25, CoreMlShapeLadder::Full) => COREML_FAST_CHUNK_CONFIGS,
            (CoreMlChunkLayout::PerWindow, CoreMlShapeLadder::Full) => &[],
            (_, CoreMlShapeLadder::Reduced) => &[],
        };
    }

    match mode {
        ExecutionMode::CoreMlFast => COREML_FAST_CHUNK_CONFIGS,
        ExecutionMode::CoreMl => COREML_CHUNK_CONFIGS,
        _ => &[],
    }
}

fn chunk_session_specs(
    model_path: &Path,
    mode: ExecutionMode,
    runtime: &RuntimeConfig,
) -> Vec<ChunkSessionSpec> {
    chunk_session_config(mode, runtime)
        .iter()
        .filter_map(|&config| {
            let coreml_path = chunk_native_asset_path(model_path, config)?;

            Some(ChunkSessionSpec {
                coreml_path,
                num_windows: config.num_windows,
                fbank_frames: config.fbank_frames(),
                num_masks: config.num_masks(),
            })
        })
        .collect()
}

pub(super) fn load_chunk_session(
    spec: &ChunkSessionSpec,
    compute_units: MLComputeUnits,
) -> Result<ChunkEmbeddingSession, CoreMlError> {
    let model = SharedCoreMlModel::load(
        &spec.coreml_path,
        compute_units,
        "output",
        GpuPrecision::Low,
    )?;
    Ok(ChunkEmbeddingSession {
        model: Arc::new(model),
        num_windows: spec.num_windows,
        fbank_frames: spec.fbank_frames,
        num_masks: spec.num_masks,
        cached_fbank_shape: Arc::new(CachedInputShape::new(
            "fbank",
            &[1, spec.fbank_frames, FBANK_FEATURES],
        )),
        cached_masks_shape: Arc::new(CachedInputShape::new(
            "masks",
            &[spec.num_masks, MASK_FRAMES],
        )),
    })
}

fn runtime_uses_native_chunk_sessions(mode: ExecutionMode, _runtime: &RuntimeConfig) -> bool {
    #[cfg(feature = "_metrics")]
    if let Some(experiment) = _runtime.experiment {
        return experiment
            .coreml_chunk_layout()
            .uses_native_chunk_sessions();
    }

    mode.is_coreml()
}

fn chunk_native_asset_path(model_path: &Path, config: ChunkSessionConfig) -> Option<PathBuf> {
    let stem = config.model_stem();
    let fp32_path = model_path.with_file_name(format!("{stem}.mlmodelc"));
    if fp32_path.exists() {
        return Some(fp32_path);
    }

    let w8a16_path = model_path.with_file_name(format!("{stem}-w8a16.mlmodelc"));
    w8a16_path.exists().then_some(w8a16_path)
}

fn require_chunk_native_asset(
    model_path: &Path,
    config: ChunkSessionConfig,
    mode: ExecutionMode,
) -> Result<(), ModelLoadError> {
    if chunk_native_asset_path(model_path, config).is_some() {
        return Ok(());
    }

    Err(ModelLoadError::MissingNativeAsset {
        mode,
        path: model_path.with_file_name(format!("{}.mlmodelc", config.model_stem())),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(prefix: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("speakrs-{prefix}-{unique}"));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn write_invalid_mlmodelc(&self, name: &str) {
            let bundle = self.path().join(name);
            fs::create_dir_all(bundle.join("weights")).unwrap();
            fs::create_dir_all(bundle.join("analytics")).unwrap();
            fs::write(bundle.join("model.mil"), b"invalid").unwrap();
            fs::write(bundle.join("coremldata.bin"), b"invalid").unwrap();
            fs::write(bundle.join("weights/weight.bin"), b"invalid").unwrap();
            fs::write(bundle.join("analytics/coremldata.bin"), b"invalid").unwrap();
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const REQUIRED_BUNDLES: &[&str] = &[
        "wespeaker-fbank.mlmodelc",
        "wespeaker-fbank-b32.mlmodelc",
        "wespeaker-voxceleb-resnet34-tail.mlmodelc",
        "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
        "wespeaker-multimask-tail-b32.mlmodelc",
    ];

    fn expect_missing(model_path: &Path) -> PathBuf {
        let result = CoreMlEmbeddingAssets::resolve(
            model_path,
            ExecutionMode::CoreMl,
            &RuntimeConfig::default(),
        );
        match result {
            Err(ModelLoadError::MissingNativeAsset {
                mode: ExecutionMode::CoreMl,
                path,
            }) => path,
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("missing bundles should error"),
        }
    }

    #[test]
    fn resolve_requires_bundles_without_onnx_files() {
        let dir = TestDir::new("emb-assets");
        // the ONNX files are never read in CoreML modes, so none exist here
        let model_path = dir.path().join("wespeaker-voxceleb-resnet34.onnx");

        assert_eq!(
            expect_missing(&model_path),
            dir.path().join("wespeaker-fbank.mlmodelc")
        );

        for bundle in REQUIRED_BUNDLES {
            dir.write_invalid_mlmodelc(bundle);
        }
        assert_eq!(
            expect_missing(&model_path),
            dir.path().join("wespeaker-fbank-30s.mlmodelc")
        );

        dir.write_invalid_mlmodelc("wespeaker-fbank-30s.mlmodelc");
        for config in COREML_CHUNK_CONFIGS {
            dir.write_invalid_mlmodelc(&format!("{}.mlmodelc", config.model_stem()));
        }
        let assets = CoreMlEmbeddingAssets::resolve(
            &model_path,
            ExecutionMode::CoreMl,
            &RuntimeConfig::default(),
        )
        .unwrap();

        assert_eq!(assets.fbank, dir.path().join("wespeaker-fbank.mlmodelc"));
        assert_eq!(
            assets.multi_mask,
            dir.path().join("wespeaker-multimask-tail-b32.mlmodelc")
        );
        assert!(assets.fbank_30s.is_some());
        assert!(assets.tail_primary_batched.is_none());
        assert_eq!(assets.chunk_sessions.len(), COREML_CHUNK_CONFIGS.len());
        assert!(!model_path.exists());
    }

    #[test]
    fn one_second_chunk_configs_derive_exact_shapes() {
        let configs: Vec<_> = COREML_CHUNK_CONFIGS
            .iter()
            .copied()
            .map(|config| {
                (
                    config.model_stem(),
                    config.fbank_frames(),
                    config.num_masks(),
                )
            })
            .collect();

        assert_eq!(
            configs,
            [
                ("wespeaker-chunk-emb-p1s-w21".to_owned(), 3000, 63),
                ("wespeaker-chunk-emb-p1s-w36".to_owned(), 4500, 108),
                ("wespeaker-chunk-emb-p1s-w51".to_owned(), 6000, 153),
                ("wespeaker-chunk-emb-p1s-w81".to_owned(), 9000, 243),
                ("wespeaker-chunk-emb-p1s-w111".to_owned(), 12000, 333),
            ]
        );
    }

    #[cfg(feature = "_metrics")]
    #[test]
    fn reduced_one_second_ladder_keeps_endpoint_and_middle_capacities() {
        let capacities: Vec<_> = COREML_REDUCED_CHUNK_CONFIGS
            .iter()
            .map(|config| config.num_windows)
            .collect();

        assert_eq!(capacities, [21, 51, 111]);
    }

    #[cfg(feature = "_metrics")]
    fn runtime_with_layout(layout: CoreMlChunkLayout) -> RuntimeConfig {
        RuntimeConfig {
            chunk_emb_compute_units: crate::inference::CoreMlComputeUnits::All,
            experiment: Some(crate::pipeline::ExperimentInferenceConfig::new(layout)),
            ..RuntimeConfig::default()
        }
    }

    #[cfg(feature = "_metrics")]
    #[test]
    fn experiment_layouts_select_fixed_chunk_tables() {
        type ExpectedSession<'a> = (&'a str, usize, usize);
        type LayoutCase<'a> = (CoreMlChunkLayout, ExecutionMode, &'a [ExpectedSession<'a>]);

        let cases: &[LayoutCase<'_>] = &[
            (
                CoreMlChunkLayout::OneSecondPhased,
                ExecutionMode::CoreMl,
                &[
                    ("wespeaker-chunk-emb-p1s-w21", 3000, 63),
                    ("wespeaker-chunk-emb-p1s-w36", 4500, 108),
                    ("wespeaker-chunk-emb-p1s-w51", 6000, 153),
                    ("wespeaker-chunk-emb-p1s-w81", 9000, 243),
                    ("wespeaker-chunk-emb-p1s-w111", 12000, 333),
                ],
            ),
            (
                CoreMlChunkLayout::FastS25,
                ExecutionMode::CoreMlFast,
                &[
                    ("wespeaker-chunk-emb-s25-w11", 3000, 33),
                    ("wespeaker-chunk-emb-s25-w16", 4000, 48),
                    ("wespeaker-chunk-emb-s25-w21", 5000, 63),
                    ("wespeaker-chunk-emb-s25-w26", 6000, 78),
                    ("wespeaker-chunk-emb-s25-w36", 8000, 108),
                    ("wespeaker-chunk-emb-s25-w46", 10000, 138),
                    ("wespeaker-chunk-emb-s25-w56", 12000, 168),
                ],
            ),
        ];

        for &(layout, mode, expected) in cases {
            let runtime = runtime_with_layout(layout);
            let actual: Vec<_> = chunk_session_config(mode, &runtime)
                .iter()
                .map(|config| {
                    (
                        config.model_stem(),
                        config.fbank_frames(),
                        config.num_masks(),
                    )
                })
                .collect();
            let expected: Vec<_> = expected
                .iter()
                .map(|(stem, fbank_frames, num_masks)| {
                    ((*stem).to_owned(), *fbank_frames, *num_masks)
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[cfg(feature = "_metrics")]
    #[test]
    fn per_window_layouts_select_no_chunk_sessions() {
        let runtime = runtime_with_layout(CoreMlChunkLayout::PerWindow);

        assert!(chunk_session_config(ExecutionMode::CoreMl, &runtime).is_empty());
        assert!(!runtime_uses_native_chunk_sessions(
            ExecutionMode::CoreMl,
            &runtime
        ));
    }
}
