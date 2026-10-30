//! Window-capacity ladders of the CoreML chunk embedding models
//!
//! The catalog derives the bundles each mode downloads from these ladders, and the CoreML
//! loader derives each bundle's input shapes from the same entries, so a ladder change is
//! one edit

#[cfg(feature = "coreml")]
const CHUNK_WINDOW_FBANK_FRAMES: usize = 1000;
#[cfg(feature = "coreml")]
const ONE_SECOND_FBANK_FRAMES: usize = 100;

/// How a chunk model spaces its embedding windows over the filterbank
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

    #[cfg(feature = "coreml")]
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

/// One fixed-capacity chunk embedding model
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ChunkSessionConfig {
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

    /// Bundle name without the `.mlmodelc` or precision suffix
    pub(crate) fn model_stem(self) -> String {
        let suffix = self.layout.model_suffix();
        let num_windows = self.num_windows;
        format!("wespeaker-chunk-emb-{suffix}-w{num_windows}")
    }

    /// Embedding windows one call covers
    #[cfg(feature = "coreml")]
    pub(crate) const fn num_windows(self) -> usize {
        self.num_windows
    }

    #[cfg(feature = "coreml")]
    pub(crate) const fn fbank_frames(self) -> usize {
        self.layout.fbank_frames(self.num_windows)
    }

    #[cfg(feature = "coreml")]
    pub(crate) const fn num_masks(self) -> usize {
        self.num_windows * 3
    }
}

/// The `coreml-fast` ladder, 25 ResNet frames between windows
pub(crate) const COREML_FAST_CHUNK_CONFIGS: &[ChunkSessionConfig] = &[
    ChunkSessionConfig::aligned(11, 25),
    ChunkSessionConfig::aligned(16, 25),
    ChunkSessionConfig::aligned(21, 25),
    ChunkSessionConfig::aligned(26, 25),
    ChunkSessionConfig::aligned(36, 25),
    ChunkSessionConfig::aligned(46, 25),
    ChunkSessionConfig::aligned(56, 25),
];

/// The `coreml` ladder, one second between windows
pub(crate) const COREML_CHUNK_CONFIGS: &[ChunkSessionConfig] = &[
    ChunkSessionConfig::one_second_phased(21),
    ChunkSessionConfig::one_second_phased(36),
    ChunkSessionConfig::one_second_phased(51),
    ChunkSessionConfig::one_second_phased(81),
    ChunkSessionConfig::one_second_phased(111),
];

/// A subset of [`COREML_CHUNK_CONFIGS`] for shape-ladder experiments, so it downloads nothing
/// extra
#[cfg(all(feature = "coreml", feature = "_metrics"))]
pub(crate) const COREML_REDUCED_CHUNK_CONFIGS: &[ChunkSessionConfig] = &[
    ChunkSessionConfig::one_second_phased(21),
    ChunkSessionConfig::one_second_phased(51),
    ChunkSessionConfig::one_second_phased(111),
];
