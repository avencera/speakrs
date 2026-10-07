//! cuda-oxide device kernels for the speakrs native CUDA backend
//!
//! Each area is a module behind its own feature. `cargo xtask cuda-kernels build`
//! compiles the crate once per area feature and writes
//! `src/inference/cuda/ptx/<area>.ptx` in the speakrs tree, so PTX entry names only
//! have to be unique within an area. Prefix them with the area name anyway, because
//! the host looks kernels up by that name
//!
//! An area may also ship variants for newer GPUs. The baseline variant targets
//! `sm_75` with no tier feature, and a higher variant turns on `tier-sm80`,
//! `tier-sm90` or `tier-sm120`. Gate tier-specific code with
//! `#[cfg(feature = "tier-sm80")]`. Every variant of an area must export the same
//! kernels with the same parameters, because the host picks one variant at run time
//! and looks kernels up by name

#[cfg(feature = "embedding")]
pub mod embedding;
#[cfg(feature = "fbank")]
pub mod fbank;
#[cfg(feature = "lstm")]
pub mod lstm;
#[cfg(feature = "probe")]
pub mod probe;
#[cfg(feature = "resnet")]
pub mod resnet;
#[cfg(feature = "segmentation")]
pub mod segmentation;
#[cfg(feature = "sincnet")]
pub mod sincnet;
#[cfg(feature = "wideconv")]
pub mod wideconv;
