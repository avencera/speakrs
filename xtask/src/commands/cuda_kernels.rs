//! Build and check committed PTX and per-architecture cubins for the CUDA backend
//!
//! The kernel crate `crates/speakrs-cuda-kernels` needs a pinned nightly and CUDA 13,
//! so its PTX is generated on a GPU box and committed under
//! `crates/speakrs-ptx-<tier>/ptx`, one published crate per tier, with per-area
//! manifests in `src/inference/cuda/ptx`. The kernel crate's own docs still name the
//! old directory; editing them would change every manifest's source hash. The
//! machinery lives in `cudakit-build`; this module only declares speakrs's project

use color_eyre::eyre::Result;
use cudakit_build::{Area, CubinArch, Pins, Project, Tier, TierCrate};

use crate::cmd::project_root;

/// speakrs's kernel areas, artifact layout and toolchain pins
pub const PROJECT: Project = Project {
    name: "speakrs",
    command: "cargo xtask cuda-kernels",
    kernel_crate: "crates/speakrs-cuda-kernels",
    kernel_package: "speakrs-cuda-kernels",
    manifest_dir: "src/inference/cuda/ptx",
    host_kernels: "src/inference/cuda/kernels.rs",
    host_feature: "cuda-{tier}",
    targets: &[Tier::Sm75, Tier::Sm80, Tier::Sm90, Tier::Sm120],
    tier_crate: TierCrate {
        path: "crates/speakrs-ptx-{tier}",
        description: "Prebuilt {tier} tier CUDA kernels for speakrs

speakrs embeds these PTX and cubin files through its `cuda-*` features. Each tier
ships in its own crate so every published crate stays under the crates.io upload
limit. This crate has no stable API of its own",
        example_area: "fbank",
    },
    areas: AREAS,
    // speakrs pins its own toolchain so a cudakit-build default bump never makes these
    // artifacts stale; keep the rev in sync with crates/speakrs-cuda-kernels/Cargo.toml
    pins: Pins {
        cuda_oxide_rev: "918bbde123671153b29a89542f1781f1c7494c14",
        cuda_oxide_nightly: "nightly-2026-08-28",
        // driver 580 JITs at most PTX ISA 9.0
        max_ptx_isa: (9, 0),
        ptxas: "Cuda compilation tools, release 13.0, V13.0.88",
    },
    // explicit for the same reason as the pins: the cubin matrix, tier crates and host
    // masks in src/inference/cuda/kernels.rs all encode exactly this list
    cubin_arches: &[
        CubinArch(75),
        CubinArch(80),
        CubinArch(86),
        CubinArch(89),
        CubinArch(90),
        CubinArch(120),
    ],
};

/// Kernel areas and the tiers each one ships
///
/// A tier above the baseline needs a measured win on the GPU box and parity with the
/// baseline. Adding one also needs its feature-masked `tier_ptx!` in
/// `src/inference/cuda/kernels.rs`, which `check` verifies
const AREAS: &[Area] = &[
    // the sm80 probe variant is the same kernel; it proves the runtime dispatch
    Area::new("probe", &[Tier::Sm75, Tier::Sm80]),
    Area::new("fbank", &[Tier::Sm75]),
    Area::new("embedding", &[Tier::Sm75]),
    Area::new("segmentation", &[Tier::Sm75]),
    // candidate areas remain separate from the Library-owned areas above
    // the capability 12.0 binding pins the sm75 bytes; sm80 adds the TF32 tensor-core
    // entries that only the A100 selection plans
    Area::pinned_baseline("resnet", &[Tier::Sm75, Tier::Sm80]),
    Area::new("lstm", &[Tier::Sm75]),
    Area::new("sincnet", &[Tier::Sm75]),
    Area::new("segdense", &[Tier::Sm75, Tier::Sm80]),
    Area::new("fbankdft", &[Tier::Sm75]),
    // the driver-only LSTM stack; sm80 adds TF32 tensor projections, and sm120 is the
    // same source built for the newest target
    Area::new("lstmproj", &[Tier::Sm75, Tier::Sm80, Tier::Sm120]),
    // the sm80 variant adds the tensor-core kernels; sm75 traps in those entries. sm120
    // pairs the FP16 trunk kernels' taps into m16n8k16 products and adds FP16 Winograd
    // entries the lower tiers lack, and only the RTX 5090 binding selects it
    Area::bound_top("wideconv", &[Tier::Sm75, Tier::Sm80, Tier::Sm120]),
];

/// Regenerate PTX and cubins for the given areas, or every area when none are given
pub fn build(areas: &[String]) -> Result<()> {
    PROJECT.build(&project_root(), areas)
}

/// Build cubins only from committed PTX, without cuda-oxide
pub fn build_cubins(areas: &[String]) -> Result<()> {
    PROJECT.build_cubins(&project_root(), areas)
}

/// Check committed PTX and cubins against sources, hashes and pins
pub fn check(rebuild: bool) -> Result<()> {
    PROJECT.check(&project_root(), rebuild)
}
