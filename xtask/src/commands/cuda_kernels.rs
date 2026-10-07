//! Build and check committed PTX and per-architecture cubins for the CUDA backend
//!
//! The kernel crate `crates/speakrs-cuda-kernels` needs a pinned nightly and CUDA 13,
//! so its PTX is generated on a GPU box and committed under `src/inference/cuda/ptx`.
//! Every area ships an `sm75` baseline variant and may add higher tiers, written as
//! `<area>.<tier>.ptx`, plus one `<area>.manifest` that records, per variant, the
//! target, the hash of the sources that produced it and the hash of the PTX itself.
//! Cubins use the exact PTX bytes and pinned CUDA 13.0 ptxas, with one file per
//! compatible exact GPU capability. Manifests bind each cubin to its source PTX
//! Parallel work on different areas never edits the same generated file. Plain
//! `check` only hashes and parses files, so it runs anywhere

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::{Context, Result, bail, eyre};
use sha2::{Digest, Sha256};

use crate::cmd::{project_root, run_cmd};

mod ptx_lint;

/// A PTX target an area can ship
///
/// Only plain `sm_XY` targets: the driver JIT-compiles their PTX for every newer GPU,
/// while `a`-suffixed targets run on one exact architecture only
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Sm75,
    Sm80,
    Sm90,
    Sm120,
}

impl Tier {
    /// Every area ships this variant; it covers Turing and everything newer
    const BASELINE: Self = Self::Sm75;

    /// The name used in file names, manifests and `SPEAKRS_CUDA_PTX_TIER`
    const fn name(self) -> &'static str {
        match self {
            Self::Sm75 => "sm75",
            Self::Sm80 => "sm80",
            Self::Sm90 => "sm90",
            Self::Sm120 => "sm120",
        }
    }

    /// The cuda-oxide `--arch` and PTX `.target`
    const fn arch(self) -> &'static str {
        match self {
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm90 => "sm_90",
            Self::Sm120 => "sm_120",
        }
    }

    const fn capability(self) -> u16 {
        match self {
            Self::Sm75 => 75,
            Self::Sm80 => 80,
            Self::Sm90 => 90,
            Self::Sm120 => 120,
        }
    }

    /// The speakrs feature that selects this GPU tier
    fn host_feature(self) -> String {
        format!("cuda-{}", self.name())
    }

    /// The kernel-crate feature that turns on tier-specific code
    const fn feature(self) -> Option<&'static str> {
        match self {
            Self::Sm75 => None,
            Self::Sm80 => Some("tier-sm80"),
            Self::Sm90 => Some("tier-sm90"),
            Self::Sm120 => Some("tier-sm120"),
        }
    }

    /// Oldest PTX ISA that can target this architecture, used for stub modules
    const fn min_ptx_isa(self) -> (u32, u32) {
        match self {
            Self::Sm75 => (6, 3),
            Self::Sm80 => (7, 0),
            Self::Sm90 => (7, 8),
            Self::Sm120 => (8, 7),
        }
    }

    fn parse(name: &str) -> Option<Self> {
        [Self::Sm75, Self::Sm80, Self::Sm90, Self::Sm120]
            .into_iter()
            .find(|tier| tier.name() == name)
    }
}

/// A kernel area: one module and feature in the kernel crate, one PTX file per tier
#[derive(Debug, Clone, Copy)]
pub struct Area {
    name: &'static str,
    tiers: &'static [Tier],
}

impl Area {
    /// Rejects, at compile time, an area without the baseline or with unordered tiers
    const fn new(name: &'static str, tiers: &'static [Tier]) -> Self {
        assert!(
            !tiers.is_empty() && tiers[0] as u8 == Tier::BASELINE as u8,
            "every kernel area must ship the sm75 baseline as its first tier"
        );
        let mut index = 1;
        while index < tiers.len() {
            assert!(
                (tiers[index - 1] as u8) < tiers[index] as u8,
                "kernel area tiers must be strictly ascending"
            );
            index += 1;
        }

        Self { name, tiers }
    }

    fn variants(self) -> impl Iterator<Item = Variant> {
        self.tiers
            .iter()
            .map(move |&tier| Variant { area: self, tier })
    }
}

/// Kernel areas and the tiers each one ships
///
/// A tier above the baseline needs a measured win on the GPU box and parity with the
/// baseline. Adding one also needs its feature-masked `tier_ptx!` in
/// `src/inference/cuda/kernels.rs`, which `check` verifies
pub const AREAS: &[Area] = &[
    // the sm80 probe variant is the same kernel; it proves the runtime dispatch
    Area::new("probe", &[Tier::Sm75, Tier::Sm80]),
    Area::new("fbank", &[Tier::Sm75]),
    Area::new("embedding", &[Tier::Sm75]),
    Area::new("segmentation", &[Tier::Sm75]),
    // candidate areas: kernels that may replace a library call, qualified by
    // `cargo xtask cuda-qualify` and kept apart from the Library-owned areas above
    Area::new("resnet", &[Tier::Sm75]),
    Area::new("lstm", &[Tier::Sm75]),
    Area::new("sincnet", &[Tier::Sm75]),
    Area::new("fbankdft", &[Tier::Sm75]),
];

/// One PTX file: an area built for one tier
#[derive(Debug, Clone, Copy)]
struct Variant {
    area: Area,
    tier: Tier,
}

impl Variant {
    fn file_name(self) -> String {
        format!("{}.{}.ptx", self.area.name, self.tier.name())
    }

    fn cubins(self) -> impl Iterator<Item = Cubin> {
        CUBIN_ARCHES
            .iter()
            .copied()
            .filter(move |arch| arch.0 >= self.tier.capability())
            .map(move |arch| Cubin {
                variant: self,
                arch,
            })
    }

    fn features(self) -> String {
        match self.tier.feature() {
            Some(tier) => format!("{},{tier}", self.area.name),
            None => self.area.name.to_string(),
        }
    }
}

/// Exact GPU capabilities for which ready SASS is shipped
const CUBIN_ARCHES: &[CubinArch] = &[
    CubinArch(75),
    CubinArch(80),
    CubinArch(86),
    CubinArch(89),
    CubinArch(90),
    CubinArch(120),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CubinArch(u16);

impl CubinArch {
    fn name(self) -> String {
        format!("sm_{}", self.0)
    }

    fn parse(name: &str) -> Option<Self> {
        CUBIN_ARCHES
            .iter()
            .copied()
            .find(|arch| arch.name() == name)
    }
}

#[derive(Debug, Clone, Copy)]
struct Cubin {
    variant: Variant,
    arch: CubinArch,
}

impl Cubin {
    fn file_name(self) -> String {
        format!(
            "{}.{}.{}.cubin",
            self.variant.area.name,
            self.variant.tier.name(),
            self.arch.name()
        )
    }
}

/// Pin the CUDA 13.0 patch release as well as the release for byte identity
const PTXAS_VERSION: &str = "Cuda compilation tools, release 13.0, V13.0.88";
/// Exact argv template: default optimization, no debug or line information
const PTXAS_FLAGS: &str = "-arch={arch} {input} -o {output}";

struct Ptxas(PathBuf);

impl Ptxas {
    fn pinned() -> Result<Self> {
        let toolkit = std::env::var_os("CUDA13_HOME")
            .or_else(|| std::env::var_os("CUDA_TOOLKIT_PATH"))
            .ok_or_else(|| eyre!("set CUDA13_HOME or CUDA_TOOLKIT_PATH to CUDA 13.0; ptxas is never taken from PATH"))?;
        let path = PathBuf::from(toolkit).join("bin/ptxas");
        let output = Command::new(&path)
            .arg("--version")
            .output()
            .wrap_err_with(|| format!("running {} --version", path.display()))?;
        let version = String::from_utf8(output.stdout)?;
        if !output.status.success() || !version.lines().any(|line| line == PTXAS_VERSION) {
            bail!(
                "wrong ptxas version from {}: expected `{PTXAS_VERSION}`, got `{}`",
                path.display(),
                version.trim()
            );
        }

        Ok(Self(path))
    }

    fn compile(&self, ptx_dir: &Path, out_dir: &Path, cubin: Cubin) -> Result<()> {
        run_cmd(
            Command::new(&self.0)
                .arg(format!("-arch={}", cubin.arch.name()))
                .arg(ptx_dir.join(cubin.variant.file_name()))
                .arg("-o")
                .arg(out_dir.join(cubin.file_name())),
        )
        .wrap_err_with(|| format!("building {} from committed PTX", cubin.file_name()))
    }
}

/// cuda-oxide commit; keep in sync with the kernel crate's Cargo.toml and
/// scripts/cuda/setup-gpu-box.sh
const CUDA_OXIDE_REV: &str = "918bbde123671153b29a89542f1781f1c7494c14";
/// Toolchain pinned by that cuda-oxide commit
const CUDA_OXIDE_NIGHTLY: &str = "nightly-2026-08-28";
/// Newest PTX ISA the oldest supported driver can JIT: CUDA 13.0 drivers (580.x)
/// accept PTX ISA 9.0, and a newer `.version` fails to load there
const MAX_PTX_ISA: (u32, u32) = (9, 0);

const KERNEL_CRATE: &str = "crates/speakrs-cuda-kernels";
const PTX_DIR: &str = "src/inference/cuda/ptx";
/// The host file that embeds each variant for its selected GPU tiers
const HOST_KERNELS: &str = "src/inference/cuda/kernels.rs";
/// Name cuda-oxide gives the PTX of the kernel crate
const OXIDE_PTX_NAME: &str = "speakrs_cuda_kernels.ptx";

/// Regenerate PTX for every variant of the given areas, or of every area when none
/// are given, then build their cubins with the pinned ptxas
pub fn build(areas: &[String]) -> Result<()> {
    let root = project_root();
    let areas = selected_areas(areas)?;
    let ptxas = Ptxas::pinned()?;
    let crate_dir = root.join(KERNEL_CRATE);
    let ptx_dir = root.join(PTX_DIR);
    fs::create_dir_all(&ptx_dir)?;

    for area in areas {
        build_area(&crate_dir, &ptx_dir, area)?;
        build_area_cubins(&crate_dir, &ptx_dir, area, &ptxas)?;
    }

    Ok(())
}

fn build_area(crate_dir: &Path, ptx_dir: &Path, area: Area) -> Result<()> {
    let mut built = Vec::new();
    for variant in area.variants() {
        let ptx = compile_variant(crate_dir, variant)?;
        let version = check_ptx_header(variant, &ptx)?;
        ptx_lint::check_shared_truncation(&ptx)?;
        built.push((variant, ptx, version));
    }

    // compare before writing anything, so a mismatch leaves the committed files alone
    let modules: Vec<_> = built
        .iter()
        .map(|(variant, ptx, _)| (*variant, ptx.as_str()))
        .collect();
    check_entry_points(&modules)?;

    remove_undeclared_variants(ptx_dir, area)?;
    let mut manifest = Manifest::default();
    for (variant, ptx, version) in built {
        fs::write(ptx_dir.join(variant.file_name()), &ptx)?;
        manifest.variants.push(ManifestVariant {
            tier: variant.tier,
            sources: sources_hash(crate_dir, variant)?,
            ptx: sha256_hex(ptx.as_bytes()),
            cubins: Vec::new(),
        });
        println!(
            "{}: wrote {PTX_DIR}/{} (PTX ISA {}.{}, {})",
            area.name,
            variant.file_name(),
            version.0,
            version.1,
            variant.tier.arch()
        );
    }

    fs::write(
        ptx_dir.join(format!("{}.manifest", area.name)),
        manifest.render(),
    )?;
    Ok(())
}

/// Build only cubins from the exact PTX bytes already in the committed directory
///
/// Requires the pinned CUDA 13.0 ptxas, but never invokes cuda-oxide
pub fn build_cubins(areas: &[String]) -> Result<()> {
    let root = project_root();
    let areas = selected_areas(areas)?;
    let ptxas = Ptxas::pinned()?;
    for area in areas {
        build_area_cubins(&root.join(KERNEL_CRATE), &root.join(PTX_DIR), area, &ptxas)?;
    }

    Ok(())
}

fn build_area_cubins(crate_dir: &Path, ptx_dir: &Path, area: Area, ptxas: &Ptxas) -> Result<()> {
    let mut manifest = read_manifest(ptx_dir, area)?;
    check_area_ptx(crate_dir, ptx_dir, area, &manifest)?;
    // keep temporary writes inside the checkout, including on the rented box
    let out = tempfile::tempdir_in(ptx_dir)?;
    for (variant, entry) in area.variants().zip(&mut manifest.variants) {
        entry.cubins.clear();
        for cubin in variant.cubins() {
            ptxas.compile(ptx_dir, out.path(), cubin)?;
            entry.cubins.push(ManifestCubin {
                arch: cubin.arch,
                sha256: sha256_hex(&fs::read(out.path().join(cubin.file_name()))?),
                ptx: entry.ptx.clone(),
            });
        }
    }

    // finish the whole area before replacing artifacts or their manifest
    for cubin in area.variants().flat_map(Variant::cubins) {
        fs::copy(
            out.path().join(cubin.file_name()),
            ptx_dir.join(cubin.file_name()),
        )?;
    }

    remove_undeclared_variants(ptx_dir, area)?;
    manifest.ptxas = Some(PTXAS_VERSION.into());
    manifest.ptxas_flags = Some(PTXAS_FLAGS.into());
    fs::write(
        ptx_dir.join(format!("{}.manifest", area.name)),
        manifest.render(),
    )?;
    println!("{}: built cubins from committed PTX", area.name);
    Ok(())
}

fn rebuild_cubins(ptx_dir: &Path) -> Result<()> {
    let ptxas = Ptxas::pinned()?;
    let out = tempfile::tempdir_in(ptx_dir)?;
    for area in AREAS {
        for cubin in area.variants().flat_map(Variant::cubins) {
            ptxas.compile(ptx_dir, out.path(), cubin)?;
            if fs::read(out.path().join(cubin.file_name()))?
                != fs::read(ptx_dir.join(cubin.file_name()))?
            {
                bail!(
                    "{} rebuild is not byte-identical to the committed cubin",
                    cubin.file_name()
                );
            }
        }

        println!("{}: rebuilt cubins are byte-identical", area.name);
    }

    Ok(())
}

/// Fail when any committed PTX is stale, edited by hand, missing, not embedded by the
/// host, or exports different kernels than the other variants of its area
///
/// Also checks the complete cubin matrix, hashes and build pins without a toolkit
/// With `rebuild`, requires pinned ptxas and byte-identical rebuilt cubins
pub fn check(rebuild: bool) -> Result<()> {
    let root = project_root();
    let crate_dir = root.join(KERNEL_CRATE);
    let ptx_dir = root.join(PTX_DIR);
    let mut problems = Vec::new();

    for area in AREAS {
        if let Err(error) = check_area(&crate_dir, &ptx_dir, *area) {
            problems.push(format!("{}: {error:#}", area.name));
        }
    }

    problems.extend(unexpected_ptx_files(&ptx_dir)?);
    problems.extend(host_embed_problems(&root.join(HOST_KERNELS))?);
    if !problems.is_empty() {
        bail!(
            "committed CUDA PTX or cubins failed checks; use `cargo xtask cuda-kernels build` for PTX or `build-cubins` for cubins on the GPU box\n  {}",
            problems.join("\n  ")
        );
    }

    let variants: Vec<_> = AREAS
        .iter()
        .flat_map(|area| area.variants())
        .map(|variant| format!("{}.{}", variant.area.name, variant.tier.name()))
        .collect();
    if rebuild {
        rebuild_cubins(&ptx_dir)?;
    }

    println!(
        "CUDA PTX and cubins are up to date for {}",
        variants.join(", ")
    );
    Ok(())
}

fn selected_areas(areas: &[String]) -> Result<Vec<Area>> {
    if areas.is_empty() {
        return Ok(AREAS.to_vec());
    }

    areas
        .iter()
        .map(|name| {
            AREAS
                .iter()
                .copied()
                .find(|area| area.name == name)
                .ok_or_else(|| eyre!("unknown kernel area `{name}`; known: {}", area_names()))
        })
        .collect()
}

fn area_names() -> String {
    AREAS
        .iter()
        .map(|area| area.name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn compile_variant(crate_dir: &Path, variant: Variant) -> Result<String> {
    let out_dir = tempfile::tempdir()?;
    // a shared CARGO_TARGET_DIR is built by the stable toolchain; keep the nightly
    // kernel build in its own subdirectory so the two never invalidate each other
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(|dir| PathBuf::from(dir).join("cuda-kernels"))
        .unwrap_or_else(|| crate_dir.join("target"));

    // cuda-oxide writes PTX only while rustc compiles the crate, and Cargo skips a
    // crate it considers fresh, so clean it first to always get a new PTX file
    run_cmd(nightly_cargo(crate_dir, &target_dir).args(["clean", "-p", "speakrs-cuda-kernels"]))?;
    let mut cmd = nightly_cargo(crate_dir, &target_dir);
    cmd.env("CUDA_OXIDE_PTX_DIR", out_dir.path())
        .args([
            "oxide",
            "build",
            "--arch",
            variant.tier.arch(),
            "--features",
        ])
        .arg(variant.features());
    run_cmd(&mut cmd).wrap_err_with(|| {
        format!(
            "building the `{}` kernels for {}",
            variant.area.name,
            variant.tier.arch()
        )
    })?;

    let ptx_path = out_dir.path().join(OXIDE_PTX_NAME);
    if ptx_path.exists() {
        return Ok(fs::read_to_string(&ptx_path)?);
    }

    // a crate without device code produces no PTX at all
    if area_has_kernels(crate_dir, variant.area.name)? {
        bail!(
            "cuda-oxide did not write {} for `{}`",
            ptx_path.display(),
            variant.file_name()
        );
    }

    Ok(empty_module_ptx(variant))
}

/// Cargo pinned to the cuda-oxide nightly, independent of the toolchain running xtask
fn nightly_cargo(crate_dir: &Path, target_dir: &Path) -> Command {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(crate_dir)
        // `cargo xtask` runs under the stable toolchain, and rustup would pass that
        // choice down through these variables instead of the pinned nightly
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("CARGO")
        .env_remove("RUSTC")
        .env("CARGO_TARGET_DIR", target_dir)
        .arg(format!("+{CUDA_OXIDE_NIGHTLY}"));
    cmd
}

fn area_has_kernels(crate_dir: &Path, area: &str) -> Result<bool> {
    let mut files = Vec::new();
    collect_files(crate_dir, &crate_dir.join("src"), &mut files)?;
    for relative in files
        .iter()
        .filter(|relative| area_of(relative) == Some(area))
    {
        if fs::read_to_string(crate_dir.join(relative))?.contains("#[kernel]") {
            return Ok(true);
        }
    }

    Ok(false)
}

/// A loadable module with no entries, so the host can embed and load every area
/// before the area has kernels
fn empty_module_ptx(variant: Variant) -> String {
    let (major, minor) = variant.tier.min_ptx_isa();
    format!(
        "//\n// speakrs: the `{}` area has no kernels yet\n//\n\n.version {major}.{minor}\n.target {}\n.address_size 64\n",
        variant.area.name,
        variant.tier.arch()
    )
}

/// Deletes variant files of `area` that it no longer declares, including the
/// untiered `<area>.ptx` from before tiers existed
fn remove_undeclared_variants(ptx_dir: &Path, area: Area) -> Result<()> {
    let declared: BTreeSet<_> = area
        .variants()
        .flat_map(|variant| {
            std::iter::once(variant.file_name()).chain(variant.cubins().map(Cubin::file_name))
        })
        .collect();
    for entry in fs::read_dir(ptx_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        let ours = (name.ends_with(".ptx") || name.ends_with(".cubin"))
            && file_area(name) == Some(area.name);
        if ours && !declared.contains(name) {
            fs::remove_file(&path)?;
            println!("{}: removed undeclared {PTX_DIR}/{name}", area.name);
        }
    }

    Ok(())
}

fn read_manifest(ptx_dir: &Path, area: Area) -> Result<Manifest> {
    let manifest_path = ptx_dir.join(format!("{}.manifest", area.name));
    fs::read_to_string(&manifest_path)
        .map_err(|_| eyre!("missing {}", manifest_path.display()))
        .and_then(|text| Manifest::parse(&text))
}

fn check_area(crate_dir: &Path, ptx_dir: &Path, area: Area) -> Result<()> {
    let manifest = read_manifest(ptx_dir, area)?;
    check_area_ptx(crate_dir, ptx_dir, area, &manifest)?;
    check_area_cubins(ptx_dir, area, &manifest)
}

fn check_area_cubins(ptx_dir: &Path, area: Area, manifest: &Manifest) -> Result<()> {
    if manifest.ptxas.as_deref() != Some(PTXAS_VERSION) {
        bail!(
            "wrong ptxas version in manifest: expected `{PTXAS_VERSION}`, got {:?}",
            manifest.ptxas
        );
    }

    if manifest.ptxas_flags.as_deref() != Some(PTXAS_FLAGS) {
        bail!("wrong ptxas flags in manifest: expected `{PTXAS_FLAGS}`");
    }

    for (variant, entry) in area.variants().zip(&manifest.variants) {
        let expected: Vec<_> = variant.cubins().map(|cubin| cubin.arch).collect();
        let actual: Vec<_> = entry.cubins.iter().map(|cubin| cubin.arch).collect();
        if expected != actual {
            bail!(
                "{} manifest cubin arches {:?}, expected {:?}",
                variant.file_name(),
                actual,
                expected
            );
        }

        for record in &entry.cubins {
            let cubin = Cubin {
                variant,
                arch: record.arch,
            };
            let name = cubin.file_name();
            if record.ptx != entry.ptx {
                bail!("{name} PTX hash does not match its section's ptx hash");
            }

            let bytes =
                fs::read(ptx_dir.join(&name)).wrap_err_with(|| format!("missing cubin {name}"))?;
            if sha256_hex(&bytes) != record.sha256 {
                bail!("{name} cubin sha256 does not match its manifest");
            }
        }
    }

    Ok(())
}

fn check_area_ptx(crate_dir: &Path, ptx_dir: &Path, area: Area, manifest: &Manifest) -> Result<()> {
    let listed: Vec<_> = manifest.variants.iter().map(|entry| entry.tier).collect();
    if listed != area.tiers {
        bail!(
            "manifest lists tiers {}, but the area declares {}",
            tier_names(&listed),
            tier_names(area.tiers)
        );
    }

    let mut modules = Vec::new();
    for (variant, entry) in area.variants().zip(&manifest.variants) {
        let ptx_path = ptx_dir.join(variant.file_name());
        let ptx =
            fs::read_to_string(&ptx_path).map_err(|_| eyre!("missing {}", ptx_path.display()))?;

        if sources_hash(crate_dir, variant)? != entry.sources {
            bail!(
                "kernel sources or build pins changed since {} was generated",
                variant.file_name()
            );
        }

        if sha256_hex(ptx.as_bytes()) != entry.ptx {
            bail!(
                "{} does not match its manifest; regenerate it instead of editing it",
                variant.file_name()
            );
        }

        check_ptx_header(variant, &ptx)?;
        ptx_lint::check_shared_truncation(&ptx)
            .wrap_err_with(|| format!("linting {}", variant.file_name()))?;
        modules.push((variant, ptx));
    }

    let modules: Vec<_> = modules
        .iter()
        .map(|(variant, ptx)| (*variant, ptx.as_str()))
        .collect();
    check_entry_points(&modules)
}

fn tier_names(tiers: &[Tier]) -> String {
    let names: Vec<_> = tiers.iter().map(|tier| tier.name()).collect();
    format!("[{}]", names.join(", "))
}

/// The area a PTX directory file belongs to, from its first dot-separated part
fn file_area(name: &str) -> Option<&'static str> {
    let stem = name.split('.').next()?;
    AREAS
        .iter()
        .map(|area| area.name)
        .find(|area| *area == stem)
}

fn unexpected_ptx_files(ptx_dir: &Path) -> Result<Vec<String>> {
    let expected: BTreeSet<String> = AREAS
        .iter()
        .flat_map(|area| {
            area.variants()
                .flat_map(|variant| {
                    std::iter::once(variant.file_name())
                        .chain(variant.cubins().map(Cubin::file_name))
                })
                .chain([format!("{}.manifest", area.name)])
        })
        .collect();

    let mut problems = Vec::new();
    for entry in fs::read_dir(ptx_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        if !expected.contains(name) {
            problems.push(format!(
                "{} is not a declared PTX, cubin or manifest; declare its tier in AREAS or delete the file",
                path.display()
            ));
        }
    }

    Ok(problems)
}

/// The host must embed each declared variant only for tiers that select it
fn host_embed_problems(host_kernels: &Path) -> Result<Vec<String>> {
    let source = fs::read_to_string(host_kernels)
        .wrap_err_with(|| format!("reading {}", host_kernels.display()))?;
    Ok(host_embed_source_problems(&source))
}

fn host_embed_source_problems(source: &str) -> Vec<String> {
    let embedded = embedded_ptx_files(source);
    let declared: BTreeMap<String, Vec<String>> = AREAS
        .iter()
        .flat_map(|area| {
            area.variants().map(|variant| {
                // each GPU tier uses the newest variant that this area ships
                let features = [Tier::Sm75, Tier::Sm80, Tier::Sm90, Tier::Sm120]
                    .into_iter()
                    .filter(|tier| {
                        area.tiers.iter().rev().find(|shipped| **shipped <= *tier)
                            == Some(&variant.tier)
                    })
                    .map(Tier::host_feature)
                    .collect();
                (variant.file_name(), features)
            })
        })
        .collect();

    let mut problems = Vec::new();
    for (name, expected) in &declared {
        let Some(uses) = embedded.get(name) else {
            problems.push(format!("{HOST_KERNELS} does not embed ptx/{name}"));
            continue;
        };

        if uses.len() != 1 {
            problems.push(format!("{HOST_KERNELS} embeds ptx/{name} more than once"));
        }

        for features in uses {
            let mut actual = features.clone().unwrap_or_default();
            actual.sort();
            let mut expected = expected.clone();
            expected.sort();
            if features.is_none() || actual != expected {
                problems.push(format!(
                    "{HOST_KERNELS} embeds ptx/{name} with {features:?}, expected {expected:?}"
                ));
            }
        }
    }

    // binary masks share the PTX invocation; an omitted or foreign architecture
    // would silently force JIT and invalidate production artifact matching
    for (start, _) in source.match_indices("tier_ptx!(") {
        let Some((args, _)) = source[start + "tier_ptx!(".len()..].split_once(')') else {
            continue;
        };
        let Some((_, tail)) = args.split_once(']') else {
            continue;
        };
        let Some(stem) = tail
            .split('"')
            .nth(1)
            .and_then(|path| path.strip_prefix("ptx/"))
        else {
            continue;
        };
        // old syntax occurs only in parser-negative fixtures
        if stem.ends_with(".ptx") {
            continue;
        }
        let Some(variant) = AREAS
            .iter()
            .flat_map(|area| area.variants())
            .find(|variant| variant.file_name() == format!("{stem}.ptx"))
        else {
            continue;
        };
        let arches = tail
            .rsplit_once('[')
            .and_then(|(_, list)| list.split_once(']'))
            .map(|(list, _)| {
                list.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::parse::<u16>)
                    .collect::<std::result::Result<Vec<_>, _>>()
            });
        let expected: Vec<_> = variant.cubins().map(|cubin| cubin.arch.0).collect();
        if arches != Some(Ok(expected)) {
            problems.push(format!(
                "{HOST_KERNELS} cubin architecture mask differs for {stem}"
            ));
        }
    }

    for name in embedded.keys().filter(|name| !declared.contains_key(*name)) {
        problems.push(format!(
            "{HOST_KERNELS} embeds ptx/{name}, which no area declares"
        ));
    }

    problems
}

/// Every PTX inclusion, preserving duplicate calls and feature names for validation
fn embedded_ptx_files(source: &str) -> BTreeMap<String, Vec<Option<Vec<String>>>> {
    const PLAIN: &str = "include_str!(";
    const TIERED: &str = "tier_ptx!(";
    let mut files: BTreeMap<String, Vec<Option<Vec<String>>>> = BTreeMap::new();
    for (start, _) in source.match_indices(PLAIN) {
        let Some(rest) = source[start + PLAIN.len()..]
            .trim_start()
            .strip_prefix("\"ptx/")
        else {
            continue;
        };

        if let Some((name, _)) = rest.split_once('"')
            && name.ends_with(".ptx")
        {
            files.entry(name.to_string()).or_default().push(None);
        }
    }

    for (start, _) in source.match_indices(TIERED) {
        let Some((args, _)) = source[start + TIERED.len()..].split_once(')') else {
            continue;
        };

        let Some((mask, tail)) = args.split_once(']') else {
            continue;
        };
        let mask = format!("{mask}]");
        let Some(path) = tail.split('"').nth(1) else {
            continue;
        };
        let Some(stem) = path.strip_prefix("ptx/") else {
            continue;
        };
        let name = if stem.ends_with(".ptx") {
            stem.to_string()
        } else {
            format!("{stem}.ptx")
        };

        let features = mask
            .trim()
            .strip_prefix('[')
            .and_then(|mask| mask.strip_suffix(']'))
            .and_then(|mask| {
                mask.split(',')
                    .map(str::trim)
                    .filter(|feature| !feature.is_empty())
                    .map(|feature| {
                        feature
                            .strip_prefix('"')
                            .and_then(|feature| feature.strip_suffix('"'))
                            .map(str::to_string)
                    })
                    .collect()
            });
        files.entry(name).or_default().push(features);
    }

    files
}

/// Checks `.version` and `.target` so a toolchain bump cannot silently produce PTX
/// that older drivers or GPUs reject
fn check_ptx_header(variant: Variant, ptx: &str) -> Result<(u32, u32)> {
    let file = variant.file_name();
    let directive = |name: &str| {
        ptx.lines()
            .map(str::trim)
            .find_map(|line| line.strip_prefix(name))
            .map(str::trim)
    };

    let version = directive(".version")
        .and_then(|version| version.split_once('.'))
        .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)))
        .ok_or_else(|| eyre!("{file} has no readable `.version` directive"))?;
    if version > MAX_PTX_ISA {
        bail!(
            "{file} uses PTX ISA {}.{}, newer than {}.{} that CUDA 13.0 drivers accept",
            version.0,
            version.1,
            MAX_PTX_ISA.0,
            MAX_PTX_ISA.1
        );
    }

    let target = directive(".target").unwrap_or_default();
    let arch = variant.tier.arch();
    if target.split(',').next().map(str::trim) != Some(arch) {
        bail!("{file} targets `{target}`, expected `{arch}`");
    }

    Ok(version)
}

/// Kernel entry points of a PTX module: each `.entry` name with its parameter
/// declarations, minus the parameter names
type EntryPoints = BTreeMap<String, Vec<String>>;

fn entry_points(ptx: &str) -> Result<EntryPoints> {
    // drop `//` comments, so a comment that mentions `.entry` cannot add a kernel
    let code: String = ptx
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .fold(String::new(), |mut code, line| {
            code.push_str(line);
            code.push('\n');
            code
        });

    let mut entries = EntryPoints::new();
    let mut rest = code.as_str();
    while let Some(start) = find_directive(rest, ".entry") {
        let after = rest[start + ".entry".len()..].trim_start();
        let name_end = after
            .find(|c: char| c == '(' || c == '{' || c.is_whitespace())
            .unwrap_or(after.len());
        let name = &after[..name_end];
        if name.is_empty() {
            bail!("`.entry` without a kernel name");
        }

        let tail = after[name_end..].trim_start();
        let (params, next) = match tail.strip_prefix('(') {
            Some(list) => {
                let (list, next) = list
                    .split_once(')')
                    .ok_or_else(|| eyre!("unterminated parameter list of `{name}`"))?;
                (parse_params(list), next)
            }
            None => (Vec::new(), tail),
        };

        if entries.insert(name.to_string(), params).is_some() {
            bail!("kernel `{name}` is defined twice");
        }
        rest = next;
    }

    Ok(entries)
}

/// Position of a directive token, so `.entry` does not match inside a longer word
fn find_directive(text: &str, directive: &str) -> Option<usize> {
    text.match_indices(directive)
        .map(|(index, _)| index)
        .find(|&index| {
            let before = text[..index].chars().next_back();
            let after = text[index + directive.len()..].chars().next();
            before.is_none_or(char::is_whitespace) && after.is_some_and(char::is_whitespace)
        })
}

/// `.param .u64 .ptr .align 4 k_param_1` becomes `.param .u64 .ptr .align 4`
fn parse_params(list: &str) -> Vec<String> {
    list.split(',')
        .map(|param| {
            let tokens: Vec<_> = param.split_whitespace().collect();
            tokens[..tokens.len().saturating_sub(1)].join(" ")
        })
        .filter(|param| !param.is_empty())
        .collect()
}

/// The host picks one variant per area at run time and looks kernels up by name, so
/// every variant must export the baseline's kernels with the same parameters
fn check_entry_points(modules: &[(Variant, &str)]) -> Result<()> {
    let Some(((baseline, baseline_ptx), higher)) = modules.split_first() else {
        return Ok(());
    };

    let expected =
        entry_points(baseline_ptx).wrap_err_with(|| format!("parsing {}", baseline.file_name()))?;
    let mut problems = Vec::new();
    for (variant, ptx) in higher {
        let actual =
            entry_points(ptx).wrap_err_with(|| format!("parsing {}", variant.file_name()))?;
        problems.extend(entry_point_differences(
            &baseline.file_name(),
            &expected,
            &variant.file_name(),
            &actual,
        ));
    }

    if !problems.is_empty() {
        bail!(
            "variants export different kernels:\n    {}",
            problems.join("\n    ")
        );
    }

    Ok(())
}

fn entry_point_differences(
    expected_file: &str,
    expected: &EntryPoints,
    actual_file: &str,
    actual: &EntryPoints,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (name, params) in expected {
        match actual.get(name) {
            None => problems.push(format!("{actual_file} lacks `{name}` from {expected_file}")),
            Some(other) if other != params => {
                let mut message = format!("`{name}` parameters differ:");
                let _ = write!(message, " {expected_file} ({})", params.join(", "));
                let _ = write!(message, " vs {actual_file} ({})", other.join(", "));
                problems.push(message);
            }
            Some(_) => {}
        }
    }

    for name in actual.keys().filter(|name| !expected.contains_key(*name)) {
        problems.push(format!(
            "{actual_file} adds `{name}`, absent from {expected_file}"
        ));
    }

    problems
}

/// Hash of everything that determines one variant's PTX: the build pins, the target
/// and features, the crate's shared files and this area's own module. Other areas'
/// modules are left out, so a change in one area never marks another area stale
fn sources_hash(crate_dir: &Path, variant: Variant) -> Result<String> {
    let mut files = Vec::new();
    collect_files(crate_dir, crate_dir, &mut files)?;
    files.retain(|relative| belongs_to(relative, variant.area.name));
    files.sort();

    let mut digest = Sha256::new();
    let pins = format!(
        "cuda-oxide={CUDA_OXIDE_REV} toolchain={CUDA_OXIDE_NIGHTLY} arch={} features={}",
        variant.tier.arch(),
        variant.features()
    );
    hash_entry(&mut digest, "pins", pins.as_bytes());
    for relative in files {
        let contents = fs::read(crate_dir.join(&relative))?;
        hash_entry(&mut digest, &relative, &contents);
    }

    Ok(format!("{:x}", digest.finalize()))
}

/// Whether a crate file feeds the given area's PTX
fn belongs_to(relative: &str, area: &str) -> bool {
    let Some(module) = area_of(relative) else {
        return true;
    };

    module == area
}

/// The area that owns `src/<area>.rs` or `src/<area>/...`, if any
fn area_of(relative: &str) -> Option<&'static str> {
    let rest = relative.strip_prefix("src/")?;
    AREAS.iter().map(|area| area.name).find(|area| {
        rest.strip_prefix(area)
            .is_some_and(|tail| tail == ".rs" || tail.starts_with('/'))
    })
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        // build output and editor or OS litter never feed the PTX
        if name == "target" || name.to_string_lossy().starts_with('.') {
            continue;
        }

        if entry.file_type()?.is_dir() {
            collect_files(root, &path, files)?;
            continue;
        }

        let relative = path
            .strip_prefix(root)?
            .to_str()
            .ok_or_else(|| eyre!("non UTF-8 path {}", path.display()))?
            .replace('\\', "/");
        files.push(relative);
    }

    Ok(())
}

/// Length-prefixed so that moving bytes between a name and its contents changes the hash
fn hash_entry(digest: &mut Sha256, name: &str, contents: &[u8]) {
    digest.update((name.len() as u64).to_le_bytes());
    digest.update(name.as_bytes());
    digest.update((contents.len() as u64).to_le_bytes());
    digest.update(contents);
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Contents of `<area>.manifest`: one section per variant, baseline first
#[derive(Debug, Default, PartialEq, Eq)]
struct Manifest {
    ptxas: Option<String>,
    ptxas_flags: Option<String>,
    variants: Vec<ManifestVariant>,
}

#[derive(Debug, PartialEq, Eq)]
struct ManifestVariant {
    tier: Tier,
    sources: String,
    ptx: String,
    cubins: Vec<ManifestCubin>,
}

#[derive(Debug, PartialEq, Eq)]
struct ManifestCubin {
    arch: CubinArch,
    sha256: String,
    ptx: String,
}

impl Manifest {
    const HEADER: &str = "# generated by `cargo xtask cuda-kernels build`; checked by `cargo xtask cuda-kernels check`";

    fn render(&self) -> String {
        let mut text = format!(
            "{}\ncuda-oxide = {CUDA_OXIDE_REV}\ntoolchain = {CUDA_OXIDE_NIGHTLY}\n",
            Self::HEADER
        );
        if let Some(version) = &self.ptxas {
            let _ = writeln!(text, "ptxas = {version}");
        }

        if let Some(flags) = &self.ptxas_flags {
            let _ = writeln!(text, "ptxas-flags = {flags}");
        }

        for variant in &self.variants {
            let _ = write!(
                text,
                "\n[{}]\ntarget = {}\nsources = {}\nptx = {}\n",
                variant.tier.name(),
                variant.tier.arch(),
                variant.sources,
                variant.ptx
            );
            for cubin in &variant.cubins {
                let _ = writeln!(
                    text,
                    "cubin.{} = {} {}",
                    cubin.arch.name(),
                    cubin.sha256,
                    cubin.ptx
                );
            }
        }

        text
    }

    fn parse(text: &str) -> Result<Self> {
        let mut header = BTreeMap::new();
        let mut sections: Vec<(&str, BTreeMap<&str, &str>)> = Vec::new();
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(name) = line
                .strip_prefix('[')
                .and_then(|line| line.strip_suffix(']'))
            {
                sections.push((name, BTreeMap::new()));
                continue;
            }

            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| eyre!("invalid manifest line `{line}`"))?;
            let fields = sections
                .last_mut()
                .map(|(_, fields)| fields)
                .unwrap_or(&mut header);
            if fields.insert(key.trim(), value.trim()).is_some() {
                bail!("duplicate manifest field `{}`", key.trim());
            }
        }

        let variants = sections
            .into_iter()
            .map(|(name, fields)| ManifestVariant::parse(name, &fields))
            .collect::<Result<_>>()?;
        Ok(Self {
            ptxas: header.get("ptxas").map(|value| value.to_string()),
            ptxas_flags: header.get("ptxas-flags").map(|value| value.to_string()),
            variants,
        })
    }
}

impl ManifestVariant {
    fn parse(name: &str, fields: &BTreeMap<&str, &str>) -> Result<Self> {
        let tier =
            Tier::parse(name).ok_or_else(|| eyre!("manifest has unknown tier `[{name}]`"))?;
        let field = |key: &str| {
            fields
                .get(key)
                .map(|value| value.to_string())
                .ok_or_else(|| eyre!("manifest section `[{name}]` has no `{key}` field"))
        };

        let target = field("target")?;
        if target != tier.arch() {
            bail!(
                "manifest section `[{name}]` has target `{target}`, expected `{}`",
                tier.arch()
            );
        }

        let mut cubins = Vec::new();
        for (key, value) in fields {
            let Some(arch) = key.strip_prefix("cubin.") else {
                continue;
            };
            let arch =
                CubinArch::parse(arch).ok_or_else(|| eyre!("unknown cubin arch `{arch}`"))?;
            let hashes: Vec<_> = value.split_whitespace().collect();
            if hashes.len() != 2 {
                bail!("manifest `{key}` needs cubin sha256 and source PTX sha256");
            }

            cubins.push(ManifestCubin {
                arch,
                sha256: hashes[0].into(),
                ptx: hashes[1].into(),
            });
        }

        cubins.sort_by_key(|cubin| cubin.arch.0);
        Ok(Self {
            tier,
            sources: field("sources")?,
            ptx: field("ptx")?,
            cubins,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AREAS, Manifest, ManifestCubin, ManifestVariant, PTXAS_FLAGS, PTXAS_VERSION, Tier, Variant,
        area_of, belongs_to, check_area_cubins, check_entry_points, check_ptx_header, entry_points,
        host_embed_source_problems, sha256_hex, unexpected_ptx_files,
    };

    const PROBE_PTX: &str = "//\n// Generated by LLVM\n//\n.version 6.3\n.target sm_75\n.address_size 64\n\n\t// .globl\tprobe_scale_add // .entry fake(\n.visible .entry probe_scale_add(\n\t.param .f32 probe_scale_add_param_0,\n\t.param .u64 .ptr .align 4 probe_scale_add_param_1,\n\t.param .u64 probe_scale_add_param_2\n)\n{\n\tret;\n}\n";

    fn probe(tier: Tier) -> Variant {
        Variant {
            area: AREAS[0],
            tier,
        }
    }

    #[test]
    fn area_files_only_feed_their_own_area() {
        assert_eq!(area_of("src/fbank.rs"), Some("fbank"));
        assert_eq!(area_of("src/embedding/pool.rs"), Some("embedding"));
        assert_eq!(area_of("src/fbank_common.rs"), None);
        assert!(belongs_to("src/lib.rs", "fbank"));
        assert!(belongs_to("Cargo.lock", "probe"));
        assert!(belongs_to("src/fbank.rs", "fbank"));
        assert!(!belongs_to("src/fbank.rs", "probe"));
        // the record-owned area shares a name prefix with the always-on area, so an
        // fbankdft edit must never mark the pinned fbank PTX stale
        assert_eq!(area_of("src/fbankdft.rs"), Some("fbankdft"));
        assert!(!belongs_to("src/fbankdft.rs", "fbank"));
        assert!(!belongs_to("src/fbank.rs", "fbankdft"));
    }

    #[test]
    fn ptx_header_rejects_isa_newer_than_cuda_13_0_and_wrong_targets() {
        let ok = ".version 6.3\n.target sm_75\n.address_size 64\n";
        assert_eq!(check_ptx_header(probe(Tier::Sm75), ok).ok(), Some((6, 3)));

        let too_new = ".version 9.2\n.target sm_75\n";
        assert!(check_ptx_header(probe(Tier::Sm75), too_new).is_err());

        let wrong_target = ".version 8.0\n.target sm_80\n";
        assert!(check_ptx_header(probe(Tier::Sm75), wrong_target).is_err());
        assert!(check_ptx_header(probe(Tier::Sm80), wrong_target).is_ok());
    }

    #[test]
    fn entry_points_ignore_comments_and_parameter_names() {
        let entries = entry_points(PROBE_PTX).expect("parse");
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries["probe_scale_add"],
            [".param .f32", ".param .u64 .ptr .align 4", ".param .u64"]
        );
    }

    #[test]
    fn entry_points_must_match_across_variants() {
        let same = PROBE_PTX.replace("sm_75", "sm_80");
        assert!(
            check_entry_points(&[(probe(Tier::Sm75), PROBE_PTX), (probe(Tier::Sm80), &same)])
                .is_ok()
        );

        let renamed = same.replace("probe_scale_add", "probe_scale_add_v2");
        assert!(
            check_entry_points(&[
                (probe(Tier::Sm75), PROBE_PTX),
                (probe(Tier::Sm80), &renamed)
            ])
            .is_err()
        );

        let retyped = same.replace(".param .f32", ".param .f64");
        assert!(
            check_entry_points(&[
                (probe(Tier::Sm75), PROBE_PTX),
                (probe(Tier::Sm80), &retyped)
            ])
            .is_err()
        );
    }

    fn host_source() -> String {
        AREAS
            .iter()
            .flat_map(|area| area.variants())
            .map(|variant| {
                let mask = [Tier::Sm75, Tier::Sm80, Tier::Sm90, Tier::Sm120]
                    .into_iter()
                    .filter(|tier| {
                        variant
                            .area
                            .tiers
                            .iter()
                            .rev()
                            .find(|shipped| **shipped <= *tier)
                            == Some(&variant.tier)
                    })
                    .map(|tier| format!("{:?}", tier.host_feature()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("tier_ptx!([{mask}], \"ptx/{}\")", variant.file_name())
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn host_masks_select_the_newest_shipped_variant() {
        let source = host_source();
        assert!(host_embed_source_problems(&source).is_empty());
        assert!(source.contains("tier_ptx!([\"cuda-sm75\"], \"ptx/probe.sm75.ptx\")"));
        assert!(source.contains(
            "tier_ptx!([\"cuda-sm80\", \"cuda-sm90\", \"cuda-sm120\"], \"ptx/probe.sm80.ptx\")"
        ));
        assert!(source.contains("tier_ptx!([\"cuda-sm75\", \"cuda-sm80\", \"cuda-sm90\", \"cuda-sm120\"], \"ptx/fbank.sm75.ptx\")"));
    }

    #[test]
    fn host_masks_reject_missing_extra_and_duplicate_features() {
        for mask in [
            "[\"cuda-sm75\", \"cuda-sm80\"]",
            "[\"cuda-sm75\", \"unexpected\"]",
            "[\"cuda-sm75\", \"cuda-sm75\"]",
            "[]",
            "\"cuda-sm75\"",
        ] {
            let source = host_source().replace("[\"cuda-sm75\"]", mask);
            assert!(!host_embed_source_problems(&source).is_empty(), "{mask}");
        }
    }

    #[test]
    fn host_inclusions_reject_unconditional_duplicate_and_unknown_files() {
        let source = host_source();
        for extra in [
            "include_str!(\"ptx/probe.sm75.ptx\")",
            "include_str!(\n \"ptx/probe.sm75.ptx\")",
            "tier_ptx!([\"cuda-sm75\"], \"ptx/probe.sm75.ptx\")",
            "tier_ptx!([\"cuda-sm75\"], \"ptx/unknown.sm75.ptx\")",
        ] {
            assert!(!host_embed_source_problems(&format!("{source}\n{extra}")).is_empty());
        }
        assert!(!host_embed_source_problems("").is_empty());
    }

    #[test]
    fn binary_masks_reject_missing_extra_and_foreign_architectures() {
        let source = host_source().replace(
            "\"ptx/probe.sm75.ptx\")",
            "\"ptx/probe.sm75\", [75, 80, 86, 89, 90, 120])",
        );
        assert!(host_embed_source_problems(&source).is_empty());
        for arches in [
            "[75, 80, 86, 89, 90]",
            "[75, 80, 86, 89, 90, 120, 121]",
            "[75, 80, 86, 89, 90, 90]",
        ] {
            assert!(
                !host_embed_source_problems(&source.replace("[75, 80, 86, 89, 90, 120]", arches))
                    .is_empty()
            );
        }
    }

    #[test]
    fn manifest_round_trips() {
        let manifest = Manifest {
            ptxas: Some(PTXAS_VERSION.into()),
            ptxas_flags: Some(PTXAS_FLAGS.into()),
            variants: vec![
                ManifestVariant {
                    tier: Tier::Sm75,
                    sources: "abc".into(),
                    ptx: "def".into(),
                    cubins: fixture_variant(probe(Tier::Sm75)).cubins,
                },
                ManifestVariant {
                    tier: Tier::Sm80,
                    sources: "ghi".into(),
                    ptx: "jkl".into(),
                    cubins: fixture_variant(probe(Tier::Sm80)).cubins,
                },
            ],
        };
        assert_eq!(Manifest::parse(&manifest.render()).ok(), Some(manifest));
    }

    fn fixture_variant(variant: Variant) -> ManifestVariant {
        ManifestVariant {
            tier: variant.tier,
            sources: "sources".into(),
            ptx: "ptx-hash".into(),
            cubins: variant
                .cubins()
                .map(|cubin| ManifestCubin {
                    arch: cubin.arch,
                    sha256: sha256_hex(b"cubin"),
                    ptx: "ptx-hash".into(),
                })
                .collect(),
        }
    }

    fn cubin_fixture() -> (tempfile::TempDir, Manifest) {
        let dir = tempfile::tempdir().expect("temporary directory");
        let manifest = Manifest {
            ptxas: Some(PTXAS_VERSION.into()),
            ptxas_flags: Some(PTXAS_FLAGS.into()),
            variants: AREAS[0].variants().map(fixture_variant).collect(),
        };
        for cubin in AREAS[0].variants().flat_map(Variant::cubins) {
            std::fs::write(dir.path().join(cubin.file_name()), b"cubin").expect("write cubin");
        }

        check_area_cubins(dir.path(), AREAS[0], &manifest).expect("valid cubins");
        (dir, manifest)
    }

    #[test]
    fn check_rejects_tampered_cubin() {
        let (dir, manifest) = cubin_fixture();
        let cubin = probe(Tier::Sm75).cubins().next().expect("cubin");
        std::fs::write(dir.path().join(cubin.file_name()), b"tampered").expect("tamper");
        let error = check_area_cubins(dir.path(), AREAS[0], &manifest).expect_err("tampered hash");
        assert!(error.to_string().contains("cubin sha256 does not match"));
    }

    #[test]
    fn check_rejects_cubin_ptx_mismatch() {
        let (dir, mut manifest) = cubin_fixture();
        manifest.variants[0].cubins[0].ptx = "different-ptx".into();
        let error = check_area_cubins(dir.path(), AREAS[0], &manifest).expect_err("PTX mismatch");
        assert!(error.to_string().contains("PTX hash does not match"));
    }

    #[test]
    fn check_rejects_wrong_ptxas_version_and_flags() {
        let (dir, mut manifest) = cubin_fixture();
        manifest.ptxas = Some("Cuda compilation tools, release 12.8, V12.8.93".into());
        let error = check_area_cubins(dir.path(), AREAS[0], &manifest).expect_err("wrong ptxas");
        assert!(error.to_string().contains("wrong ptxas version"));
        manifest.ptxas = Some(PTXAS_VERSION.into());
        manifest.ptxas_flags = Some("-lineinfo".into());
        let error = check_area_cubins(dir.path(), AREAS[0], &manifest).expect_err("wrong flags");
        assert!(error.to_string().contains("wrong ptxas flags"));
    }

    #[test]
    fn check_rejects_missing_cubin() {
        let (dir, manifest) = cubin_fixture();
        let cubin = probe(Tier::Sm75).cubins().next().expect("cubin");
        std::fs::remove_file(dir.path().join(cubin.file_name())).expect("remove cubin");
        let error = check_area_cubins(dir.path(), AREAS[0], &manifest).expect_err("missing cubin");
        assert!(
            error
                .to_string()
                .contains("missing cubin probe.sm75.sm_75.cubin")
        );
    }

    #[test]
    fn check_rejects_stray_cubin() {
        let (dir, _) = cubin_fixture();
        std::fs::write(dir.path().join("probe.sm75.sm_100.cubin"), b"stray").expect("stray cubin");
        let errors = unexpected_ptx_files(dir.path()).expect("scan directory");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("probe.sm75.sm_100.cubin is not a declared"));
    }

    #[test]
    fn check_rejects_missing_or_incompatible_manifest_arches() {
        let (dir, mut manifest) = cubin_fixture();
        manifest.variants[0].cubins.pop();
        assert!(check_area_cubins(dir.path(), AREAS[0], &manifest).is_err());
        manifest.variants[1].cubins = fixture_variant(probe(Tier::Sm75)).cubins;
        manifest.variants[0] = fixture_variant(probe(Tier::Sm75));
        assert!(check_area_cubins(dir.path(), AREAS[0], &manifest).is_err());
    }
}
