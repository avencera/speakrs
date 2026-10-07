//! Linux loader evidence in a fresh process, separate from parity and timing tests

use std::error::Error;
use std::path::PathBuf;

use super::{
    CudaError, CudaFbank, CudaLibrary, CudaMath, CudaRuntime, CudaSegmentation, PtxTier,
    ResNetEmbedding, SafetensorsFile, SegmentationOptions,
};

fn maps(stage: &str) -> Result<String, Box<dyn Error>> {
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    let directory = PathBuf::from(std::env::var("SPEAKRS_LOADER_EVIDENCE")?);
    std::fs::create_dir_all(&directory)?;
    std::fs::write(directory.join(format!("{stage}.maps")), &maps)?;
    Ok(maps)
}

fn no_libraries(maps: &str) {
    for library in ["libcublas", "libcudnn", "libnvrtc"] {
        assert!(
            !maps.contains(library),
            "unexpected library mapping: {library}"
        );
    }
}

fn options(math: CudaMath) -> SegmentationOptions {
    SegmentationOptions {
        math,
        #[cfg(feature = "_cuda-libraries")]
        lstm_algo: super::CudaLstmAlgorithm::PersistStaticSmallH,
        cuda_graph: true,
    }
}

fn rejected(error: CudaError, runtime: &CudaRuntime, math: CudaMath) {
    eprintln!("loader policy: {error}");
    let CudaError::NotDriverOnly {
        area,
        boundary,
        batch,
        math: actual_math,
        tier,
        device,
        library,
    } = error
    else {
        panic!("expected NotDriverOnly");
    };
    assert_eq!(batch, 1);
    assert_eq!(actual_math, math);
    assert_eq!(tier, PtxTier::Sm75);
    assert_eq!(device, runtime.compute_capability());
    assert!(!area.is_empty() && !boundary.is_empty());
    assert!(matches!(library, CudaLibrary::Cublas | CudaLibrary::Cudnn));
}

#[test]
#[ignore = "loader proof; run one mode in a fresh GPU-locked process"]
fn loader_proof() -> Result<(), Box<dyn Error>> {
    let runtime = CudaRuntime::new(0)?;
    no_libraries(&maps("runtime")?);
    let mode = std::env::var("SPEAKRS_LOADER_MODE")?;
    if mode == "runtime" {
        return Ok(());
    }
    if mode == "fbank" {
        let model = CudaFbank::new(&runtime, CudaMath::Fp32)?;
        let mut buffers = model.buffers(&runtime, 1)?;
        let samples = vec![0.0; super::FBANK_WINDOW_SAMPLES];
        model.compute_host(&runtime, &[&samples], &mut buffers)?;
        runtime.synchronize()?;
        let maps = maps("fbank-forward")?;
        assert!(maps.contains("libcublas"));
        assert!(!maps.contains("libcudnn") && !maps.contains("libnvrtc"));
        return Ok(());
    }
    let assets = PathBuf::from(std::env::var("SPEAKRS_CUDA_ASSETS")?);
    let segmentation = SafetensorsFile::open(assets.join("segmentation-3.0.safetensors"))?;
    if mode == "segmentation" {
        let mut model = CudaSegmentation::new(&runtime, &segmentation, options(CudaMath::Fp32))?;
        let samples = vec![0.0; 160_000];
        model.run(&runtime, 1, &samples)?;
        runtime.synchronize()?;
        let maps = maps("segmentation-forward")?;
        assert!(!maps.contains("libcudnn_adv") && !maps.contains("libnvrtc"));
        return Ok(());
    }
    assert_eq!(mode, "driver-only");
    assert!(super::driver_only());
    assert_eq!(PtxTier::native(runtime.compute_capability()), PtxTier::Sm75);
    let embedding = SafetensorsFile::open(assets.join("wespeaker-multimask-tail.safetensors"))?;
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        rejected(
            CudaSegmentation::new(&runtime, &segmentation, options(math)).unwrap_err(),
            &runtime,
            math,
        );
        rejected(
            ResNetEmbedding::load(&runtime, &embedding, math).unwrap_err(),
            &runtime,
            math,
        );
        rejected(CudaFbank::new(&runtime, math).unwrap_err(), &runtime, math);
    }
    #[cfg(feature = "_cuda-libraries")]
    for library in [CudaLibrary::Cublas, CudaLibrary::Cudnn, CudaLibrary::Nvrtc] {
        assert!(matches!(
            runtime.prepare_library(library),
            Err(CudaError::NotDriverOnly { .. })
        ));
    }
    for area in [
        super::KernelModule::Fbank,
        super::KernelModule::Embedding,
        super::KernelModule::Segmentation,
        super::KernelModule::Resnet,
        super::KernelModule::Lstm,
        super::KernelModule::Sincnet,
    ] {
        let loaded = runtime.load_kernels(area)?;
        eprintln!("loader area: {} {}", area.name(), loaded.tier());
        assert_eq!(loaded.tier(), PtxTier::Sm75);
    }
    if !PtxTier::Sm80.is_compiled_in() {
        assert!(matches!(
            CudaRuntime::with_ptx_tier(0, Some(PtxTier::Sm80)),
            Err(CudaError::TierNotCompiledIn {
                tier: PtxTier::Sm80,
                ..
            })
        ));
    } else {
        let forced = CudaRuntime::with_ptx_tier(0, Some(PtxTier::Sm80))?;
        assert_eq!(
            forced.load_kernels(super::KernelModule::Lstm)?.tier(),
            PtxTier::Sm75
        );
    }
    no_libraries(&maps("driver-only-model-errors")?);
    Ok(())
}
