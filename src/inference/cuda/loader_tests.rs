//! Linux loader evidence in a fresh process, separate from parity and timing tests

use std::error::Error;
use std::path::PathBuf;

use super::{
    CudaError, CudaFbank, CudaMath, CudaRuntime, CudaSegmentation, PtxTier, ResNetEmbedding,
    SafetensorsFile, SegmentationOptions,
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

fn rejected(error: CudaError, math: CudaMath) {
    eprintln!("loader policy: {error}");
    let CudaError::MissingKernel {
        boundary,
        batch,
        math: actual_math,
    } = error
    else {
        panic!("expected MissingKernel");
    };
    assert_eq!(batch, 2);
    assert_eq!(actual_math, math);
    assert_eq!(boundary, "linear0");
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
        let mut library_selected = false;
        for batch in 1..=32 {
            let selected = super::implementation::plan_selection(
                &runtime,
                super::implementation::BoundaryId::named("fbank.dft"),
                batch,
                CudaMath::Fp32,
                #[cfg(feature = "_cuda-libraries")]
                None,
            )?;
            library_selected |= matches!(selected, super::implementation::Selected::Library);
        }
        assert_eq!(maps.contains("libcublas"), library_selected);
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
    let embedding = SafetensorsFile::open(assets.join("wespeaker-multimask-tail.safetensors"))?;
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        let mut segmenter = CudaSegmentation::new(&runtime, &segmentation, options(math))?;
        let embedder = ResNetEmbedding::load(&runtime, &embedding, math)?;
        let fbank = CudaFbank::new(&runtime, math)?;
        let mut buffers = fbank.buffers(&runtime, 32)?;
        for batch in super::implementation::MODEL_BATCHES {
            let samples = vec![0.0; 160_000 * batch];
            let output = segmenter.run(&runtime, batch, &samples)?;
            assert!(!output.is_empty() && output.iter().all(|value| value.is_finite()));
            let waveforms: Vec<_> = samples.chunks_exact(super::FBANK_WINDOW_SAMPLES).collect();
            let features = fbank.compute_host(&runtime, &waveforms, &mut buffers)?;
            let values = runtime.stream().clone_dtoh(&features)?;
            assert_eq!(
                values.len(),
                batch * super::FBANK_FRAMES * super::FBANK_MEL_BINS
            );
            assert!(values.iter().all(|value| value.is_finite()));
            let mut embedding_batch = embedder.batch(&runtime, batch)?;
            embedding_batch
                .fbank_mut()
                .copy_from_host(runtime.stream(), &values)?;
            let masks = vec![1.0; embedding_batch.masks_mut().len()];
            embedding_batch
                .masks_mut()
                .copy_from_host(runtime.stream(), &masks)?;
            embedding_batch.capture_graph(&runtime)?;
            embedding_batch.forward(&runtime)?;
            let output = embedding_batch.download_output(&runtime)?;
            assert_eq!(output.len(), batch * 3 * super::EMBEDDING_DIM);
            assert!(output.iter().all(|value| value.is_finite()));
            runtime.synchronize()?;
            no_libraries(&maps(&format!("driver-only-{math:?}-b{batch}-forward"))?);
        }
        rejected(
            super::implementation::plan_selection(
                &runtime,
                super::implementation::BoundaryId::named("linear0"),
                2,
                math,
                #[cfg(feature = "_cuda-libraries")]
                None,
            )
            .unwrap_err(),
            math,
        );
    }
    if std::env::var_os(super::kernels::FORCE_PTX_JIT_ENV).is_some_and(|value| value == "1") {
        let kernels = runtime.load_kernels(super::KernelModule::Segdense)?;
        assert!(matches!(
            kernels.artifact(),
            super::kernels::LoadedArtifact::PtxJit { .. }
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
    no_libraries(&maps("driver-only-forward-and-missing-kernel")?);
    Ok(())
}
