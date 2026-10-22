//! Graph reuse and row ordering through the production embedding wrapper

use super::{Batches, CHUNK_MASKS_LEN, FBANK_LEN};
use crate::inference::cuda::batch_class_tests::{Case, load_case, parity, setup};
use crate::inference::cuda::fp16_range_tests::{SyntheticModel, runtime, scratch};
use crate::inference::cuda::{
    CudaError, CudaMath, CudaRuntime, CudaSession, DeviceTensor, EMBEDDING_DIM, ResNetEmbedding,
    SafetensorsFile,
};

fn session(weights: &SafetensorsFile, math: CudaMath) -> Result<CudaSession<Batches>, CudaError> {
    CudaSession::new(crate::inference::cuda::CudaRuntime::new(0)?, |runtime| {
        Ok(Batches {
            model: ResNetEmbedding::load(runtime, weights, math)?,
            graphs: true,
            plans: std::array::from_fn(|_| None),
            spare: None,
            next_spare: false,
            activations: None,
        })
    })
}

fn reordered(source: &Case, chunks: usize, phase: usize) -> Case {
    let mut fbank = Vec::with_capacity(chunks * FBANK_LEN);
    let mut masks = Vec::with_capacity(chunks * CHUNK_MASKS_LEN);
    let mut expected = Vec::with_capacity(chunks * 3 * 256);
    for index in 0..chunks {
        let row = (index + phase) % source.chunks;
        fbank.extend_from_slice(&source.fbank[row * FBANK_LEN..(row + 1) * FBANK_LEN]);
        for speaker in 0..3 {
            let source_row = row * 3 + (speaker + phase) % 3;
            masks.extend_from_slice(&source.masks[source_row * 589..(source_row + 1) * 589]);
            expected.extend_from_slice(&source.expected[source_row * 256..(source_row + 1) * 256]);
        }
    }
    Case {
        fbank,
        masks,
        expected,
        chunks,
    }
}

fn embed(session: &mut CudaSession<Batches>, case: &Case) -> Result<Vec<f32>, CudaError> {
    session.run(|runtime, batches| {
        batches.embed(
            runtime,
            case.chunks,
            &case.masks,
            |runtime, rows, target| {
                target.copy_from_host(
                    runtime.stream(),
                    &case.fbank[rows.start * FBANK_LEN..rows.end * FBANK_LEN],
                )
            },
        )
    })
}

#[test]
#[ignore = "GPU wrapper check; run under the GPU lock with reference tensors"]
fn embedding_wrapper_remainders_and_graph_reuse() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_wrapper_remainders_and_graph_reuse") else {
        return Ok(());
    };
    drop(runtime);
    let source = load_case(&root, "wespeaker-multimask-tail-b32", "test_and_short_b32");
    assert_eq!(source.chunks, 32);
    let weights = SafetensorsFile::open(std::env::var_os("TRUNK_WEIGHTS").map_or_else(
        || root.join("wespeaker-multimask-tail/wespeaker-multimask-tail.safetensors"),
        std::path::PathBuf::from,
    ))?;
    for math in [CudaMath::Tf32, CudaMath::Fp32] {
        let mut reused = session(&weights, math)?;
        for chunks in 1..=32 {
            let case = reordered(&source, chunks, 0);
            let actual = embed(&mut reused, &case)?;
            assert!(actual.iter().all(|value| value.is_finite()));
            assert!(parity(&actual, &case.expected, 256).min_cosine >= 0.999);
        }

        // change both window order and speaker order before reusing every captured class
        let mut fresh = session(&weights, math)?;
        for chunks in (1..=32).rev() {
            let case = reordered(&source, chunks, 7);
            let actual = embed(&mut reused, &case)?;
            let control = embed(&mut fresh, &case)?;
            assert!(actual.iter().all(|value| value.is_finite()));
            assert_eq!(
                actual, control,
                "{math:?} remainder {chunks}: stale graph state"
            );
            assert!(parity(&actual, &case.expected, 256).min_cosine >= 0.999);
        }
        eprintln!(
            "WRAPPER_REUSE math={math:?} remainders=32 changed_inputs=true fresh_session_parity=true"
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU retained-memory check; run under the GPU lock with reference tensors"]
fn embedding_32_then_31_retains_one_activation_allocation() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_32_then_31_retains_one_activation_allocation")
    else {
        return Ok(());
    };
    drop(runtime);
    let source = load_case(&root, "wespeaker-multimask-tail-b32", "test_and_short_b32");
    let weights = SafetensorsFile::open(std::env::var_os("TRUNK_WEIGHTS").map_or_else(
        || root.join("wespeaker-multimask-tail/wespeaker-multimask-tail.safetensors"),
        std::path::PathBuf::from,
    ))?;
    let mut reused = session(&weights, CudaMath::Tf32)?;
    for chunks in [32, 31] {
        let case = reordered(&source, chunks, 0);
        let actual = embed(&mut reused, &case)?;
        assert!(parity(&actual, &case.expected, 256).min_cosine >= 0.999);
        reused.run(|_, batches| {
            // measure real device storage, not the sum of each class's logical views
            assert_eq!(
                batches.activations.as_ref().unwrap().retained_bytes(),
                1_144_586_240
            );
            Ok::<_, CudaError>(())
        })?;
    }
    reused.run(|_, batches| {
        assert!(batches.plans.iter().all(Option::is_some));
        let allocation = batches.activations.as_ref().unwrap();
        assert!(
            batches
                .plans
                .iter()
                .flatten()
                .all(|batch| batch.shares_activations(allocation))
        );
        Ok::<_, CudaError>(())
    })?;
    Ok(())
}

/// A full batch whose download is queued keeps its result while the next full batch runs
/// on the other 32-chunk batch, including the FP16 range recompute that reruns the
/// queued batch from its own inputs after the next one has overwritten the shared
/// activations
///
/// The synthetic model passes the filterbank value plus 128 through `resnet.layer1.0`
/// to embedding element 0: an input of 0 gives 256 and saturates FP16 tiles where the
/// device selects them, and an input of -120 gives 16, in range
#[test]
fn queued_download_survives_the_next_full_batch() -> Result<(), CudaError> {
    let Some(runtime) = runtime("queued_download_survives_the_next_full_batch") else {
        return Ok(());
    };
    let mut model = SyntheticModel::channel_zero_path(128.0);
    model.center("resnet.conv1", 1.0);
    model.center("resnet.layer1.0.conv1", 1.0);
    model.center("resnet.layer1.0.conv2", 1.0);
    let path = scratch("queued-download");
    let weights = model.write(&path);
    let mut session = CudaSession::new(runtime, |runtime| {
        Ok(Batches {
            model: ResNetEmbedding::load(runtime, &weights, CudaMath::Tf32)?,
            graphs: true,
            plans: std::array::from_fn(|_| None),
            spare: None,
            next_spare: false,
            activations: None,
        })
    })?;
    let masks = vec![1.0; 32 * CHUNK_MASKS_LEN];
    let fill = |value: f32| {
        move |runtime: &CudaRuntime, rows: std::ops::Range<usize>, target: &mut DeviceTensor| {
            target.copy_from_host(runtime.stream(), &vec![value; rows.len() * FBANK_LEN])
        }
    };

    // three full batches, each launched before the previous one is read, so both
    // 32-chunk batches take a turn holding a queued download
    let inputs = [(0.0, 256.0), (-120.0, 16.0), (0.0, 256.0)];
    let (outputs, saturating, recomputed) = session.run(|runtime, batches| {
        let mut outputs = Vec::new();
        let mut queued: Option<super::Launch> = None;
        for (input, _) in inputs {
            let (done, launch) = batches.launch(runtime, 32, &masks, fill(input))?;
            assert!(done.is_empty());
            let launch = launch.expect("a full batch is left running");
            if let Some(previous) = queued.take() {
                assert_ne!(previous.spare, launch.spare);
                outputs.push(batches.finish_download(runtime, previous)?);
            }
            batches.enqueue_download(runtime, &launch)?;
            queued = Some(launch);
        }
        let last = queued.expect("the last batch is queued");
        outputs.push(batches.finish_download(runtime, last)?);

        let batch = batches.plans[4].as_ref().expect("planned batch");
        let saturating = batch.fp16_convs().contains(&"resnet.layer1.0.conv1");
        Ok::<_, CudaError>((outputs, saturating, batch.recomputed_without_fp16()))
    })?;
    eprintln!("QUEUED_DOWNLOAD saturating={saturating} recomputed={recomputed}");
    // the batches given 0 ran in the non-spare batch, and saturate where FP16 tiles run
    assert_eq!(recomputed, saturating);
    for ((_, expected), rows) in inputs.iter().zip(&outputs) {
        assert_eq!(rows.len(), 32 * 3 * EMBEDDING_DIM);
        for (row, embedding) in rows.chunks(EMBEDDING_DIM).enumerate() {
            assert_eq!(embedding[0], *expected, "row {row}");
            assert!(embedding[1..].iter().all(|&value| value == 0.0));
        }
    }
    drop(session);
    drop(weights);
    std::fs::remove_file(&path).expect("removed scratch model");
    Ok(())
}
