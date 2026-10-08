//! Graph reuse and row ordering through the production embedding wrapper

use super::{Batches, CHUNK_MASKS_LEN, FBANK_LEN};
use crate::inference::cuda::batch_class_tests::{Case, load_case, parity, setup};
use crate::inference::cuda::{CudaError, CudaMath, CudaSession, ResNetEmbedding, SafetensorsFile};

fn session(weights: &SafetensorsFile, math: CudaMath) -> Result<CudaSession<Batches>, CudaError> {
    CudaSession::new(crate::inference::cuda::CudaRuntime::new(0)?, |runtime| {
        Ok(Batches {
            model: ResNetEmbedding::load(runtime, weights, math)?,
            graphs: true,
            plans: std::array::from_fn(|_| None),
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
