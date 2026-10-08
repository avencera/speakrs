use std::sync::{Arc, Mutex};

use ndarray::{Array2, Array3, s};
use ort::value::TensorRef;

use crate::inference::{InferenceError, SharedSession};

use super::super::FBANK_BATCH_SIZE;
use super::super::tensor::{array2_from_shape_vec, fbank_hw_from_i64, first_output};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FbankPoolRoute {
    SessionPool,
    Standard,
}

pub(super) fn fbank_pool_route(
    input_count: usize,
    pool_size: usize,
    has_batched: bool,
) -> FbankPoolRoute {
    if input_count <= 1 || pool_size == 0 {
        return FbankPoolRoute::Standard;
    }
    if pool_size == 1 && input_count >= FBANK_BATCH_SIZE && has_batched {
        return FbankPoolRoute::Standard;
    }

    FbankPoolRoute::SessionPool
}

/// CPU filterbank sessions that split one request across worker threads
#[derive(Clone)]
pub(super) struct SharedFbankPool(Arc<FbankPool>);

struct FbankPool {
    sessions: Vec<SharedSession>,
    execution: Mutex<()>,
}

impl SharedFbankPool {
    pub(super) fn new(sessions: Vec<SharedSession>) -> Self {
        Self(Arc::new(FbankPool {
            sessions,
            execution: Mutex::new(()),
        }))
    }

    pub(super) fn len(&self) -> usize {
        self.0.sessions.len()
    }

    pub(super) fn run(
        &self,
        audios: &[&[f32]],
        window_samples: usize,
    ) -> Result<Vec<Array2<f32>>, InferenceError> {
        let _execution = self
            .0
            .execution
            .lock()
            .map_err(|_| InferenceError::LockPoisoned {
                resource: "shared filterbank pool",
            })?;

        compute_fbanks_with_pool(&self.0.sessions, audios, window_samples)
    }
}

fn compute_fbanks_with_pool(
    pool: &[SharedSession],
    audios: &[&[f32]],
    window_samples: usize,
) -> Result<Vec<Array2<f32>>, InferenceError> {
    let worker_count = audios.len().min(pool.len());
    let inputs_per_worker = audios.len().div_ceil(worker_count);

    std::thread::scope(|scope| {
        let handles: Vec<_> = pool
            .iter()
            .take(worker_count)
            .zip(audios.chunks(inputs_per_worker))
            .map(|(session, inputs)| {
                scope.spawn(move || compute_fbank_worker(session, inputs, window_samples))
            })
            .collect();
        let mut results = Vec::with_capacity(audios.len());
        let mut failure = None;

        for handle in handles {
            match handle.join() {
                Ok(Ok(worker_results)) if failure.is_none() => results.extend(worker_results),
                Ok(Ok(_)) => {}
                Ok(Err(error)) if failure.is_none() => failure = Some(error),
                Ok(Err(_)) => {}
                Err(_) if failure.is_none() => {
                    failure = Some(InferenceError::WorkerPanic {
                        worker: "filterbank session pool",
                    });
                }
                Err(_) => {}
            }
        }

        match failure {
            Some(error) => Err(error),
            None => Ok(results),
        }
    })
}

fn compute_fbank_worker(
    session: &SharedSession,
    audios: &[&[f32]],
    window_samples: usize,
) -> Result<Vec<Array2<f32>>, InferenceError> {
    let mut session = session.lock()?;
    let mut waveform = Array3::<f32>::zeros((1, 1, window_samples));
    let mut results = Vec::with_capacity(audios.len());

    for audio in audios {
        waveform.fill(0.0);
        let copy_len = audio.len().min(window_samples);
        waveform
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        let waveform_tensor = TensorRef::from_array_view(waveform.view())?;
        let outputs = session.run(ort::inputs!["waveform" => waveform_tensor])?;
        let output = first_output(outputs.values(), "pool chunk fbank output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let (frames, features) = fbank_hw_from_i64(shape, "pool chunk fbank output")?;
        results.push(array2_from_shape_vec(
            frames,
            features,
            data.to_vec(),
            "pool chunk fbank output",
        )?);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::inference::{OrtProvider, ensure_ort_ready};
    use crate::pipeline::OrtThreadCount;

    use super::super::build_fbank_session;

    #[test]
    fn one_session_pool_keeps_the_full_batched_route() {
        assert_eq!(
            fbank_pool_route(FBANK_BATCH_SIZE, 1, true),
            FbankPoolRoute::Standard
        );
        assert_eq!(
            fbank_pool_route(FBANK_BATCH_SIZE * 2, 1, true),
            FbankPoolRoute::Standard
        );
    }

    #[test]
    fn multiple_sessions_use_the_pool_for_a_full_batch() {
        assert_eq!(
            fbank_pool_route(FBANK_BATCH_SIZE, 2, true),
            FbankPoolRoute::SessionPool
        );
    }

    #[test]
    fn partial_batches_use_an_available_pool() {
        assert_eq!(fbank_pool_route(7, 1, true), FbankPoolRoute::SessionPool);
        assert_eq!(fbank_pool_route(7, 0, true), FbankPoolRoute::Standard);
    }

    #[test]
    fn pool_preserves_output_values_and_input_order() {
        let model_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/models/wespeaker-fbank.onnx");
        if !model_path.is_file() || ensure_ort_ready().is_err() {
            eprintln!(
                "skipping fbank pool test; missing {} or ONNX Runtime",
                model_path.display()
            );
            return;
        }
        let threads = OrtThreadCount::new(1).unwrap();
        let sessions: Vec<_> = (0..2)
            .map(|_| {
                build_fbank_session(&model_path, OrtProvider::Cpu, threads).map(SharedSession::new)
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let reference = build_fbank_session(&model_path, OrtProvider::Cpu, threads)
            .map(SharedSession::new)
            .unwrap();
        let first = vec![0.0; 24_000];
        let second: Vec<_> = (0..32_000)
            .map(|index| (index % 97) as f32 / 97.0)
            .collect();
        let third: Vec<_> = (0..16_000)
            .map(|index| -((index % 53) as f32) / 53.0)
            .collect();
        let audios = [first.as_slice(), second.as_slice(), third.as_slice()];

        let expected = compute_fbank_worker(&reference, &audios, 160_000).unwrap();
        let actual = compute_fbanks_with_pool(&sessions, &audios, 160_000).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.dim(), expected.dim());
            for (actual, expected) in actual.iter().zip(&expected) {
                approx::assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
            }
        }
    }
}
