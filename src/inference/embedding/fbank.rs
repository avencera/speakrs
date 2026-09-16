use ndarray::{Array2, Array3, s};
use ort::value::TensorRef;

#[cfg(feature = "coreml")]
use super::fbank_hw_from_shape;
#[cfg(feature = "coreml")]
use super::tensor::array3_slice;
use super::{
    EmbeddingModel, FBANK_BATCH_SIZE, array2_from_shape_vec, fbank_hw_from_i64, first_output,
};
use crate::inference::SharedSession;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FbankPoolRoute {
    SessionPool,
    Standard,
}

fn fbank_pool_route(input_count: usize, pool_size: usize, has_batched: bool) -> FbankPoolRoute {
    if input_count <= 1 || pool_size == 0 {
        return FbankPoolRoute::Standard;
    }
    if pool_size == 1 && input_count >= FBANK_BATCH_SIZE && has_batched {
        return FbankPoolRoute::Standard;
    }

    FbankPoolRoute::SessionPool
}

impl EmbeddingModel {
    /// Compute fbank features for a single audio chunk via the split fbank model
    pub(crate) fn compute_chunk_fbank(&mut self, audio: &[f32]) -> Result<Array2<f32>, ort::Error> {
        let copy_len = audio.len().min(self.meta.window_samples);
        self.buffers
            .split_waveform_buffer
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        if copy_len < self.meta.window_samples {
            self.buffers
                .split_waveform_buffer
                .slice_mut(s![0, 0, copy_len..])
                .fill(0.0);
        }

        #[cfg(feature = "coreml")]
        {
            self.ensure_native_fbank_loaded()?;
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.coreml.native_fbank_session.as_ref() {
            let input_data = array3_slice(
                &self.buffers.split_waveform_buffer,
                "native chunk fbank input",
            )?;
            let tensor = native
                .predict_cached(&[(&self.coreml.cached_fbank_single_shape, input_data)])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            let (frames, features) =
                fbank_hw_from_shape(tensor.layout().dims(), "native chunk fbank output")?;
            return array2_from_shape_vec(
                frames,
                features,
                tensor.into_data(),
                "native chunk fbank output",
            );
        }

        let waveform_tensor =
            TensorRef::from_array_view(self.buffers.split_waveform_buffer.view())?;
        let mut session = self
            .ort
            .split_fbank_session
            .as_ref()
            .ok_or_else(|| ort::Error::new("missing split fbank session"))?
            .lock()?;
        let outputs = session.run(ort::inputs!["waveform" => waveform_tensor])?;
        let output = first_output(outputs.values(), "chunk fbank output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let (frames, features) = fbank_hw_from_i64(shape, "chunk fbank output")?;
        array2_from_shape_vec(frames, features, data.to_vec(), "chunk fbank output")
    }

    /// Compute fbank features for multiple audio chunks in a single batched call
    pub fn compute_chunk_fbanks_batch(
        &mut self,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        let has_batched = self.has_batched_fbank();
        if fbank_pool_route(audios.len(), self.ort.split_fbank_pool.len(), has_batched)
            == FbankPoolRoute::SessionPool
        {
            return self.compute_fbanks_with_pool(audios);
        }

        if !has_batched {
            tracing::debug!(
                count = audios.len(),
                "fbank: no batched session, falling back to per-window"
            );
            return audios
                .iter()
                .map(|audio| self.compute_chunk_fbank(audio))
                .collect();
        }
        let mut results = Vec::with_capacity(audios.len());
        for batch_start in (0..audios.len()).step_by(FBANK_BATCH_SIZE) {
            let batch_end = (batch_start + FBANK_BATCH_SIZE).min(audios.len());
            let batch = &audios[batch_start..batch_end];

            if batch.len() == 1 {
                for audio in batch {
                    results.push(self.compute_chunk_fbank(audio)?);
                }
                continue;
            }

            self.fill_split_fbank_batch_buffer(batch);

            #[cfg(feature = "coreml")]
            if self.try_push_native_fbank_batch(&mut results, batch.len())? {
                continue;
            }

            if batch.len() < FBANK_BATCH_SIZE {
                for audio in batch {
                    results.push(self.compute_chunk_fbank(audio)?);
                }
                continue;
            }

            let waveform_tensor =
                TensorRef::from_array_view(self.buffers.split_fbank_batch_buffer.view())?;
            let mut session = self
                .ort
                .split_fbank_batched_session
                .as_ref()
                .ok_or_else(|| ort::Error::new("missing split fbank batched session"))?
                .lock()?;
            let outputs = session.run(ort::inputs!["waveform" => waveform_tensor])?;
            let output = first_output(outputs.values(), "batched chunk fbank output")?;
            let (shape, data) = output.try_extract_tensor::<f32>()?;
            let (frames, features) = fbank_hw_from_i64(shape, "batched chunk fbank output")?;
            Self::push_fbank_batch_results(&mut results, data, frames, features, batch.len())?;
        }

        Ok(results)
    }

    fn compute_fbanks_with_pool(&self, audios: &[&[f32]]) -> Result<Vec<Array2<f32>>, ort::Error> {
        self.ort
            .split_fbank_pool
            .run(audios, self.meta.window_samples)
    }

    fn fill_split_fbank_batch_buffer(&mut self, audios: &[&[f32]]) {
        self.buffers.split_fbank_batch_buffer.fill(0.0);
        for (idx, audio) in audios.iter().enumerate() {
            let copy_len = audio.len().min(self.meta.window_samples);
            self.buffers
                .split_fbank_batch_buffer
                .slice_mut(s![idx, 0, ..copy_len])
                .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        }
    }

    fn push_fbank_batch_results(
        results: &mut Vec<Array2<f32>>,
        data: &[f32],
        frames: usize,
        features: usize,
        count: usize,
    ) -> Result<(), ort::Error> {
        let stride = frames * features;
        for idx in 0..count {
            let start = idx * stride;
            let batch = array2_from_shape_vec(
                frames,
                features,
                data[start..start + stride].to_vec(),
                "batched fbank output",
            )?;
            results.push(batch);
        }
        Ok(())
    }

    #[cfg(feature = "coreml")]
    fn try_push_native_fbank_batch(
        &mut self,
        results: &mut Vec<Array2<f32>>,
        count: usize,
    ) -> Result<bool, ort::Error> {
        self.ensure_native_fbank_batched_loaded()?;
        let Some(native) = self.coreml.native_fbank_batched_session.as_ref() else {
            return Ok(false);
        };

        let input_data = array3_slice(
            &self.buffers.split_fbank_batch_buffer,
            "native batched fbank input",
        )?;
        let tensor = native
            .predict_cached(&[(&self.coreml.cached_fbank_batch_shape, input_data)])
            .map_err(|e| ort::Error::new(e.to_string()))?;
        let (frames, features) =
            fbank_hw_from_shape(tensor.layout().dims(), "native batched fbank output")?;
        Self::push_fbank_batch_results(results, &tensor.into_data(), frames, features, count)?;
        Ok(true)
    }
}

pub(super) fn compute_fbanks_with_pool(
    pool: &[SharedSession],
    audios: &[&[f32]],
    window_samples: usize,
) -> Result<Vec<Array2<f32>>, ort::Error> {
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
                    failure = Some(ort::Error::new("filterbank session pool worker panicked"));
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
) -> Result<Vec<Array2<f32>>, ort::Error> {
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
    use crate::inference::{ExecutionMode, ensure_ort_ready};
    use crate::pipeline::OrtThreadCount;

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
            return;
        }
        let threads = OrtThreadCount::new(1).unwrap();
        let sessions: Vec<_> = (0..2)
            .map(|_| {
                EmbeddingModel::build_fbank_session(&model_path, ExecutionMode::Cpu, threads)
                    .map(SharedSession::new)
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let reference =
            EmbeddingModel::build_fbank_session(&model_path, ExecutionMode::Cpu, threads)
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
