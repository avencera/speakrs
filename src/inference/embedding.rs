use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, s};
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;

const PRIMARY_BATCH_SIZE: usize = 32;
const SECONDARY_BATCH_SIZE: usize = 16;

pub struct MaskedEmbeddingInput<'a> {
    pub audio: &'a [f32],
    pub mask: &'a [f32],
    pub clean_mask: Option<&'a [f32]>,
}

pub struct EmbeddingModel {
    model_path: String,
    session: Session,
    primary_batched_session: Option<Session>,
    secondary_batched_session: Option<Session>,
    waveform_buffer: Array3<f32>,
    weights_buffer: Array2<f32>,
    primary_batch_waveform_buffer: Array3<f32>,
    primary_batch_weights_buffer: Array2<f32>,
    secondary_batch_waveform_buffer: Array3<f32>,
    secondary_batch_weights_buffer: Array2<f32>,
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    min_num_samples: usize,
}

impl EmbeddingModel {
    /// Load the WeSpeaker embedding model
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        let metadata_path = Path::new(model_path)
            .with_extension("min_num_samples.txt")
            .to_string_lossy()
            .into_owned();

        Ok(Self {
            model_path: model_path.to_owned(),
            session: Self::build_session(model_path)?,
            primary_batched_session: batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_batched_session(path.to_str().unwrap()))
                .transpose()?,
            secondary_batched_session: batched_model_path(model_path, SECONDARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_batched_session(path.to_str().unwrap()))
                .transpose()?,
            waveform_buffer: Array3::zeros((1, 1, 160_000)),
            weights_buffer: Array2::zeros((1, 589)),
            primary_batch_waveform_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, 160_000)),
            primary_batch_weights_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
            secondary_batch_waveform_buffer: Array3::zeros((SECONDARY_BATCH_SIZE, 1, 160_000)),
            secondary_batch_weights_buffer: Array2::zeros((SECONDARY_BATCH_SIZE, 589)),
            sample_rate: 16_000,
            window_samples: 160_000,
            mask_frames: 589,
            min_num_samples: read_min_num_samples(&metadata_path).unwrap_or(400),
        })
    }

    fn build_session(model_path: &str) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_memory_pattern(false)?;
        let builder = builder
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?;
        let mut builder = builder;
        builder.commit_from_file(model_path)
    }

    fn build_batched_session(model_path: &str) -> Result<Session, ort::Error> {
        #[cfg(feature = "coreml")]
        {
            let cache_dir = std::env::temp_dir().join("speakrs-coreml-cache");
            let _ = std::fs::create_dir_all(&cache_dir);
            let builder = Session::builder()?
                .with_independent_thread_pool()?
                .with_intra_threads(1)?
                .with_inter_threads(1)?
                .with_execution_providers([ep::CoreML::default()
                    .with_static_input_shapes(true)
                    .with_model_cache_dir(cache_dir.display().to_string())
                    .build()])?;
            let mut builder = builder;
            return builder.commit_from_file(model_path);
        }

        #[cfg(not(feature = "coreml"))]
        {
            Self::build_session(model_path)
        }
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn min_num_samples(&self) -> usize {
        self.min_num_samples
    }

    pub fn primary_batch_size(&self) -> usize {
        if self.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else if self.secondary_batched_session.is_some() {
            SECONDARY_BATCH_SIZE
        } else {
            1
        }
    }

    pub fn best_batch_len(&self, pending_len: usize) -> usize {
        if pending_len >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else if pending_len >= SECONDARY_BATCH_SIZE && self.secondary_batched_session.is_some() {
            SECONDARY_BATCH_SIZE
        } else {
            pending_len.min(1)
        }
    }

    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.session = Self::build_session(&self.model_path)?;
        self.primary_batched_session = batched_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_batched_session(path.to_str().unwrap()))
            .transpose()?;
        self.secondary_batched_session = batched_model_path(&self.model_path, SECONDARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_batched_session(path.to_str().unwrap()))
            .transpose()?;
        Ok(())
    }

    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let weights = vec![1.0; self.mask_frames];
        self.embed_single(audio, &weights)
    }

    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        let used_mask = select_mask(mask, clean_mask, audio.len(), self.min_num_samples);
        self.embed_single(audio, used_mask)
    }

    pub fn embed_batch(
        &mut self,
        inputs: &[MaskedEmbeddingInput<'_>],
    ) -> Result<Array2<f32>, ort::Error> {
        if inputs.len() == PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
            return Self::embed_full_batch(
                inputs,
                self.window_samples,
                self.mask_frames,
                self.min_num_samples,
                &mut self.primary_batched_session,
                self.primary_batch_waveform_buffer.view_mut(),
                self.primary_batch_weights_buffer.view_mut(),
            );
        }

        if inputs.len() == SECONDARY_BATCH_SIZE && self.secondary_batched_session.is_some() {
            return Self::embed_full_batch(
                inputs,
                self.window_samples,
                self.mask_frames,
                self.min_num_samples,
                &mut self.secondary_batched_session,
                self.secondary_batch_waveform_buffer.view_mut(),
                self.secondary_batch_weights_buffer.view_mut(),
            );
        }

        let mut stacked = Array2::<f32>::zeros((inputs.len(), 256));
        for (idx, input) in inputs.iter().enumerate() {
            let embedding = self.embed_masked(input.audio, input.mask, input.clean_mask)?;
            stacked.row_mut(idx).assign(&embedding);
        }
        Ok(stacked)
    }

    fn embed_full_batch(
        inputs: &[MaskedEmbeddingInput<'_>],
        window_samples: usize,
        mask_frames: usize,
        min_num_samples: usize,
        session: &mut Option<Session>,
        mut waveform_buffer: ndarray::ArrayViewMut3<f32>,
        mut weights_buffer: ndarray::ArrayViewMut2<f32>,
    ) -> Result<Array2<f32>, ort::Error> {
        waveform_buffer.fill(0.0);
        weights_buffer.fill(0.0);

        for (batch_idx, input) in inputs.iter().enumerate() {
            let used_mask = select_mask(
                input.mask,
                input.clean_mask,
                input.audio.len(),
                min_num_samples,
            );
            Self::prepare_waveform(batch_idx, input.audio, window_samples, &mut waveform_buffer);
            Self::prepare_weights(batch_idx, used_mask, mask_frames, &mut weights_buffer);
        }

        let waveform_tensor = TensorRef::from_array_view(waveform_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(weights_buffer.view())?;
        let outputs = session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array2::from_shape_vec((inputs.len(), 256), data.to_vec()).unwrap())
    }

    fn embed_single(&mut self, audio: &[f32], weights: &[f32]) -> Result<Array1<f32>, ort::Error> {
        self.waveform_buffer.fill(0.0);
        self.weights_buffer.fill(0.0);
        let copy_len = audio.len().min(self.window_samples);
        self.waveform_buffer
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        self.prepare_single_weights(weights);

        let waveform_tensor = TensorRef::from_array_view(self.waveform_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.weights_buffer.view())?;
        let outputs = self
            .session
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn prepare_waveform(
        batch_idx: usize,
        audio: &[f32],
        window_samples: usize,
        waveform_buffer: &mut ndarray::ArrayViewMut3<f32>,
    ) {
        let copy_len = audio.len().min(window_samples);
        waveform_buffer
            .slice_mut(s![batch_idx, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
    }

    fn prepare_weights(
        batch_idx: usize,
        weights: &[f32],
        mask_frames: usize,
        weights_buffer: &mut ndarray::ArrayViewMut2<f32>,
    ) {
        if weights.len() == mask_frames {
            weights_buffer
                .row_mut(batch_idx)
                .assign(&ndarray::ArrayView1::from(weights));
            return;
        }

        let copy_len = weights.len().min(mask_frames);
        weights_buffer
            .slice_mut(s![batch_idx, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&weights[..copy_len]));
    }

    fn prepare_single_weights(&mut self, weights: &[f32]) {
        if weights.len() == self.mask_frames {
            self.weights_buffer
                .row_mut(0)
                .assign(&ndarray::ArrayView1::from(weights));
            return;
        }

        let copy_len = weights.len().min(self.mask_frames);
        self.weights_buffer
            .slice_mut(s![0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&weights[..copy_len]));
    }
}

fn batched_model_path(model_path: &str, batch_size: usize) -> Option<std::path::PathBuf> {
    let path = Path::new(model_path);
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}

fn read_min_num_samples(path: &str) -> Option<usize> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn select_mask<'a>(
    mask: &'a [f32],
    clean_mask: Option<&'a [f32]>,
    num_samples: usize,
    min_num_samples: usize,
) -> &'a [f32] {
    let Some(clean_mask) = clean_mask else {
        return mask;
    };

    if clean_mask.len() != mask.len() || num_samples == 0 {
        return mask;
    }

    let min_mask_frames = (mask.len() * min_num_samples).div_ceil(num_samples) as f32;
    let clean_weight: f32 = clean_mask.iter().copied().sum();
    if clean_weight > min_mask_frames {
        clean_mask
    } else {
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_mask_prefers_clean_mask_when_it_is_long_enough() {
        let mask = [1.0, 1.0, 1.0, 0.0];
        let clean = [1.0, 1.0, 1.0, 0.0];

        let selected = select_mask(&mask, Some(&clean), 16_000, 6_000);

        assert_eq!(selected, clean);
    }

    #[test]
    fn select_mask_falls_back_to_full_mask_when_clean_mask_is_too_short() {
        let mask = [1.0, 1.0, 1.0, 0.0];
        let clean = [1.0, 0.0, 0.0, 0.0];

        let selected = select_mask(&mask, Some(&clean), 16_000, 6_000);

        assert_eq!(selected, mask);
    }
}
