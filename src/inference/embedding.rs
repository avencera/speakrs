use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, ArrayView2, s};
use ort::session::Session;
use ort::value::TensorRef;

use crate::inference::{ExecutionMode, with_execution_mode};

const PRIMARY_BATCH_SIZE: usize = 32;
const CHUNK_SPEAKER_BATCH_SIZE: usize = 3;
const FBANK_FRAMES: usize = 998;
const FBANK_FEATURES: usize = 80;

pub struct MaskedEmbeddingInput<'a> {
    pub audio: &'a [f32],
    pub mask: &'a [f32],
    pub clean_mask: Option<&'a [f32]>,
}

pub(crate) struct SplitTailInput<'a> {
    pub fbank: &'a Array2<f32>,
    pub weights: &'a [f32],
}

pub struct EmbeddingModel {
    model_path: String,
    mode: ExecutionMode,
    session: Session,
    primary_batched_session: Option<Session>,
    split_fbank_session: Option<Session>,
    split_fbank_batched_session: Option<Session>,
    split_tail_session: Option<Session>,
    split_tail_batched_session: Option<Session>,
    split_primary_tail_batched_session: Option<Session>,
    waveform_buffer: Array3<f32>,
    weights_buffer: Array2<f32>,
    primary_batch_waveform_buffer: Array3<f32>,
    primary_batch_weights_buffer: Array2<f32>,
    split_waveform_buffer: Array3<f32>,
    split_fbank_batch_buffer: Array3<f32>,
    split_feature_batch_buffer: Array3<f32>,
    split_weights_batch_buffer: Array2<f32>,
    split_primary_feature_batch_buffer: Array3<f32>,
    split_primary_weights_batch_buffer: Array2<f32>,
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    min_num_samples: usize,
}

impl EmbeddingModel {
    /// Load the WeSpeaker embedding model
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        Self::with_mode(model_path, ExecutionMode::ExactCpu)
    }

    /// Load the WeSpeaker embedding model with the requested execution mode
    pub fn with_mode(model_path: &str, mode: ExecutionMode) -> Result<Self, ort::Error> {
        let metadata_path = Path::new(model_path)
            .with_extension("min_num_samples.txt")
            .to_string_lossy()
            .into_owned();
        let split_fbank_path = split_fbank_model_path(model_path);
        let split_fbank_batched_path = split_fbank_batched_model_path(model_path);
        let split_tail_path = split_tail_model_path(model_path, 1);
        let split_tail_batched_path = split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path = split_tail_model_path(model_path, PRIMARY_BATCH_SIZE);
        let use_split_backend =
            mode == ExecutionMode::CoreMl && split_fbank_path.exists() && split_tail_path.exists();

        Ok(Self {
            model_path: model_path.to_owned(),
            mode,
            session: Self::build_session(model_path, Self::single_execution_mode(mode))?,
            primary_batched_session: batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_batched_session(path.to_str().unwrap(), mode))
                .transpose()?,
            split_fbank_session: use_split_backend
                .then(|| {
                    Self::build_fbank_session(
                        split_fbank_path.to_str().unwrap(),
                        ExecutionMode::ExactCpu,
                    )
                })
                .transpose()?,
            split_fbank_batched_session: use_split_backend
                .then_some(split_fbank_batched_path)
                .filter(|path| path.exists())
                .map(|path| {
                    Self::build_fbank_session(path.to_str().unwrap(), ExecutionMode::ExactCpu)
                })
                .transpose()?,
            split_tail_session: use_split_backend
                .then(|| Self::build_session(split_tail_path.to_str().unwrap(), mode))
                .transpose()?,
            split_tail_batched_session: use_split_backend
                .then_some(split_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(path.to_str().unwrap(), mode))
                .transpose()?,
            split_primary_tail_batched_session: use_split_backend
                .then_some(split_primary_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(path.to_str().unwrap(), mode))
                .transpose()?,
            waveform_buffer: Array3::zeros((1, 1, 160_000)),
            weights_buffer: Array2::zeros((1, 589)),
            primary_batch_waveform_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, 160_000)),
            primary_batch_weights_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
            split_waveform_buffer: Array3::zeros((1, 1, 160_000)),
            split_fbank_batch_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, 160_000)),
            split_feature_batch_buffer: Array3::zeros((
                CHUNK_SPEAKER_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            split_weights_batch_buffer: Array2::zeros((CHUNK_SPEAKER_BATCH_SIZE, 589)),
            split_primary_feature_batch_buffer: Array3::zeros((
                PRIMARY_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            split_primary_weights_batch_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
            sample_rate: 16_000,
            window_samples: 160_000,
            mask_frames: 589,
            min_num_samples: read_min_num_samples(&metadata_path).unwrap_or(400),
        })
    }

    fn build_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_memory_pattern(false)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    fn build_fbank_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(1);
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_memory_pattern(false)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    fn single_execution_mode(mode: ExecutionMode) -> ExecutionMode {
        match mode {
            // keep single embeddings on the CPU path until the CoreML session
            // reproduces the Python fixture numerically
            ExecutionMode::CoreMl => ExecutionMode::ExactCpu,
            _ => mode,
        }
    }

    fn build_batched_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        Self::build_session(model_path, Self::single_execution_mode(mode))
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
        } else {
            1
        }
    }

    pub fn best_batch_len(&self, pending_len: usize) -> usize {
        if pending_len >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            pending_len.min(1)
        }
    }

    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.session =
            Self::build_session(&self.model_path, Self::single_execution_mode(self.mode))?;
        self.primary_batched_session = batched_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_batched_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        let split_fbank_path = split_fbank_model_path(&self.model_path);
        let split_tail_path = split_tail_model_path(&self.model_path, 1);
        let split_tail_batched_path =
            split_tail_model_path(&self.model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path =
            split_tail_model_path(&self.model_path, PRIMARY_BATCH_SIZE);
        let use_split_backend = self.mode == ExecutionMode::CoreMl
            && split_fbank_path.exists()
            && split_tail_path.exists();
        let split_fbank_batched_path = split_fbank_batched_model_path(&self.model_path);
        self.split_fbank_session = use_split_backend
            .then(|| {
                Self::build_fbank_session(
                    split_fbank_path.to_str().unwrap(),
                    ExecutionMode::ExactCpu,
                )
            })
            .transpose()?;
        self.split_fbank_batched_session = use_split_backend
            .then_some(split_fbank_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_fbank_session(path.to_str().unwrap(), ExecutionMode::ExactCpu))
            .transpose()?;
        self.split_tail_session = use_split_backend
            .then(|| Self::build_session(split_tail_path.to_str().unwrap(), self.mode))
            .transpose()?;
        self.split_tail_batched_session = use_split_backend
            .then_some(split_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        self.split_primary_tail_batched_session = use_split_backend
            .then_some(split_primary_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        Ok(())
    }

    pub fn prefers_chunk_embedding_path(&self) -> bool {
        self.split_fbank_session.is_some() && self.split_tail_session.is_some()
    }

    pub(crate) fn split_primary_batch_size(&self) -> usize {
        if self.split_primary_tail_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            0
        }
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

        let mut stacked = Array2::<f32>::zeros((inputs.len(), 256));
        for (idx, input) in inputs.iter().enumerate() {
            let embedding = self.embed_masked(input.audio, input.mask, input.clean_mask)?;
            stacked.row_mut(idx).assign(&embedding);
        }
        Ok(stacked)
    }

    pub fn embed_chunk_speakers(
        &mut self,
        audio: &[f32],
        segmentations: ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
    ) -> Result<Array2<f32>, ort::Error> {
        let speaker_count = segmentations.ncols();
        let mut embeddings = Array2::<f32>::zeros((speaker_count, 256));
        if !self.prefers_chunk_embedding_path() {
            for speaker_idx in 0..speaker_count {
                let mask = segmentations.column(speaker_idx).to_owned();
                let clean_mask = clean_masks.column(speaker_idx).to_owned();
                let embedding = self.embed_masked(
                    audio,
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                )?;
                embeddings.row_mut(speaker_idx).assign(&embedding);
            }
            return Ok(embeddings);
        }

        let fbank = self.compute_chunk_fbank(audio)?;
        if speaker_count == CHUNK_SPEAKER_BATCH_SIZE && self.split_tail_batched_session.is_some() {
            return self.embed_tail_batch(&fbank, &segmentations, clean_masks, audio.len());
        }

        for speaker_idx in 0..speaker_count {
            let mask = segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let used_mask = select_mask(
                mask.as_slice().unwrap(),
                Some(clean_mask.as_slice().unwrap()),
                audio.len(),
                self.min_num_samples,
            );
            let embedding = self.embed_tail_single(&fbank, used_mask)?;
            embeddings.row_mut(speaker_idx).assign(&embedding);
        }

        Ok(embeddings)
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

    pub fn compute_chunk_fbank(&mut self, audio: &[f32]) -> Result<Array2<f32>, ort::Error> {
        self.split_waveform_buffer.fill(0.0);
        let copy_len = audio.len().min(self.window_samples);
        self.split_waveform_buffer
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        let waveform_tensor = TensorRef::from_array_view(self.split_waveform_buffer.view())?;
        let outputs = self
            .split_fbank_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["waveform" => waveform_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let frames = shape[1] as usize;
        let features = shape[2] as usize;
        Ok(Array2::from_shape_vec((frames, features), data.to_vec()).unwrap())
    }

    pub fn compute_chunk_fbanks_batch(
        &mut self,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        if self.split_fbank_batched_session.is_none() {
            return audios
                .iter()
                .map(|audio| self.compute_chunk_fbank(audio))
                .collect();
        }

        let mut results = Vec::with_capacity(audios.len());
        for batch_start in (0..audios.len()).step_by(PRIMARY_BATCH_SIZE) {
            let batch_end = (batch_start + PRIMARY_BATCH_SIZE).min(audios.len());
            let batch = &audios[batch_start..batch_end];

            if batch.len() == PRIMARY_BATCH_SIZE {
                self.split_fbank_batch_buffer.fill(0.0);
                for (idx, audio) in batch.iter().enumerate() {
                    let copy_len = audio.len().min(self.window_samples);
                    self.split_fbank_batch_buffer
                        .slice_mut(s![idx, 0, ..copy_len])
                        .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
                }

                let waveform_tensor =
                    TensorRef::from_array_view(self.split_fbank_batch_buffer.view())?;
                let outputs = self
                    .split_fbank_batched_session
                    .as_mut()
                    .unwrap()
                    .run(ort::inputs!["waveform" => waveform_tensor])?;
                let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
                let frames = shape[1] as usize;
                let features = shape[2] as usize;
                let flat = data.to_vec();
                let stride = frames * features;
                for idx in 0..PRIMARY_BATCH_SIZE {
                    let start = idx * stride;
                    results.push(
                        Array2::from_shape_vec(
                            (frames, features),
                            flat[start..start + stride].to_vec(),
                        )
                        .unwrap(),
                    );
                }
            } else {
                for audio in batch {
                    self.split_waveform_buffer.fill(0.0);
                    let copy_len = audio.len().min(self.window_samples);
                    self.split_waveform_buffer
                        .slice_mut(s![0, 0, ..copy_len])
                        .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
                    let waveform_tensor =
                        TensorRef::from_array_view(self.split_waveform_buffer.view())?;
                    let outputs = self
                        .split_fbank_session
                        .as_mut()
                        .unwrap()
                        .run(ort::inputs!["waveform" => waveform_tensor])?;
                    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
                    let frames = shape[1] as usize;
                    let features = shape[2] as usize;
                    results
                        .push(Array2::from_shape_vec((frames, features), data.to_vec()).unwrap());
                }
            }
        }

        Ok(results)
    }

    pub fn has_batched_fbank(&self) -> bool {
        self.split_fbank_batched_session.is_some()
    }

    pub(crate) fn select_chunk_mask<'a>(
        &self,
        mask: &'a [f32],
        clean_mask: Option<&'a [f32]>,
        num_samples: usize,
    ) -> &'a [f32] {
        select_mask(mask, clean_mask, num_samples, self.min_num_samples)
    }

    pub(crate) fn embed_tail_batch_inputs(
        &mut self,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, ort::Error> {
        debug_assert!(self.split_primary_tail_batched_session.is_some());
        debug_assert!(inputs.len() <= PRIMARY_BATCH_SIZE);

        self.split_primary_feature_batch_buffer.fill(0.0);
        self.split_primary_weights_batch_buffer.fill(0.0);

        for (batch_idx, input) in inputs.iter().enumerate() {
            self.split_primary_feature_batch_buffer
                .slice_mut(s![batch_idx, ..input.fbank.nrows(), ..input.fbank.ncols()])
                .assign(input.fbank);
            Self::prepare_weights(
                batch_idx,
                input.weights,
                self.mask_frames,
                &mut self.split_primary_weights_batch_buffer.view_mut(),
            );
        }

        let fbank_tensor =
            TensorRef::from_array_view(self.split_primary_feature_batch_buffer.view())?;
        let weights_tensor =
            TensorRef::from_array_view(self.split_primary_weights_batch_buffer.view())?;
        let outputs = self
            .split_primary_tail_batched_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let batch = Array2::from_shape_vec((PRIMARY_BATCH_SIZE, 256), data.to_vec()).unwrap();
        Ok(batch.slice(s![0..inputs.len(), ..]).to_owned())
    }

    fn embed_tail_single(
        &mut self,
        fbank: &Array2<f32>,
        weights: &[f32],
    ) -> Result<Array1<f32>, ort::Error> {
        self.split_feature_batch_buffer.fill(0.0);
        self.split_weights_batch_buffer.fill(0.0);
        self.split_feature_batch_buffer
            .slice_mut(s![0, ..fbank.nrows(), ..fbank.ncols()])
            .assign(fbank);
        Self::prepare_weights(
            0,
            weights,
            self.mask_frames,
            &mut self.split_weights_batch_buffer.view_mut(),
        );
        let feature_slice = self.split_feature_batch_buffer.slice(s![0..1, .., ..]);
        let weight_slice = self.split_weights_batch_buffer.slice(s![0..1, ..]);
        let fbank_tensor = TensorRef::from_array_view(feature_slice.view())?;
        let weights_tensor = TensorRef::from_array_view(weight_slice.view())?;
        let outputs = self
            .split_tail_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn embed_tail_batch(
        &mut self,
        fbank: &Array2<f32>,
        segmentations: &ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
        num_samples: usize,
    ) -> Result<Array2<f32>, ort::Error> {
        self.split_feature_batch_buffer.fill(0.0);
        self.split_weights_batch_buffer.fill(0.0);

        for speaker_idx in 0..segmentations.ncols() {
            self.split_feature_batch_buffer
                .slice_mut(s![speaker_idx, ..fbank.nrows(), ..fbank.ncols()])
                .assign(fbank);
            let mask = segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let used_mask = select_mask(
                mask.as_slice().unwrap(),
                Some(clean_mask.as_slice().unwrap()),
                num_samples,
                self.min_num_samples,
            );
            Self::prepare_weights(
                speaker_idx,
                used_mask,
                self.mask_frames,
                &mut self.split_weights_batch_buffer.view_mut(),
            );
        }

        let fbank_tensor = TensorRef::from_array_view(self.split_feature_batch_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.split_weights_batch_buffer.view())?;
        let outputs = self
            .split_tail_batched_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array2::from_shape_vec((segmentations.ncols(), 256), data.to_vec()).unwrap())
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

fn split_fbank_model_path(model_path: &str) -> std::path::PathBuf {
    let path = Path::new(model_path);
    path.with_file_name("wespeaker-fbank.onnx")
}

fn split_fbank_batched_model_path(model_path: &str) -> std::path::PathBuf {
    let path = Path::new(model_path);
    path.with_file_name("wespeaker-fbank-b32.onnx")
}

fn split_tail_model_path(model_path: &str, batch_size: usize) -> std::path::PathBuf {
    let path = Path::new(model_path);
    if batch_size == 1 {
        path.with_file_name("wespeaker-voxceleb-resnet34-tail.onnx")
    } else {
        path.with_file_name(format!(
            "wespeaker-voxceleb-resnet34-tail-b{batch_size}.onnx"
        ))
    }
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
