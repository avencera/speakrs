use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;

#[cfg(feature = "native-coreml")]
use crate::inference::coreml::{CoreMlModel, coreml_model_path, coreml_model_path_f16};
use crate::inference::{ExecutionMode, with_execution_mode};
#[cfg(feature = "native-coreml")]
use objc2_core_ml::MLComputeUnits;

const PRIMARY_BATCH_SIZE: usize = 32;

pub struct SegmentationModel {
    model_path: String,
    mode: ExecutionMode,
    session: Session,
    primary_batched_session: Option<Session>,
    #[cfg(feature = "native-coreml")]
    native_session: Option<CoreMlModel>,
    #[cfg(feature = "native-coreml")]
    native_batched_session: Option<CoreMlModel>,
    input_buffer: ndarray::Array3<f32>,
    primary_batch_input_buffer: ndarray::Array3<f32>,
    window_samples: usize,
    step_samples: usize,
    sample_rate: usize,
}

impl SegmentationModel {
    /// Load a segmentation-3.0 ONNX model
    pub fn new(model_path: &str, step_duration: f32) -> Result<Self, ort::Error> {
        Self::with_mode(model_path, step_duration, ExecutionMode::ExactCpu)
    }

    /// Load a segmentation-3.0 ONNX model with the requested execution mode
    pub fn with_mode(
        model_path: &str,
        step_duration: f32,
        mode: ExecutionMode,
    ) -> Result<Self, ort::Error> {
        let sample_rate = 16000;
        let window_duration = 10.0;
        let window_samples = (window_duration * sample_rate as f32) as usize;
        let step_samples = (step_duration * sample_rate as f32) as usize;

        Ok(Self {
            model_path: model_path.to_owned(),
            mode,
            session: Self::build_session(model_path, mode)?,
            primary_batched_session: batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(path.to_str().unwrap(), mode))
                .transpose()?,
            #[cfg(feature = "native-coreml")]
            native_session: Self::load_native_coreml(model_path, mode),
            #[cfg(feature = "native-coreml")]
            native_batched_session: Self::load_native_coreml_batched(model_path, mode),
            input_buffer: ndarray::Array3::zeros((1, 1, window_samples)),
            primary_batch_input_buffer: ndarray::Array3::zeros((
                PRIMARY_BATCH_SIZE,
                1,
                window_samples,
            )),
            window_samples,
            step_samples,
            sample_rate,
        })
    }

    fn build_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(Self::available_threads().min(6))?
            .with_inter_threads(1)?
            .with_memory_pattern(false)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    fn available_threads() -> usize {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    pub fn step_samples(&self) -> usize {
        self.step_samples
    }

    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.session = Self::build_session(&self.model_path, self.mode)?;
        self.primary_batched_session = batched_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        #[cfg(feature = "native-coreml")]
        {
            self.native_session = Self::load_native_coreml(&self.model_path, self.mode);
            self.native_batched_session =
                Self::load_native_coreml_batched(&self.model_path, self.mode);
        }
        Ok(())
    }

    /// Run segmentation on audio, returning raw logits per window
    ///
    /// Returns Vec<Array2<f32>> where each element is [frames, 7] logits
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>, ort::Error> {
        let mut windows = Vec::new();
        let mut offset = 0;

        while offset + self.window_samples <= audio.len() {
            windows.push(audio[offset..offset + self.window_samples].to_vec());
            offset += self.step_samples;
        }

        // handle last partial window by zero-padding
        if offset < audio.len() && audio.len() > self.window_samples {
            let mut padded = vec![0.0f32; self.window_samples];
            let remaining = audio.len() - offset;
            padded[..remaining].copy_from_slice(&audio[offset..]);
            windows.push(padded);
        }

        let mut results = Vec::with_capacity(windows.len());
        let mut next_idx = 0;
        while next_idx < windows.len() {
            let remaining = windows.len() - next_idx;
            if remaining >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
                let batch = self.run_batch(&windows[next_idx..next_idx + PRIMARY_BATCH_SIZE])?;
                results.extend(batch);
                next_idx += PRIMARY_BATCH_SIZE;
                continue;
            }

            let logits = self.run_window(&windows[next_idx])?;
            results.push(logits);
            next_idx += 1;
        }

        Ok(results)
    }

    fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, ort::Error> {
        #[cfg(feature = "native-coreml")]
        if let Some(ref native) = self.native_session {
            return Self::run_native_single(native, window, &mut self.input_buffer);
        }

        self.input_buffer.fill(0.0);
        self.input_buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_tensor = TensorRef::from_array_view(self.input_buffer.view())?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let frames = shape[1] as usize;
        let classes = shape[2] as usize;

        Ok(Array2::from_shape_vec((frames, classes), data.to_vec()).unwrap())
    }

    fn run_batch(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Array2<f32>>, ort::Error> {
        #[cfg(feature = "native-coreml")]
        if let Some(ref native) = self.native_batched_session {
            return Self::run_native_batch(native, windows, &mut self.primary_batch_input_buffer);
        }

        self.primary_batch_input_buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            self.primary_batch_input_buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(window.as_slice()));
        }
        let input_tensor = TensorRef::from_array_view(self.primary_batch_input_buffer.view())?;

        let outputs = self
            .primary_batched_session
            .as_mut()
            .unwrap()
            .run(ort::inputs![input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let batch = shape[0] as usize;
        let frames = shape[1] as usize;
        let classes = shape[2] as usize;
        let flat = data.to_vec();

        Ok((0..batch)
            .map(|batch_idx| {
                let start = batch_idx * frames * classes;
                let end = start + frames * classes;
                Array2::from_shape_vec((frames, classes), flat[start..end].to_vec()).unwrap()
            })
            .collect())
    }

    #[cfg(feature = "native-coreml")]
    fn resolve_coreml_path(model_path: &str, mode: ExecutionMode) -> Option<std::path::PathBuf> {
        match mode {
            ExecutionMode::CoreMl => Some(coreml_model_path(model_path)),
            ExecutionMode::MiniCoreMl => Some(coreml_model_path_f16(model_path)),
            _ => None,
        }
    }

    #[cfg(feature = "native-coreml")]
    fn compute_units_for_mode(mode: ExecutionMode) -> MLComputeUnits {
        match mode {
            ExecutionMode::MiniCoreMl => CoreMlModel::fp16_compute_units(),
            _ => CoreMlModel::default_compute_units(),
        }
    }

    #[cfg(feature = "native-coreml")]
    fn load_native_coreml(model_path: &str, mode: ExecutionMode) -> Option<CoreMlModel> {
        let coreml_path = Self::resolve_coreml_path(model_path, mode)?;
        if !coreml_path.exists() {
            eprintln!(
                "warning: native CoreML model not found at {}, falling back to ORT CPU",
                coreml_path.display()
            );
            return None;
        }
        match CoreMlModel::load(&coreml_path, Self::compute_units_for_mode(mode), "output") {
            Ok(model) => Some(model),
            Err(e) => {
                eprintln!("warning: failed to load native CoreML segmentation: {e}");
                None
            }
        }
    }

    #[cfg(feature = "native-coreml")]
    fn load_native_coreml_batched(model_path: &str, mode: ExecutionMode) -> Option<CoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::MiniCoreMl) {
            return None;
        }
        let batched_onnx = batched_model_path(model_path, PRIMARY_BATCH_SIZE)?;
        let onnx_str = batched_onnx.to_str().unwrap();
        let coreml_path = match mode {
            ExecutionMode::MiniCoreMl => coreml_model_path_f16(onnx_str),
            _ => coreml_model_path(onnx_str),
        };
        if !coreml_path.exists() {
            return None;
        }
        match CoreMlModel::load(&coreml_path, Self::compute_units_for_mode(mode), "output") {
            Ok(model) => Some(model),
            Err(e) => {
                eprintln!("warning: failed to load native CoreML batched segmentation: {e}");
                None
            }
        }
    }

    #[cfg(feature = "native-coreml")]
    fn run_native_single(
        native: &CoreMlModel,
        window: &[f32],
        buffer: &mut ndarray::Array3<f32>,
    ) -> Result<Array2<f32>, ort::Error> {
        buffer.fill(0.0);
        buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_data = buffer.as_slice().unwrap();
        let shape = [buffer.shape()[0], buffer.shape()[1], buffer.shape()[2]];

        let (data, out_shape) = native
            .predict(&[("input", &shape, input_data)])
            .map_err(|e| ort::Error::new(e.to_string()))?;

        let frames = out_shape[1];
        let classes = out_shape[2];
        Ok(Array2::from_shape_vec((frames, classes), data).unwrap())
    }

    #[cfg(feature = "native-coreml")]
    fn run_native_batch(
        native: &CoreMlModel,
        windows: &[Vec<f32>],
        buffer: &mut ndarray::Array3<f32>,
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(window.as_slice()));
        }
        let input_data = buffer.as_slice().unwrap();
        let shape = [buffer.shape()[0], buffer.shape()[1], buffer.shape()[2]];

        let (data, out_shape) = native
            .predict(&[("input", &shape, input_data)])
            .map_err(|e| ort::Error::new(e.to_string()))?;

        let batch = out_shape[0];
        let frames = out_shape[1];
        let classes = out_shape[2];

        Ok((0..batch)
            .map(|batch_idx| {
                let start = batch_idx * frames * classes;
                let end = start + frames * classes;
                Array2::from_shape_vec((frames, classes), data[start..end].to_vec()).unwrap()
            })
            .collect())
    }
}

fn batched_model_path(model_path: &str, batch_size: usize) -> Option<std::path::PathBuf> {
    let path = std::path::Path::new(model_path);
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}
