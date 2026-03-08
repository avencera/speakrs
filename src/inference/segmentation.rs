use ndarray::Array2;
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;

pub struct SegmentationModel {
    model_path: String,
    session: Session,
    input_buffer: ndarray::Array3<f32>,
    window_samples: usize,
    step_samples: usize,
    sample_rate: usize,
}

impl SegmentationModel {
    /// Load a segmentation-3.0 ONNX model
    pub fn new(model_path: &str, step_duration: f32) -> Result<Self, ort::Error> {
        let sample_rate = 16000;
        let window_duration = 10.0;
        let window_samples = (window_duration * sample_rate as f32) as usize;
        let step_samples = (step_duration * sample_rate as f32) as usize;

        Ok(Self {
            model_path: model_path.to_owned(),
            session: Self::build_session(model_path)?,
            input_buffer: ndarray::Array3::zeros((1, 1, window_samples)),
            window_samples,
            step_samples,
            sample_rate,
        })
    }

    fn build_session(model_path: &str) -> Result<Session, ort::Error> {
        let mut builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(Self::available_threads().min(6))?
            .with_inter_threads(1)?
            .with_memory_pattern(false)?
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?;
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
        self.session = Self::build_session(&self.model_path)?;
        Ok(())
    }

    /// Run segmentation on audio, returning raw logits per window
    ///
    /// Returns Vec<Array2<f32>> where each element is [frames, 7] logits
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>, ort::Error> {
        let mut results = Vec::new();
        let mut offset = 0;

        while offset + self.window_samples <= audio.len() {
            let window = &audio[offset..offset + self.window_samples];
            let logits = self.run_window(window)?;
            results.push(logits);
            offset += self.step_samples;
        }

        // handle last partial window by zero-padding
        if offset < audio.len() && audio.len() > self.window_samples {
            let mut padded = vec![0.0f32; self.window_samples];
            let remaining = audio.len() - offset;
            padded[..remaining].copy_from_slice(&audio[offset..]);
            let logits = self.run_window(&padded)?;
            results.push(logits);
        }

        Ok(results)
    }

    fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, ort::Error> {
        self.input_buffer.fill(0.0);
        self.input_buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_tensor = TensorRef::from_array_view(self.input_buffer.view())?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // Shape derefs to [i64], index directly
        let frames = shape[1] as usize;
        let classes = shape[2] as usize;

        Ok(Array2::from_shape_vec((frames, classes), data.to_vec()).unwrap())
    }
}
