use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;

pub struct SegmentationModel {
    session: Session,
    window_samples: usize,
    step_samples: usize,
    sample_rate: usize,
}

impl SegmentationModel {
    /// Load a segmentation-3.0 ONNX model
    pub fn new(model_path: &str, step_duration: f32) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(model_path)?;

        let sample_rate = 16000;
        let window_duration = 10.0;
        let window_samples = (window_duration * sample_rate as f32) as usize;
        let step_samples = (step_duration * sample_rate as f32) as usize;

        Ok(Self {
            session,
            window_samples,
            step_samples,
            sample_rate,
        })
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
        let input_array =
            ndarray::Array3::from_shape_vec((1, 1, self.window_samples), window.to_vec()).unwrap();
        let input_tensor = Tensor::from_array(input_array)?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // Shape derefs to [i64], index directly
        let frames = shape[1] as usize;
        let classes = shape[2] as usize;

        Ok(Array2::from_shape_vec((frames, classes), data.to_vec()).unwrap())
    }
}
