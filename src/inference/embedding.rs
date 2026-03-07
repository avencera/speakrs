use kaldi_native_fbank::FrameOptions;
use kaldi_native_fbank::fbank::{FbankComputer, FbankOptions};
use kaldi_native_fbank::mel::MelOptions;
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;

pub struct EmbeddingModel {
    session: Session,
    sample_rate: usize,
    num_mel_bins: usize,
}

impl EmbeddingModel {
    /// Load a WeSpeaker ONNX embedding model
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(model_path)?;

        Ok(Self {
            session,
            sample_rate: 16000,
            num_mel_bins: 80,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Extract a 256-dim embedding from an audio segment
    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let features = self.compute_fbank(audio);

        if features.is_empty() {
            return Ok(Array1::zeros(256));
        }

        let num_frames = features.len() / self.num_mel_bins;
        let input_array =
            ndarray::Array3::from_shape_vec((1, num_frames, self.num_mel_bins), features).unwrap();
        let input_tensor = Tensor::from_array(input_array)?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(Array1::from_vec(data.to_vec()))
    }

    fn compute_fbank(&self, audio: &[f32]) -> Vec<f32> {
        let frame_opts = FrameOptions {
            samp_freq: self.sample_rate as f32,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 0.0,
            ..Default::default()
        };

        let mel_opts = MelOptions {
            num_bins: self.num_mel_bins,
            ..Default::default()
        };

        let fbank_opts = FbankOptions {
            frame_opts,
            mel_opts,
            use_energy: false,
            ..Default::default()
        };

        let computer = FbankComputer::new(fbank_opts).expect("failed to create fbank computer");
        let mut online = OnlineFeature::new(FeatureComputer::Fbank(computer));

        online.accept_waveform(self.sample_rate as f32, audio);
        online.input_finished();

        let num_frames = online.num_frames_ready();
        let mut all_features = Vec::with_capacity(num_frames * self.num_mel_bins);

        for i in 0..num_frames {
            if let Some(frame) = online.get_frame(i) {
                all_features.extend_from_slice(frame);
            }
        }

        all_features
    }
}
