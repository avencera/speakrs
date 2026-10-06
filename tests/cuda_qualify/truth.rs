//! FP32 Library stage truth on the same inputs, independent of candidate math

use super::{Run, WINDOW, read_batch};
use crate::inference::cuda::{
    CudaError, CudaLstmAlgorithm, CudaMath, CudaSegmentation, ResNetEmbedding, SafetensorsFile,
    SegmentationOptions,
};

impl Run<'_> {
    pub(super) fn truth(&self) -> Result<[Vec<f32>; 2], CudaError> {
        let runtime = self.runtime;
        if self.target == "resnet" {
            let weights = SafetensorsFile::open(
                "/workspace/models-native/wespeaker-multimask-tail.safetensors",
            )?;
            let model = ResNetEmbedding::load(runtime, &weights, CudaMath::Fp32)?;
            let mut batch = model.batch(runtime, self.batch)?;
            let mut outputs = Vec::new();
            for file in self.files {
                let fbank = read_batch(file, "input/fbank", self.batch)?;
                let masks = read_batch(file, "input/masks", self.batch)?;
                batch.fbank_mut().copy_from_host(runtime.stream(), &fbank)?;
                batch.masks_mut().copy_from_host(runtime.stream(), &masks)?;
                batch.forward_with_taps(runtime, &mut |_, _| Ok(()))?;
                outputs.push(batch.download_output(runtime)?);
            }
            return Ok(outputs.try_into().expect("two input sets"));
        }
        let weights =
            SafetensorsFile::open("/workspace/models-native/segmentation-3.0.safetensors")?;
        let options = SegmentationOptions {
            math: CudaMath::Fp32,
            lstm_algo: CudaLstmAlgorithm::PersistStaticSmallH,
            cuda_graph: false,
        };
        let mut model = CudaSegmentation::new(runtime, &weights, options)?;
        let mut outputs = Vec::new();
        for file in self.files {
            let audio = read_batch(file, "input/input", self.batch)?;
            model
                .workspace(runtime, self.batch, WINDOW)?
                .upload_input(runtime, &audio)?;
            model.forward_eager(runtime, self.batch, WINDOW)?;
            outputs.push(
                model
                    .find_workspace(self.batch, WINDOW)
                    .expect("workspace")
                    .download_output(runtime)?,
            );
        }
        Ok(outputs.try_into().expect("two input sets"))
    }
}
