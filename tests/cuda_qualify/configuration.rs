//! Locked receipts of the exact configuration passed to a successful candidate plan

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::inference::cuda::CudaMath;
use crate::inference::cuda::candidate::{ConfigPin, ConvPin, FbankPin, LstmPin, SincPin};
use serde_json::{Value, json};

thread_local! {
    static PINS: RefCell<BTreeMap<(String, usize, String), ConfigPin>> = const { RefCell::new(BTreeMap::new()) };
}

/// A stable closed representation; variant spelling is not derived from Debug
pub(crate) fn pin_json(pin: ConfigPin) -> Value {
    match pin {
        ConfigPin::Conv(ConvPin::LegacyWaves(shape)) => {
            json!({"kind":"Conv", "selection":"LegacyWaves", "shape": match shape {
                crate::inference::cuda::candidate::ConvShape::C32 => "C32",
                crate::inference::cuda::candidate::ConvShape::C64 => "C64",
                crate::inference::cuda::candidate::ConvShape::C32Stride2 => "C32Stride2",
            }})
        }
        ConfigPin::Conv(ConvPin::Kernel(entry)) => {
            json!({"kind":"Conv", "selection":"Kernel", "entry": match entry {
                crate::inference::cuda::candidate::ConvKernel::C32 => "C32",
                crate::inference::cuda::candidate::ConvKernel::C64 => "C64",
                crate::inference::cuda::candidate::ConvKernel::C64Small => "C64Small",
                crate::inference::cuda::candidate::ConvKernel::C32Stride2 => "C32Stride2",
                crate::inference::cuda::candidate::ConvKernel::C32Stride2Small => "C32Stride2Small",
            }})
        }
        ConfigPin::Lstm(LstmPin::LegacyCooperative) => {
            json!({"kind":"Lstm", "selection":"LegacyCooperative"})
        }
        ConfigPin::Lstm(LstmPin::Projected(projection)) => {
            json!({"kind":"Lstm", "selection":"Projected", "projection": match projection {
                crate::inference::cuda::candidate::LstmProjection::Small => "Small",
                crate::inference::cuda::candidate::LstmProjection::Large => "Large",
                crate::inference::cuda::candidate::LstmProjection::Tensor => "Tensor",
            }})
        }
        ConfigPin::Sinc(SincPin::ConvAbsPool) => json!({"kind":"Sinc", "selection":"ConvAbsPool"}),
        ConfigPin::Fbank(FbankPin::FftMelAccurate) => {
            json!({"kind":"Fbank", "selection":"FftMelAccurate"})
        }
        ConfigPin::Segdense(pin) => {
            json!({"kind":"Segdense", "kernel": pin.config().kernel, "splits": pin.splits()})
        }
    }
}

/// Record after plan construction, rejecting two configurations for the same tuple
pub(crate) fn record(boundary: &str, batch: usize, math: CudaMath, pin: ConfigPin) {
    let mode = match math {
        CudaMath::Fp32 => "fp32",
        CudaMath::Tf32 => "tf32",
    };
    PINS.with(|pins| {
        let mut pins = pins.borrow_mut();
        let key = (boundary.to_owned(), batch, mode.to_owned());
        if let Some(known) = pins.get(&key) {
            assert_eq!(*known, pin, "one configuration per qualification tuple");
        }
        pins.insert(key, pin);
    });
}

/// All planned tuples, in deterministic order, without candidate-written receipts
pub(crate) fn planned() -> Value {
    PINS.with(|pins| {
        Value::Array(pins.borrow().iter().map(|((boundary, batch, math), pin)|
        json!({"tuple":[boundary,batch,math], "pin":pin_json(*pin)})
    ).collect())
    })
}

/// The complete boundary and batch domain, exported from the typed owner
pub(crate) fn boundary_domain() -> Value {
    use crate::inference::cuda::implementation::BoundaryId;
    Value::Array(
        BoundaryId::all()
            .map(|boundary| {
                let batches: Vec<_> = (1..=32)
                    .filter(|batch| boundary.batches().contains(*batch))
                    .collect();
                json!({"boundary":boundary.name(), "batches":batches})
            })
            .collect(),
    )
}

/// The pinned model bytes of the integrated native pipeline
pub(crate) fn model_identity() -> Value {
    let assets: Value = serde_json::from_str(include_str!("ASSETS.json")).expect("locked assets");
    json!({
        "embedding": assets["files"]["/workspace/models-native/wespeaker-multimask-tail.safetensors"],
        "segmentation": assets["files"]["/workspace/models-native/segmentation-3.0.safetensors"],
    })
}

/// A DER source inventory must come from locked live sources, not receipt claims
enum DerInventory {
    Missing { sources: [&'static str; 4] },
}

/// Export absent sources explicitly; qualification assets are not integrated DER inputs
pub(crate) fn der_inventory() -> Value {
    let inventory = DerInventory::Missing {
        sources: [
            "DER audio content manifest with file count",
            "DER reference annotation content inventory",
            "DER serialized pipeline configuration",
            "DER Library baseline control archive",
        ],
    };
    match inventory {
        DerInventory::Missing { sources } => json!({"kind": "Missing", "sources": sources}),
    }
}

/// Always-on Library-owned GPU bytes used by the full integrated plan
pub(crate) fn library_artifacts() -> Value {
    use crate::inference::cuda::kernels::ArtifactHash;
    json!({
        "fbank": {"tier":"sm75", "artifact":{"kind":"PtxJit", "sha256":ArtifactHash::of(include_bytes!("../../src/inference/cuda/ptx/fbank.sm75.ptx")).to_string()}},
        "embedding": {"tier":"sm75", "artifact":{"kind":"PtxJit", "sha256":ArtifactHash::of(include_bytes!("../../src/inference/cuda/ptx/embedding.sm75.ptx")).to_string()}},
        "segmentation": {"tier":"sm75", "artifact":{"kind":"PtxJit", "sha256":ArtifactHash::of(include_bytes!("../../src/inference/cuda/ptx/segmentation.sm75.ptx")).to_string()}},
    })
}

#[cfg(test)]
mod tests {
    use super::{pin_json, planned, record};
    use crate::inference::cuda::CudaMath;
    use crate::inference::cuda::candidate::{ConfigPin, ConvKernel, ConvPin, ConvShape};

    #[test]
    fn a_fixed_entry_is_not_the_legacy_selection_rule() {
        let legacy = ConfigPin::Conv(ConvPin::LegacyWaves(ConvShape::C64));
        let fixed = ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C64));
        assert_ne!(pin_json(legacy), pin_json(fixed));
        record("resnet.layer2.1.conv1", 32, CudaMath::Fp32, legacy);
        record("resnet.layer2.1.conv1", 32, CudaMath::Fp32, legacy);
        let rows = planned();
        assert_eq!(rows.as_array().unwrap().len(), 1);
        assert_eq!(rows[0]["pin"], pin_json(legacy));
        assert!(
            std::panic::catch_unwind(|| record("resnet.layer2.1.conv1", 32, CudaMath::Fp32, fixed))
                .is_err()
        );
        assert_eq!(planned(), rows);
    }
}
