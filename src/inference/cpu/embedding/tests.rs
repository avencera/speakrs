use super::{CpuResNet34, WeightsFile};
use crate::inference::cpu::fbank::CpuFbank;
use ndarray::{Array2, Array3, Axis, s};
use ndarray_npy::{NpzReader, read_npy};
use std::{fs::File, path::PathBuf, time::Instant};

fn load() -> CpuResNet34 {
    CpuResNet34::load(
        &WeightsFile::open(
            crate::test_support::model_fixture_dir().join("wespeaker-multimask-tail.safetensors"),
        )
        .expect("required pinned model fixture"),
    )
    .unwrap()
}

const EMBEDDING_MAX_ERROR: f32 = 5e-3;
const EMBEDDING_MIN_COSINE: f64 = 0.99999;

#[derive(Debug)]
struct EmbeddingAlignment {
    max_error: f32,
    minimum_cosine: f64,
    nan_count: usize,
}

/// Checks the independent same-input budget: exact NaN layout, absolute error and cosine
fn embedding_alignment(
    actual: &Array2<f32>,
    expected: &Array2<f32>,
) -> Result<EmbeddingAlignment, String> {
    if actual.dim() != expected.dim() {
        return Err(format!("shape {:?} != {:?}", actual.dim(), expected.dim()));
    }
    let mut alignment = EmbeddingAlignment {
        max_error: 0.0,
        minimum_cosine: 1.0,
        nan_count: 0,
    };
    for (row, (a, b)) in actual.rows().into_iter().zip(expected.rows()).enumerate() {
        for (a, b) in a.iter().zip(b) {
            if a.is_nan() != b.is_nan() {
                return Err(format!("row {row}: NaN layout differs"));
            }
            if a.is_nan() {
                alignment.nan_count += 1;
                continue;
            }
            if !a.is_finite() || !b.is_finite() {
                return Err(format!("row {row}: non-finite value"));
            }
            alignment.max_error = alignment.max_error.max((a - b).abs());
        }
        if a.iter().any(|v| v.is_nan()) {
            continue;
        }
        let dot = a
            .iter()
            .zip(b)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>();
        let an = a.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        let bn = b.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        if an == 0.0 && bn == 0.0 {
            continue;
        }
        // a zero row against a nonzero row yields NaN, which must fail this check
        let cosine = dot / (an * bn).sqrt();
        if cosine.is_nan() || cosine < EMBEDDING_MIN_COSINE {
            return Err(format!("row {row}: cosine={cosine}"));
        }
        alignment.minimum_cosine = alignment.minimum_cosine.min(cosine);
    }
    // cosine is scale-invariant, so magnitude errors are only caught here
    if alignment.max_error > EMBEDDING_MAX_ERROR {
        return Err(format!("max_error={}", alignment.max_error));
    }
    Ok(alignment)
}

fn compare(actual: &Array2<f32>, expected: &Array2<f32>, label: &str) {
    let alignment = embedding_alignment(actual, expected).unwrap_or_else(|error| {
        panic!("{label}: {error}");
    });
    println!(
        "embedding {label} min_cosine={:.10} max_error={:.8} nan_count={}",
        alignment.minimum_cosine, alignment.max_error, alignment.nan_count
    );
}

#[test]
fn embedding_alignment_rejects_scaled_rows_and_layout_changes() {
    let reference = Array2::from_elem((1, 256), 1.0_f32);
    let scaled = Array2::from_elem((1, 256), 2.0_f32);
    let error = embedding_alignment(&scaled, &reference).unwrap_err();
    assert!(error.starts_with("max_error="), "{error}");

    let mut nearby = reference.clone();
    nearby[[0, 7]] += 1e-3;
    let alignment = embedding_alignment(&nearby, &reference).unwrap();
    assert!((alignment.max_error - 1e-3).abs() < 1e-6);
    assert!(alignment.minimum_cosine >= EMBEDDING_MIN_COSINE);

    let mut inactive = reference.clone();
    inactive.fill(f32::NAN);
    assert!(embedding_alignment(&inactive, &reference).is_err());
    assert!(embedding_alignment(&inactive, &inactive).is_ok());
    assert!(embedding_alignment(&Array2::zeros((1, 256)), &reference).is_err());
}

#[test]
fn required_model_strided_short_filterbanks_mask_groups_and_scratch_reuse() {
    let model = load();
    let clone = model.clone();
    assert!(std::sync::Arc::ptr_eq(&model.0, &clone.0));
    let mut scratch = model.workspace();
    let mut fresh = clone.workspace();
    assert_ne!(scratch.fbank.as_ptr(), fresh.fbank.as_ptr());
    let audio = wav(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/test.wav"));
    let fbank_model = CpuFbank::new();
    let features = fbank_features(&fbank_model, &audio[..160000], &mut fbank_model.workspace());
    let storage = Array2::from_shape_fn((998, 160), |(time, bin)| features[[time, bin / 2]]);
    let fbank = storage.slice(s![..,..;2]);
    let a = vec![1.0; 589];
    let b: Vec<f32> = (0..589).map(|i| if i < 294 { 0.7 } else { 0.0 }).collect();
    let c = vec![0.0; 589];
    let masks = [a.as_slice(), b.as_slice(), c.as_slice()];
    let rows = model.forward(fbank, &masks, &mut scratch).unwrap();
    assert!(rows.iter().all(|v| v.is_finite()));
    assert!(
        rows.row(0)
            .iter()
            .zip(rows.row(1))
            .any(|(a, b)| (a - b).abs() > 0.01)
    );
    for (index, mask) in masks.iter().enumerate() {
        let single = clone.forward(fbank, &[*mask], &mut fresh).unwrap();
        assert_eq!(single.row(0), rows.row(index));
    }
    let short = fbank.slice(s![..17, ..]);
    let short_rows = model.forward(short, &masks, &mut scratch).unwrap();
    let mut full = Array2::zeros((998, 80));
    full.slice_mut(s![..17, ..]).assign(&short);
    assert_eq!(
        model.forward(full.view(), &masks, &mut scratch).unwrap(),
        short_rows
    );
    assert_eq!(
        model.forward(fbank, &[], &mut scratch).unwrap().dim(),
        (0, 256)
    );
    assert!(
        model
            .forward(Array2::zeros((999, 80)).view(), &masks, &mut scratch)
            .is_err()
    );
    assert!(
        model
            .forward(Array2::zeros((10, 79)).view(), &masks, &mut scratch)
            .is_err()
    );
    let zeros = Array2::zeros((998, 80));
    model.forward(zeros.view(), &masks, &mut scratch).unwrap();
    assert_eq!(model.forward(fbank, &masks, &mut scratch).unwrap(), rows);
}

#[test]
#[ignore = "requires explicit SPEAKRS_CPU_BASELINE_DIR pinned acceptance captures"]
fn all_pinned_raw_and_saved_rust_windows() {
    let baseline = PathBuf::from(
        std::env::var_os("SPEAKRS_CPU_BASELINE_DIR").expect("required baseline directory"),
    );
    let model = load();
    let mut scratch = model.workspace();
    for case in [
        "test_first_b1",
        "test_last_partial_b1",
        "test_short_partial_b1",
    ] {
        let reference = WeightsFile::open(baseline.join(format!(
            "parity/wespeaker-multimask-tail/{case}.safetensors"
        )))
        .unwrap();
        let input = Array2::from_shape_vec(
            (998, 80),
            reference.read_f32("input/fbank", &[1, 998, 80]).unwrap(),
        )
        .unwrap();
        let masks = reference.read_f32("input/masks", &[3, 589]).unwrap();
        let masks: Vec<&[f32]> = masks
            .as_chunks::<589>()
            .0
            .iter()
            .map(|row| row.as_slice())
            .collect();
        let expected = Array2::from_shape_vec(
            (3, 256),
            reference.read_f32("tensor/output", &[3, 256]).unwrap(),
        )
        .unwrap();
        let start = Instant::now();
        let actual = model.forward(input.view(), &masks, &mut scratch).unwrap();
        println!(
            "embedding raw={case} ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        compare(&actual, &expected, case);
    }
    let manifest: serde_json::Value = serde_json::from_reader(
        File::open(baseline.join("optimized-tensors/report.json")).unwrap(),
    )
    .unwrap();
    let fbank = CpuFbank::new();
    let mut fbank_scratch = fbank.workspace();
    for record in manifest["reports"].as_array().unwrap() {
        let path = std::path::Path::new(record["input"]["path"].as_str().unwrap());
        let name = path.file_stem().unwrap().to_str().unwrap();
        let audio = wav(path);
        let mut archive = NpzReader::new(
            File::open(baseline.join(format!("optimized-tensors/{name}.npz"))).unwrap(),
        )
        .unwrap();
        let masks: Array3<f32> = archive.by_name("masks").unwrap();
        let expected: Array3<f32> = archive.by_name("embeddings").unwrap();
        let rust_embeddings: Array3<f32> =
            read_npy(baseline.join(format!("measurements/{name}/library/embeddings.npy"))).unwrap();
        let decoded: Array3<f32> =
            read_npy(baseline.join(format!("measurements/{name}/library/decoded.npy"))).unwrap();
        for (index, row) in record["rows"].as_array().unwrap().iter().enumerate() {
            let offset = row["offset_samples"].as_u64().unwrap() as usize;
            let end = (offset + 160000).min(audio.len());
            let features = fbank_features(&fbank, &audio[offset..end], &mut fbank_scratch);
            let mask = masks.index_axis(Axis(0), index);
            let mask_refs: Vec<&[f32]> = mask
                .rows()
                .into_iter()
                .map(|v| v.to_slice().unwrap())
                .collect();
            let start = Instant::now();
            let actual = model
                .forward(features.view(), &mask_refs, &mut scratch)
                .unwrap();
            println!(
                "embedding window={name}/{index} ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            compare(
                &actual,
                &expected.index_axis(Axis(0), index).to_owned(),
                &format!("optimized/{name}/{index}"),
            );
            let segmentation = decoded.index_axis(Axis(0), index);
            let clean = crate::pipeline::clean_masks(&segmentation);
            let selected: Vec<Vec<f32>> = (0..3)
                .map(|speaker| {
                    crate::pipeline::select_speaker_weights(
                        &segmentation,
                        &clean,
                        speaker,
                        end - offset,
                        400,
                    )
                    .unwrap_or_else(|| vec![0.0; 589])
                })
                .collect();
            let selected_refs: Vec<&[f32]> = selected.iter().map(Vec::as_slice).collect();
            let rust_fbank: Array2<f32> = read_npy(
                baseline.join(format!("measurements/{name}/library/fbank-{index:02}.npy")),
            )
            .unwrap();
            let mut actual = model
                .forward(rust_fbank.view(), &selected_refs, &mut scratch)
                .unwrap();
            // the pipeline, not the native tail, marks inactive speaker rows as NaN
            for speaker in 0..3 {
                if segmentation.column(speaker).sum() < 10.0 {
                    actual.row_mut(speaker).fill(f32::NAN);
                }
            }
            compare(
                &actual,
                &rust_embeddings.index_axis(Axis(0), index).to_owned(),
                &format!("rust/{name}/{index}"),
            );
        }
    }
}
fn wav(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000);
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.bits_per_sample, 16);
    reader
        .samples::<i16>()
        .map(|value| f32::from(value.unwrap()) / 32768.0)
        .collect()
}

#[test]
fn inventory_rejects_missing_extra_and_same_count_wrong_names_before_tensor_reads() {
    let inventory = super::trunk::Trunk::inventory();
    for mode in 0..3 {
        let mut names = inventory.clone();
        if mode != 1 {
            names.pop();
        }
        if mode != 0 {
            names.push("resnet.seg_2.weight".to_owned());
        }
        let tensors: serde_json::Map<String, serde_json::Value> = names
            .into_iter()
            .map(|name| {
                (
                    name,
                    serde_json::json!({"dtype":"F32","shape":[0],"data_offsets":[0,0]}),
                )
            })
            .collect();
        let header = serde_json::to_vec(&tensors).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        let path = std::env::temp_dir().join(format!(
            "speakrs-cpu-inventory-{}-{mode}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let file = WeightsFile::open(&path).unwrap();
        let error = match CpuResNet34::load(&file) {
            Ok(_) => panic!("unsupported inventory was accepted"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            crate::inference::CpuError::Shape(crate::inference::TensorShapeError::ShapeMismatch {
                context: "CPU embedding tensor inventory (missing or extra names)",
                ..
            })
        ));
        std::fs::remove_file(path).unwrap();
    }
}

fn fbank_features(
    model: &CpuFbank,
    audio: &[f32],
    scratch: &mut crate::inference::cpu::fbank::FbankWorkspace,
) -> Array2<f32> {
    let mut features = Array2::zeros((998, 80));
    model
        .compute_into(audio, features.view_mut(), scratch)
        .unwrap();
    features
}
