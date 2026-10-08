use super::{CpuPyanNet, WeightsFile};
use crate::powerset::PowersetMapping;
use ndarray::{Array2, Array3, Axis};
use ndarray_npy::{NpzReader, read_npy};
use std::{fs::File, path::PathBuf, time::Instant};

fn load() -> CpuPyanNet {
    CpuPyanNet::load(
        &WeightsFile::open(
            crate::test_support::model_fixture_dir().join("segmentation-3.0.safetensors"),
        )
        .expect("required pinned model fixture"),
    )
    .unwrap()
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

fn numerical_alignment(actual: &[f32], reference: &[f32]) -> bool {
    if actual.len() != reference.len() || actual.is_empty() {
        return false;
    }
    if actual
        .iter()
        .chain(reference)
        .any(|value| !value.is_finite())
    {
        return false;
    }
    let delta = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    if delta > 2e-3 {
        return false;
    }
    let winner = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0
    };
    let native = winner(actual);
    let expected = winner(reference);
    // this bound explains unstable argmax changes, not model quality; the fixed
    // error budget and separate same-input/end-to-end checks provide that proof
    reference[expected] - reference[native] <= 2.0 * delta
        && actual[native] - actual[expected] <= 2.0 * delta
}

#[test]
fn numerical_alignment_keeps_error_budget_at_a_captured_decision_boundary() {
    let reference = [
        -1.2150804, -2.7716389, -2.3718948, -1.2150836, -3.019075, -2.6033964, -2.0565271,
    ];
    let native = [
        -1.2150841, -2.771638, -2.3718977, -1.2150831, -3.01907, -2.6033895, -2.0565238,
    ];
    assert!(numerical_alignment(&native, &reference));
    let mut excessive = native;
    excessive[3] += 0.01;
    assert!(!numerical_alignment(&excessive, &reference));
    assert!(!numerical_alignment(&[f32::NAN], &[0.0]));
    assert!(!numerical_alignment(&[], &[]));
}

fn compare(actual: &Array2<f32>, expected: &Array2<f32>, label: &str) -> usize {
    assert_eq!(actual.dim(), expected.dim());
    assert!(actual.iter().all(|value| value.is_finite()));
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let mapping = PowersetMapping::new(3, 2);
    let decoded = mapping.hard_decode(actual).unwrap();
    let reference = mapping.hard_decode(expected).unwrap();
    let differences = decoded
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a != b)
        .count();
    println!("segmentation {label} max_error={error:.8} decoded_differences={differences}");
    assert!(error <= 2e-3, "{label}: error={error}");
    for (row, reference) in actual.rows().into_iter().zip(expected.rows()) {
        assert!(
            numerical_alignment(row.as_slice().unwrap(), reference.as_slice().unwrap()),
            "{label}: class change exceeds numerical budget"
        );
    }
    if differences > 0 {
        for time in 0..actual.nrows() {
            if decoded.row(time) != reference.row(time) {
                println!(
                    "decode boundary {label} frame={time} actual={:?} expected={:?}",
                    actual.row(time).to_vec(),
                    expected.row(time).to_vec()
                );
            }
        }
    }
    differences
}

#[test]
fn required_model_reuse_short_full_truncation_and_weight_sharing() {
    let model = load();
    let clone = model.clone();
    assert!(std::sync::Arc::ptr_eq(&model.0, &clone.0));
    let mut scratch = model.workspace();
    let mut fresh = clone.workspace();
    assert_ne!(scratch.waveform.as_ptr(), fresh.waveform.as_ptr());
    let audio = wav(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/test_short.wav"));
    let expected = model.forward(&audio, &mut scratch).unwrap();
    assert!(expected.iter().all(|value| value.is_finite()));
    let mut padded = audio.clone();
    padded.resize(160000, 0.0);
    assert_eq!(model.forward(&padded, &mut fresh).unwrap(), expected);
    padded.extend([0.2; 7]);
    assert_eq!(model.forward(&padded, &mut scratch).unwrap(), expected);
    let changed = model.forward(&[], &mut scratch).unwrap();
    assert!(
        changed
            .iter()
            .zip(&expected)
            .any(|(a, b)| (a - b).abs() > 0.1)
    );
    assert_eq!(model.forward(&audio, &mut scratch).unwrap(), expected);
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
        let reference =
            WeightsFile::open(baseline.join(format!("parity/segmentation-3.0/{case}.safetensors")))
                .unwrap();
        let input = reference.read_f32("input/input", &[1, 1, 160000]).unwrap();
        let expected = Array2::from_shape_vec(
            (589, 7),
            reference.read_f32("tensor/output", &[1, 589, 7]).unwrap(),
        )
        .unwrap();
        let start = Instant::now();
        let actual = model.forward(&input, &mut scratch).unwrap();
        println!(
            "segmentation raw={case} ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        for stage in 0..3 {
            let suffix = if stage == 0 {
                String::new()
            } else {
                format!("_{stage}")
            };
            let name = format!("tensor//sincnet/LeakyRelu{suffix}_output_0");
            let dims = scratch.pooled[stage].dim();
            let expected_stage = reference.read_f32(&name, &[1, dims.0, dims.1]).unwrap();
            let error = scratch.pooled[stage]
                .iter()
                .zip(expected_stage)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            println!("segmentation stage={stage} raw={case} max_error={error:.8}");
        }
        assert_eq!(compare(&actual, &expected, case), 0, "raw decode {case}");
    }
    let manifest: serde_json::Value = serde_json::from_reader(
        File::open(baseline.join("optimized-tensors/report.json")).unwrap(),
    )
    .unwrap();
    let mut total_decode_differences = 0;
    for record in manifest["reports"].as_array().unwrap() {
        let path = std::path::Path::new(record["input"]["path"].as_str().unwrap());
        let name = path.file_stem().unwrap().to_str().unwrap();
        let audio = wav(path);
        let mut archive = NpzReader::new(
            File::open(baseline.join(format!("optimized-tensors/{name}.npz"))).unwrap(),
        )
        .unwrap();
        let logits: Array3<f32> = archive.by_name("logits").unwrap();
        for (index, row) in record["rows"].as_array().unwrap().iter().enumerate() {
            let offset = row["offset_samples"].as_u64().unwrap() as usize;
            let end = (offset + 160000).min(audio.len());
            let start = Instant::now();
            let actual = model.forward(&audio[offset..end], &mut scratch).unwrap();
            println!(
                "segmentation window={name}/{index} ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            let optimized_differences = compare(
                &actual,
                &logits.index_axis(Axis(0), index).to_owned(),
                &format!("optimized/{name}/{index}"),
            );
            total_decode_differences += optimized_differences;
            let expected: Array2<f32> = read_npy(
                baseline.join(format!("measurements/{name}/library/logits-{index:02}.npy")),
            )
            .unwrap();
            total_decode_differences +=
                compare(&actual, &expected, &format!("rust/{name}/{index}"));
        }
    }
    println!(
        "numerically bounded decoded activity differences={total_decode_differences}; end-to-end acceptance is checked separately"
    );
}
