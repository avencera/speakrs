//! File validity is independent of GPU timing and driver availability

use std::sync::atomic::{AtomicUsize, Ordering};

use super::{ChoiceKey, DeviceKey, Entry, FileError, MathKey, TuneFile, read};
use crate::inference::cuda::device::test_support::Builder;
use crate::inference::cuda::implementation::BoundaryId;
use crate::inference::cuda::tuning::driver_version::DriverVersion;
use crate::inference::cuda::tuning::{Catalogue, Tuple};
use crate::inference::cuda::{ComputeCapability, CudaMath, PtxTier};

fn key() -> DeviceKey {
    DeviceKey {
        device_name: "NVIDIA GeForce RTX 4060 Ti".into(),
        capability: [8, 9],
        sm_count: 34,
        driver_version: DriverVersion::Nvml("595.91.07".into()),
        libraries: crate::inference::cuda::tuning::LibraryVersions::DriverOnly,
        speakrs_version: "0.6.0".into(),
        artifact_version: "exact-artifact-and-catalogue-digest".into(),
        accuracy_policy: crate::inference::cuda::tuning::accuracy::Policy::IDENTITY.into(),
    }
}

fn fixture() -> (Catalogue, TuneFile) {
    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name(&key().device_name)
        .build();
    let catalogue = Catalogue::new(&device, PtxTier::Sm80).unwrap();
    let tuple = Tuple::new(
        BoundaryId::named("resnet.layer1.0.conv1"),
        32,
        CudaMath::Tf32,
    )
    .unwrap();
    let choice = catalogue.choices(tuple).first().unwrap().key();
    let file = TuneFile::new(
        key(),
        vec![Entry {
            boundary: tuple.boundary.name().into(),
            batch: tuple.batch,
            math: MathKey::Tf32,
            choice,
            median_ms: 0.2,
        }],
    );
    (catalogue, file)
}

#[test]
fn exact_key_and_execution_pin_are_required() {
    let (catalogue, file) = fixture();
    let tuple = Tuple::new(
        BoundaryId::named("resnet.layer1.0.conv1"),
        32,
        CudaMath::Tf32,
    )
    .unwrap();
    let validated = file.clone().validate(&key(), &catalogue).unwrap();
    assert_eq!(
        validated.choice(tuple).unwrap().key(),
        file.entries[0].choice
    );
    assert!(
        validated
            .choice(Tuple::new(tuple.boundary, 1, CudaMath::Tf32).unwrap())
            .is_none()
    );

    for component in 0..8 {
        let mut wrong = key();
        match component {
            0 => wrong.device_name.push_str(" other"),
            1 => wrong.capability = [8, 6],
            2 => wrong.sm_count = 35,
            3 => wrong.driver_version = DriverVersion::Nvml("595.91.08".into()),
            4 => wrong.speakrs_version = "0.7.0".into(),
            5 => wrong.artifact_version.push_str(" changed"),
            6 => wrong.accuracy_policy.push_str(" changed"),
            7 => wrong.driver_version = DriverVersion::Sysfs("595.91.07".into()),
            _ => unreachable!(),
        }
        assert!(matches!(
            file.clone().validate(&wrong, &catalogue),
            Err(FileError::KeyMismatch)
        ));
    }
}

#[test]
fn malformed_or_unapproved_rows_reject_the_complete_file() {
    let (catalogue, file) = fixture();
    for change in 0..8 {
        let mut invalid = file.clone();
        match change {
            0 => invalid.entries.push(invalid.entries[0].clone()),
            1 => invalid.entries[0].boundary = "unknown".into(),
            2 => invalid.entries[0].batch = 7,
            3 => invalid.entries[0].median_ms = 0.0,
            4 => invalid.entries[0].median_ms = f64::NAN,
            5 => {
                invalid.entries[0].choice = ChoiceKey::Kernel {
                    module: "foreign artifact".into(),
                    config_pin: "Conv(Kernel(C32Tensor))".into(),
                }
            }
            6 => invalid.format_version = 99,
            7 => {
                let ChoiceKey::Kernel { config_pin, .. } = &mut invalid.entries[0].choice else {
                    panic!("kernel fixture")
                };
                *config_pin = "Conv(Kernel(C128Tensor))".into();
            }
            _ => unreachable!(),
        }
        assert!(invalid.validate(&key(), &catalogue).is_err());
    }
    let mut json = serde_json::to_value(&file).unwrap();
    json["key"]["extra"] = true.into();
    assert!(serde_json::from_value::<TuneFile>(json).is_err());
}

#[test]
fn json_write_read_and_missing_file_have_concrete_results() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "speakrs-tune-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let path = directory.join("device.json");
    let (catalogue, file) = fixture();
    assert!(read(&path).unwrap().is_none());
    file.write(&path).unwrap();
    let loaded = read(&path).unwrap().unwrap();
    assert_eq!(loaded.key, key());
    assert_eq!(loaded.entries[0].choice, file.entries[0].choice);
    assert!(loaded.validate(&key(), &catalogue).is_ok());
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
#[cfg(not(feature = "_cuda-libraries"))]
fn driver_only_file_cannot_grant_library_permission() {
    let (catalogue, mut file) = fixture();
    file.entries[0].choice = ChoiceKey::Library;
    assert!(file.validate(&key(), &catalogue).is_err());
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn an_implemented_unapproved_algorithm_rejects_a_tune_file() {
    let (device, tuple, module, pin) =
        crate::inference::cuda::tuning::tests::unapproved_configuration();
    let catalogue = Catalogue::new(&device, PtxTier::Sm80).unwrap();
    let directory = std::env::temp_dir().join(format!("speakrs-unapproved-{}", std::process::id()));
    let path = directory.join("device.json");
    let file = TuneFile::new(
        key(),
        vec![Entry {
            boundary: tuple.boundary.name().into(),
            batch: tuple.batch,
            math: tuple.math,
            choice: ChoiceKey::Kernel {
                module: format!("{module:?}"),
                config_pin: format!("{pin:?}"),
            },
            median_ms: 0.01,
        }],
    );
    file.write(&path).unwrap();
    let loaded = read(&path).unwrap().unwrap();
    assert!(
        matches!(loaded.validate(&key(), &catalogue), Err(FileError::Invalid(reason))
        if reason.contains("unapproved configuration"))
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn a_changed_accuracy_policy_rejects_the_old_file() {
    let (catalogue, file) = fixture();
    let mut next_policy = key();
    next_policy.accuracy_policy = "end-to-end-algorithms-v3".into();
    let old_file: TuneFile = serde_json::from_slice(&serde_json::to_vec(&file).unwrap()).unwrap();
    assert!(matches!(
        old_file.validate(&next_policy, &catalogue),
        Err(FileError::KeyMismatch)
    ));
}

#[test]
fn a_colliding_temp_file_is_preserved_and_the_complete_file_is_written() {
    let directory = std::env::temp_dir().join(format!("speakrs-collision-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("device.json");
    let collision = directory.join(format!(".device.json.{:032x}.tmp", 7));
    std::fs::write(&collision, b"another writer's partial file").unwrap();
    let (catalogue, file) = fixture();
    let mut nonces = [7, 8].into_iter();
    file.write_with_nonce(&path, || Ok(nonces.next().unwrap()))
        .unwrap();
    let loaded = read(&path).unwrap().unwrap();
    assert_eq!(loaded.entries[0].choice, file.entries[0].choice);
    assert!(loaded.validate(&key(), &catalogue).is_ok());
    assert_eq!(
        std::fs::read(&collision).unwrap(),
        b"another writer's partial file"
    );
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 2);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn the_old_string_key_is_rejected_before_key_decoding() {
    let directory = std::env::temp_dir().join(format!("speakrs-old-tune-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("device.json");
    let (_, file) = fixture();
    let mut old = serde_json::to_value(file).unwrap();
    old["format_version"] = 1.into();
    old["key"]["driver_version"] = "595.91.07".into();
    std::fs::write(&path, serde_json::to_vec(&old).unwrap()).unwrap();
    assert!(matches!(
        read(&path),
        Err(FileError::Invalid(reason)) if reason == "unsupported tune-file format"
    ));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn numerical_library_versions_are_required_even_for_a_kernel_winner() {
    use crate::inference::cuda::tuning::LibraryVersions;
    let (catalogue, mut file) = fixture();
    file.key.libraries = LibraryVersions::Hybrid {
        cudnn: 91000,
        cublas: 120604,
    };
    assert!(file.clone().validate(&file.key, &catalogue).is_ok());
    for libraries in [
        LibraryVersions::Hybrid {
            cudnn: 91001,
            cublas: 120604,
        },
        LibraryVersions::Hybrid {
            cudnn: 91000,
            cublas: 120605,
        },
        LibraryVersions::DriverOnly,
    ] {
        let mut expected = file.key.clone();
        expected.libraries = libraries;
        assert!(matches!(
            file.clone().validate(&expected, &catalogue),
            Err(FileError::KeyMismatch)
        ));
    }
    let (catalogue, driver_file) = fixture();
    assert!(
        driver_file
            .clone()
            .validate(&driver_file.key, &catalogue)
            .is_ok()
    );
}
