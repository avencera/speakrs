use std::fs;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use sha2::{Digest, Sha256};
use speakrs::imported_segmentation::{
    ArgmaxTie, AudioIdentity, BundleIdentity, ChannelSelection, ChunkGeometry, ComponentIdentity,
    CountPolicy, CountRounding, DecoderPolicy, Downmix, EmbeddingFrontend, EmbeddingPolicy,
    EmbeddingPooling, EmbeddingPrecision, FORMAT_VERSION, FilterBoundary, FilterPolicy, FrameGrid,
    IdentityReference, MaskInterpolation, OutputExtent, OutputExtentPolicy, Precision,
    RationalSample, ReconstructionBoundary, ReconstructionPolicy, ResamplingIdentity, SCHEMA_ID,
    ScoreRepresentation, ScoreStage, SegmentationBundle, SegmentationGeometry, SegmentationHead,
    SegmentationManifest, SegmentationPolicy, Sha256Digest, TailPolicy, TensorInventory,
    TensorShard, WindowPlanning, WindowPlanningKind, canonical_manifest_digest,
    load_imported_segmentation_bundle,
};

const WINDOW_SAMPLES: u64 = 128_000;
const FRAME_COUNT: u32 = 399;
const FRAME_STEP: RationalSample = RationalSample {
    numerator: 320,
    denominator: 1,
};
const FRAME_SUPPORT: RationalSample = RationalSample {
    numerator: 400,
    denominator: 1,
};

fn digest(fill: char) -> Sha256Digest {
    Sha256Digest::new(fill.to_string().repeat(64)).unwrap()
}

fn identity_reference(fill: char, name: &str) -> IdentityReference {
    IdentityReference {
        id: name.to_owned(),
        revision: "v1".to_owned(),
        sha256: digest(fill),
    }
}

fn class_mapping(local_slots: u32, max_overlap: u32) -> Vec<Vec<u32>> {
    fn extend(start: u32, remaining: usize, current: &mut Vec<u32>, output: &mut Vec<Vec<u32>>) {
        if remaining == 0 {
            output.push(current.clone());
            return;
        }
        for slot in start..=6 {
            current.push(slot);
            extend(slot + 1, remaining - 1, current, output);
            current.pop();
        }
    }

    let mut mapping = vec![Vec::new()];
    for size in 1..=max_overlap {
        let mut combinations = Vec::new();
        extend(0, size as usize, &mut Vec::new(), &mut combinations);
        mapping.extend(
            combinations
                .into_iter()
                .filter(|subset| subset.iter().all(|slot| *slot < local_slots)),
        );
    }
    mapping
}

fn npy(shape: [u64; 3], values: &[f32]) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        shape[0], shape[1], shape[2]
    )
    .into_bytes();
    while (10 + header.len() + 1) % 16 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut output = Vec::with_capacity(10 + header.len() + values.len() * 4);
    output.extend_from_slice(b"\x93NUMPY");
    output.extend_from_slice(&[1, 0]);
    output.extend_from_slice(&(header.len() as u16).to_le_bytes());
    output.extend_from_slice(&header);
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}

fn aggregate_extent_end(chunks: &[ChunkGeometry]) -> u64 {
    let Some(last) = chunks.last() else {
        return 0;
    };
    let endpoint = last.start_samples + WINDOW_SAMPLES;
    endpoint / FRAME_STEP.numerator as u64 * FRAME_STEP.numerator as u64
        + FRAME_SUPPORT.numerator as u64
}

fn file_digest(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    Sha256Digest::new(
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
    )
    .unwrap()
}

fn make_manifest(
    local_slots: u32,
    max_overlap: u32,
    sample_count: u64,
    path: &str,
    representation: ScoreRepresentation,
    values: &[f32],
) -> (SegmentationManifest, Vec<u8>) {
    let class_to_slot_subsets = class_mapping(local_slots, max_overlap);
    let classes = class_to_slot_subsets.len() as u64;
    let chunks = if sample_count == 0 {
        Vec::new()
    } else {
        vec![ChunkGeometry {
            index: 0,
            padding_samples: WINDOW_SAMPLES - sample_count.min(WINDOW_SAMPLES),
            start_samples: 0,
            valid_samples: sample_count.min(WINDOW_SAMPLES),
        }]
    };
    let shape = [chunks.len() as u64, FRAME_COUNT as u64, classes];
    let has_chunks = !chunks.is_empty();
    let shard_bytes = if chunks.is_empty() {
        Vec::new()
    } else {
        npy(shape, values)
    };
    let shard = TensorShard {
        bytes: shard_bytes.len() as u64,
        chunk_end: chunks.len() as u64,
        chunk_start: 0,
        path: path.to_owned(),
        sha256: file_digest(&shard_bytes),
        shape,
    };
    let output_extent = OutputExtent {
        end_samples: aggregate_extent_end(&chunks),
        start_samples: 0,
    };
    let geometry = SegmentationGeometry {
        aggregate_grid: FrameGrid {
            frame_count: FRAME_COUNT,
            origin: RationalSample {
                numerator: 0,
                denominator: 1,
            },
            step: FRAME_STEP,
            support: FRAME_SUPPORT,
        },
        chunks,
        frame_grid: FrameGrid {
            frame_count: FRAME_COUNT,
            origin: RationalSample {
                numerator: -241,
                denominator: 2,
            },
            step: FRAME_STEP,
            support: FRAME_SUPPORT,
        },
        output_extent,
        output_extent_policy: OutputExtentPolicy::AggregateGrid,
        window_planning: WindowPlanning {
            kind: WindowPlanningKind::RegularFixedStepV1,
            step_samples: 12_800,
            tail_policy: TailPolicy::PadFinal,
        },
        window_samples: WINDOW_SAMPLES,
    };
    let mut manifest = SegmentationManifest {
        audio: AudioIdentity {
            channel_selection: ChannelSelection::First,
            channels: 1,
            downmix: Downmix::None,
            original_recording_id: "recording-1".to_owned(),
            parent_recording_id: None,
            resampling: ResamplingIdentity {
                algorithm: "identity".to_owned(),
                id: "audio.identity".to_owned(),
                revision: "v1".to_owned(),
                sha256: digest('1'),
            },
            sample_count,
            sample_rate: 16_000,
            waveform_sha256: digest('2'),
        },
        format_version: FORMAT_VERSION,
        geometry,
        head: SegmentationHead {
            argmax_tie: ArgmaxTie::First,
            class_to_slot_subsets,
            local_slots,
            max_overlap,
            score_representation: representation,
        },
        identity: BundleIdentity {
            bundle_id: digest('3'),
            components: vec![ComponentIdentity {
                id: "wavlm.component".to_owned(),
                name: "wavlm".to_owned(),
                revision: "v1".to_owned(),
                sha256: digest('4'),
            }],
            config: identity_reference('5', "config"),
            environment: identity_reference('6', "environment"),
            manifest_digest: digest('0'),
            model: identity_reference('7', "model"),
            precision: Precision::Float32,
            producer: identity_reference('8', "producer"),
            source: identity_reference('9', "source"),
        },
        policy: SegmentationPolicy {
            count: CountPolicy {
                id: "count.reference".to_owned(),
                revision: "v1".to_owned(),
                rounding: CountRounding::NearestEven,
            },
            decoder: DecoderPolicy {
                argmax_tie: ArgmaxTie::First,
                id: "decoder.reference".to_owned(),
                representation,
                revision: "v1".to_owned(),
            },
            embedding: EmbeddingPolicy {
                embedding_model: identity_reference('a', "wespeaker-voxceleb-resnet34-fixed"),
                embedding_sidecar_sha256: digest('b'),
                frontend: EmbeddingFrontend::WeSpeakerFbankV1,
                id: "embedding.reference".to_owned(),
                mask_interpolation: MaskInterpolation::Nearest,
                min_num_samples: 1,
                plda: identity_reference('c', "plda"),
                pooling: EmbeddingPooling::MaskedStatsPoolV1,
                precision: EmbeddingPrecision::Float32,
                revision: "v1".to_owned(),
                target_frames: 100,
            },
            filter: FilterPolicy {
                boundary: FilterBoundary::Reflect,
                enabled: true,
                id: "filter.reference".to_owned(),
                revision: "v1".to_owned(),
                width: 11,
            },
            reconstruction: ReconstructionPolicy {
                boundary: ReconstructionBoundary::Explicit,
                extent_policy: OutputExtentPolicy::AggregateGrid,
                id: "reconstruction.reference".to_owned(),
                revision: "v1".to_owned(),
            },
        },
        schema_id: SCHEMA_ID.to_owned(),
        tensors: TensorInventory {
            score_stage: ScoreStage::JointScoresPreDecode,
            shards: if has_chunks { vec![shard] } else { Vec::new() },
        },
    };
    manifest.identity.bundle_id = manifest.canonical_bundle_id().unwrap();
    manifest.identity.manifest_digest = manifest.canonical_digest().unwrap();
    (manifest, shard_bytes)
}

struct BundleTempDir {
    inner: tempfile::TempDir,
}

impl BundleTempDir {
    fn path(&self) -> &Path {
        self.inner.path()
    }
}

impl Drop for BundleTempDir {
    fn drop(&mut self) {
        for path in [self.path().join("scores"), self.path().to_path_buf()] {
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                continue;
            }
            let mut permissions = metadata.permissions();
            #[cfg(unix)]
            permissions.set_mode(permissions.mode() | 0o700);
            #[cfg(not(unix))]
            permissions.set_readonly(false);
            let _ = fs::set_permissions(path, permissions);
        }
    }
}

fn bundle_tempdir() -> BundleTempDir {
    BundleTempDir {
        inner: tempfile::tempdir().unwrap(),
    }
}

fn write_bundle(root: &Path, manifest: &SegmentationManifest, shard: &[u8]) {
    fs::create_dir_all(root.join("scores")).unwrap();
    let shard_path = root.join(&manifest.tensors.shards[0].path);
    let manifest_path = root.join("manifest.json");
    let marker_path = root.join(".complete");
    fs::write(&shard_path, shard).unwrap();
    fs::write(&manifest_path, manifest.to_json().unwrap()).unwrap();
    fs::write(
        &marker_path,
        manifest.identity.manifest_digest.as_str().as_bytes(),
    )
    .unwrap();
    set_read_only(&shard_path, true);
    set_read_only(&manifest_path, true);
    set_read_only(&marker_path, true);
    set_read_only(&root.join("scores"), true);
    set_read_only(root, true);
}

fn set_read_only(path: &Path, read_only: bool) {
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_readonly(read_only);
    fs::set_permissions(path, permissions).unwrap();
}

fn rebind_shard(manifest: &SegmentationManifest, shard: &[u8]) -> SegmentationManifest {
    let mut rebound = manifest.clone();
    let declaration = &mut rebound.tensors.shards[0];
    declaration.bytes = shard.len() as u64;
    declaration.sha256 = file_digest(shard);
    rebind_identity(&mut rebound);
    rebound
}

fn rebind_identity(manifest: &mut SegmentationManifest) {
    manifest.identity.bundle_id = manifest.canonical_bundle_id().unwrap();
    manifest.identity.manifest_digest = manifest.canonical_digest().unwrap();
}

fn make_three_chunk_bundle() -> (SegmentationManifest, Vec<u8>) {
    let values = vec![0.0_f32; 3 * FRAME_COUNT as usize * 11];
    let one_chunk_values = &values[..FRAME_COUNT as usize * 11];
    let (mut manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        one_chunk_values,
    );
    manifest.audio.sample_count = 150_000;
    manifest.geometry.output_extent.end_samples = 154_000;
    manifest.geometry.chunks = vec![
        ChunkGeometry {
            index: 0,
            padding_samples: 0,
            start_samples: 0,
            valid_samples: 128_000,
        },
        ChunkGeometry {
            index: 1,
            padding_samples: 0,
            start_samples: 12_800,
            valid_samples: 128_000,
        },
        ChunkGeometry {
            index: 2,
            padding_samples: 3_600,
            start_samples: 25_600,
            valid_samples: 124_400,
        },
    ];
    let shape = [3, FRAME_COUNT as u64, 11];
    let shard = npy(shape, &values);
    let declaration = &mut manifest.tensors.shards[0];
    declaration.bytes = shard.len() as u64;
    declaration.chunk_end = 3;
    declaration.shape = shape;
    declaration.sha256 = file_digest(&shard);
    rebind_identity(&mut manifest);
    (manifest, shard)
}

#[test]
fn fixture_bundle_loads_without_a_model() {
    let classes = 11;
    let values = vec![0.0_f32; FRAME_COUNT as usize * classes];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &manifest, &shard);

    let bundle =
        speakrs::imported_segmentation::SegmentationBundle::open(temporary.path()).unwrap();
    assert_eq!(bundle.manifest.geometry.frame_grid.origin.numerator, -241);
    assert_eq!(bundle.manifest.geometry.aggregate_grid.origin.numerator, 0);
    assert_eq!(bundle.shards[0].shape, [1, 399, 11]);
    assert_eq!(bundle.shards[0].values.len(), 399 * 11);
}

#[test]
fn checked_repository_fixture_parses() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/wavlm_bridge");
    let temporary = bundle_tempdir();
    fs::copy(
        root.join("manifest.json"),
        temporary.path().join("manifest.json"),
    )
    .unwrap();
    fs::copy(root.join(".complete"), temporary.path().join(".complete")).unwrap();
    set_read_only(&temporary.path().join("manifest.json"), true);
    set_read_only(&temporary.path().join(".complete"), true);
    set_read_only(temporary.path(), true);
    let bundle = load_imported_segmentation_bundle(temporary.path()).unwrap();
    assert!(bundle.shards.is_empty());
    assert_eq!(bundle.manifest.audio.sample_count, 0);
    assert_eq!(
        bundle.manifest.geometry.output_extent_policy,
        OutputExtentPolicy::AggregateGrid
    );
}

#[test]
fn python_published_fixture_is_accepted_cross_language() {
    let fixture_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/wavlm_bridge/python_bundle");
    let published_root = bundle_tempdir();
    fs::create_dir_all(published_root.path().join("scores")).unwrap();
    for relative_path in ["manifest.json", ".complete", "scores/000000.npy"] {
        fs::copy(
            fixture_root.join(relative_path),
            published_root.path().join(relative_path),
        )
        .unwrap();
    }
    for relative_path in ["manifest.json", ".complete", "scores/000000.npy"] {
        set_read_only(&published_root.path().join(relative_path), true);
    }
    set_read_only(&published_root.path().join("scores"), true);
    set_read_only(published_root.path(), true);

    let bundle = SegmentationBundle::open(published_root.path()).unwrap();

    assert_eq!(
        bundle.manifest.identity.manifest_digest.as_str(),
        "f57270976fc903354f5cee01fb2bec442589b9bc717722266ac5ee2ce453bb6d"
    );
    assert_eq!(
        bundle.manifest.identity.bundle_id.as_str(),
        "56ac038bbed01d542f992478562e55f90c172af53f966b46471176a5ae8fe885"
    );
    assert_eq!(bundle.shards.len(), 1);
    assert_eq!(bundle.shards[0].shape, [1, 399, 11]);
    assert_eq!(bundle.shards[0].values, vec![0.0_f32; 399 * 11]);

    for path in [
        published_root.path().to_path_buf(),
        published_root.path().join("scores"),
        published_root.path().join("manifest.json"),
        published_root.path().join(".complete"),
        published_root.path().join("scores/000000.npy"),
    ] {
        let metadata = fs::symlink_metadata(&path).unwrap();
        assert!(!metadata.file_type().is_symlink());
        assert!(
            metadata.permissions().readonly(),
            "{} is writable",
            path.display()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            assert_eq!(metadata.permissions().mode() & 0o222, 0);
        }
    }
}

#[test]
fn canonical_identity_ignores_local_shard_paths() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (left, _) = make_manifest(
        4,
        2,
        1,
        "scores/left.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let (right, _) = make_manifest(
        4,
        2,
        1,
        "another/path.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    assert_eq!(
        canonical_manifest_digest(&left).unwrap(),
        canonical_manifest_digest(&right).unwrap()
    );
    assert_eq!(
        left.canonical_identity_bytes().unwrap(),
        right.canonical_identity_bytes().unwrap()
    );
}

#[test]
fn bundle_id_is_derived_and_rejects_a_caller_label() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (mut manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let derived = manifest.canonical_bundle_id().unwrap();
    manifest.identity.bundle_id = digest('3');
    assert_eq!(manifest.canonical_bundle_id().unwrap(), derived);
    assert!(manifest.validate().is_err());
}

#[test]
fn class_mapping_supports_required_head_sizes_and_permutations() {
    for (slots, overlap, classes) in [(4, 2, 11), (6, 2, 22), (6, 3, 42), (6, 6, 64)] {
        let mut mapping = class_mapping(slots, overlap);
        mapping.reverse();
        assert_eq!(mapping.len(), classes);
        let values = vec![0.0_f32; FRAME_COUNT as usize * classes];
        let (mut manifest, _) = make_manifest(
            slots,
            overlap,
            1,
            "scores/000000.npy",
            ScoreRepresentation::Logits,
            &values,
        );
        assert_eq!(manifest.head.class_to_slot_subsets.len(), classes);
        manifest.head.class_to_slot_subsets.reverse();
        rebind_identity(&mut manifest);
        manifest.validate().unwrap();
    }
}

#[test]
fn shard_ranges_must_cover_chunks_once_and_in_order() {
    let (manifest, _) = make_three_chunk_bundle();
    let original = manifest.tensors.shards[0].clone();

    let mut duplicate = manifest.clone();
    duplicate.tensors.shards = vec![original.clone(), original.clone()];
    assert!(duplicate.validate().is_err());

    let mut missing = manifest.clone();
    let mut first = original.clone();
    first.chunk_end = 1;
    first.shape[0] = 1;
    first.path = "scores/000000.npy".to_owned();
    let mut second = original.clone();
    second.chunk_start = 2;
    second.shape[0] = 1;
    second.path = "scores/000001.npy".to_owned();
    missing.tensors.shards = vec![first, second];
    assert!(missing.validate().is_err());

    let mut overlap = manifest;
    let mut first = original.clone();
    first.chunk_end = 2;
    first.shape[0] = 2;
    first.path = "scores/000000.npy".to_owned();
    let mut second = original;
    second.chunk_start = 1;
    second.shape[0] = 2;
    second.path = "scores/000001.npy".to_owned();
    overlap.tensors.shards = vec![first, second];
    assert!(overlap.validate().is_err());
}

#[test]
fn duplicate_class_subsets_are_rejected() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (mut manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    manifest.head.class_to_slot_subsets[10] = manifest.head.class_to_slot_subsets[9].clone();
    assert!(manifest.validate().is_err());
}

#[test]
fn excessive_and_overflowing_tensor_shapes_are_rejected_before_allocation() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let mut excessive = manifest.clone();
    excessive.tensors.shards[0].bytes = speakrs::imported_segmentation::MAX_SHARD_BYTES + 1;
    assert!(excessive.validate().is_err());

    let oversized_header = npy([u64::MAX, 1, 1], &[]);
    let rebound = rebind_shard(&manifest, &oversized_header);
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &rebound, &oversized_header);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());

    let overflowing_header = npy([u64::MAX, u64::MAX, 2], &[]);
    let rebound = rebind_shard(&manifest, &overflowing_header);
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &rebound, &overflowing_header);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
}

#[test]
fn malformed_manifest_and_paths_are_rejected() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, _) = make_manifest(
        4,
        2,
        1,
        "../outside.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    assert!(manifest.validate().is_err());

    let (valid, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let mut json = String::from_utf8(valid.to_json().unwrap()).unwrap();
    json.insert_str(json.len() - 1, ",\"unexpected\":true");
    assert!(speakrs::imported_segmentation::parse_manifest(json.as_bytes()).is_err());

    let mut duplicate = String::from_utf8(valid.to_json().unwrap()).unwrap();
    duplicate.insert_str(duplicate.len() - 1, ",\"format_version\":1");
    assert!(speakrs::imported_segmentation::parse_manifest(duplicate.as_bytes()).is_err());
}

#[test]
fn shard_corruption_and_trailing_bytes_are_rejected() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &manifest, &shard[..shard.len() - 1]);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());

    let temporary = bundle_tempdir();
    let mut trailing = shard;
    trailing.push(0);
    write_bundle(temporary.path(), &manifest, &trailing);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
}

#[test]
fn completion_marker_is_required_and_content_bound() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &manifest, &shard);
    let marker = temporary.path().join(".complete");
    set_read_only(&marker, false);
    fs::write(&marker, b"not-the-manifest-digest").unwrap();
    set_read_only(&marker, true);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());

    set_read_only(temporary.path(), false);
    fs::remove_file(&marker).unwrap();
    set_read_only(temporary.path(), true);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
}

#[test]
fn interrupted_staging_without_a_marker_is_not_a_bundle() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let temporary = bundle_tempdir();
    fs::create_dir_all(temporary.path().join("scores")).unwrap();
    let shard_path = temporary.path().join("scores/000000.npy");
    let manifest_path = temporary.path().join("manifest.json");
    fs::write(&shard_path, &shard).unwrap();
    fs::write(&manifest_path, manifest.to_json().unwrap()).unwrap();
    set_read_only(&shard_path, true);
    set_read_only(&manifest_path, true);
    set_read_only(&temporary.path().join("scores"), true);
    set_read_only(temporary.path(), true);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
}

#[test]
fn published_bundle_members_and_directories_must_be_read_only() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    for relative_path in ["manifest.json", ".complete", "scores/000000.npy"] {
        let temporary = bundle_tempdir();
        write_bundle(temporary.path(), &manifest, &shard);
        let path = temporary.path().join(relative_path);
        set_read_only(&path, false);
        assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
    }

    for relative_path in ["", "scores"] {
        let temporary = bundle_tempdir();
        write_bundle(temporary.path(), &manifest, &shard);
        let path = if relative_path.is_empty() {
            temporary.path().to_path_buf()
        } else {
            temporary.path().join(relative_path)
        };
        set_read_only(&path, false);
        assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
    }
}

#[test]
fn npy_headers_are_checked_before_values_are_loaded() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let mut wrong_dtype = shard.clone();
    let marker = b"'descr': '<f4'";
    let replacement = b"'descr': '>f4'";
    let position = wrong_dtype
        .windows(marker.len())
        .position(|window| window == marker)
        .unwrap();
    wrong_dtype[position..position + marker.len()].copy_from_slice(replacement);
    let rebound = rebind_shard(&manifest, &wrong_dtype);
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &rebound, &wrong_dtype);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());

    for replacement in [b"'descr': '=f4'", b"'descr': '|f4'"] {
        let mut wrong_dtype = shard.clone();
        wrong_dtype[position..position + marker.len()].copy_from_slice(replacement);
        let rebound = rebind_shard(&manifest, &wrong_dtype);
        let temporary = bundle_tempdir();
        write_bundle(temporary.path(), &rebound, &wrong_dtype);
        assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
    }

    let mut wrong_version = shard;
    wrong_version[6] = 2;
    let rebound = rebind_shard(&manifest, &wrong_version);
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &rebound, &wrong_version);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
}

#[cfg(unix)]
#[test]
fn symlinked_members_are_rejected() {
    use std::os::unix::fs::symlink;

    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let temporary = bundle_tempdir();
    fs::create_dir_all(temporary.path().join("scores")).unwrap();
    let outside = tempfile::tempdir().unwrap();
    let outside_path = outside.path().join("outside.npy");
    fs::write(&outside_path, &shard).unwrap();
    set_read_only(&outside_path, true);
    symlink(
        outside.path().join("outside.npy"),
        temporary.path().join("scores/000000.npy"),
    )
    .unwrap();
    let manifest_path = temporary.path().join("manifest.json");
    let marker_path = temporary.path().join(".complete");
    fs::write(&manifest_path, manifest.to_json().unwrap()).unwrap();
    fs::write(&marker_path, manifest.identity.manifest_digest.as_str()).unwrap();
    set_read_only(&manifest_path, true);
    set_read_only(&marker_path, true);
    set_read_only(&temporary.path().join("scores"), true);
    set_read_only(temporary.path(), true);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
}

#[cfg(unix)]
#[test]
fn symlinked_parent_directories_cannot_escape_the_bundle() {
    use std::os::unix::fs::symlink;

    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let temporary = bundle_tempdir();
    let outside = tempfile::tempdir().unwrap();
    let outside_scores = outside.path().join("scores");
    fs::create_dir_all(&outside_scores).unwrap();
    let outside_shard = outside_scores.join("000000.npy");
    fs::write(&outside_shard, &shard).unwrap();
    set_read_only(&outside_shard, true);
    symlink(&outside_scores, temporary.path().join("scores")).unwrap();

    let manifest_path = temporary.path().join("manifest.json");
    let marker_path = temporary.path().join(".complete");
    fs::write(&manifest_path, manifest.to_json().unwrap()).unwrap();
    fs::write(&marker_path, manifest.identity.manifest_digest.as_str()).unwrap();
    set_read_only(&manifest_path, true);
    set_read_only(&marker_path, true);
    set_read_only(temporary.path(), true);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());
}

#[test]
fn probability_and_log_probability_values_are_checked() {
    let probability_values = vec![1.0 / 11.0; FRAME_COUNT as usize * 11];
    let (manifest, shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Probabilities,
        &probability_values,
    );
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &manifest, &shard);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_ok());

    let mut invalid_probability_values = probability_values;
    invalid_probability_values[0] = 2.0;
    let (invalid_manifest, invalid_shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Probabilities,
        &invalid_probability_values,
    );
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &invalid_manifest, &invalid_shard);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_err());

    let mut log_values = vec![-(10.0_f32).ln(); FRAME_COUNT as usize * 11];
    for index in (0..log_values.len()).step_by(11) {
        log_values[index] = f32::NEG_INFINITY;
    }
    let (log_manifest, log_shard) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::LogProbabilities,
        &log_values,
    );
    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &log_manifest, &log_shard);
    assert!(load_imported_segmentation_bundle(temporary.path()).is_ok());
}

#[test]
fn padded_geometry_and_empty_audio_have_explicit_extent() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let chunk = &manifest.geometry.chunks[0];
    assert_eq!(chunk.valid_samples, 1);
    assert_eq!(chunk.padding_samples, WINDOW_SAMPLES - 1);
    assert_eq!(manifest.geometry.output_extent.end_samples, 128_400);
    assert!(manifest.geometry.output_extent.end_samples > manifest.audio.sample_count);

    let complete_values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (complete, _) = make_manifest(
        4,
        2,
        WINDOW_SAMPLES,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &complete_values,
    );
    assert_eq!(complete.geometry.output_extent.end_samples, 128_400);
    complete.validate().unwrap();

    let (empty, _) = make_manifest(4, 2, 0, "unused.npy", ScoreRepresentation::Logits, &[]);
    assert!(empty.geometry.chunks.is_empty());
    assert!(empty.tensors.shards.is_empty());
    empty.validate().unwrap();
}

#[test]
fn audio_extent_is_an_explicit_normalized_view() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (mut manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    manifest.geometry.output_extent_policy = OutputExtentPolicy::AudioExtent;
    manifest.geometry.output_extent.end_samples = manifest.audio.sample_count;
    manifest.policy.reconstruction.extent_policy = OutputExtentPolicy::AudioExtent;
    rebind_identity(&mut manifest);
    manifest.validate().unwrap();
}

#[test]
fn aggregate_extent_rejects_non_integral_rational_endpoints() {
    let values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (mut manifest, _) = make_manifest(
        4,
        2,
        320,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &values,
    );
    let step = RationalSample {
        numerator: 321,
        denominator: 2,
    };
    manifest.geometry.aggregate_grid.step = step;
    manifest.geometry.frame_grid.step = step;
    manifest.geometry.aggregate_grid.frame_count = 796;
    manifest.geometry.frame_grid.frame_count = 796;
    manifest.geometry.output_extent.end_samples = 127_760;
    manifest.tensors.shards[0].shape[1] = 796;
    rebind_identity(&mut manifest);
    assert!(manifest.validate().is_err());
}

#[test]
fn regular_fixed_step_geometry_covers_a_padded_tail_exactly() {
    let one_chunk = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (mut manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &one_chunk,
    );
    manifest.audio.sample_count = 150_000;
    manifest.geometry.output_extent.end_samples = 154_000;
    manifest.geometry.chunks = vec![
        ChunkGeometry {
            index: 0,
            padding_samples: 0,
            start_samples: 0,
            valid_samples: 128_000,
        },
        ChunkGeometry {
            index: 1,
            padding_samples: 0,
            start_samples: 12_800,
            valid_samples: 128_000,
        },
        ChunkGeometry {
            index: 2,
            padding_samples: 3_600,
            start_samples: 25_600,
            valid_samples: 124_400,
        },
    ];
    let values = vec![0.0_f32; 3 * FRAME_COUNT as usize * 11];
    let shard = npy([3, FRAME_COUNT as u64, 11], &values);
    manifest.tensors.shards[0].bytes = shard.len() as u64;
    manifest.tensors.shards[0].chunk_end = 3;
    manifest.tensors.shards[0].shape = [3, FRAME_COUNT as u64, 11];
    manifest.tensors.shards[0].sha256 = file_digest(&shard);
    rebind_identity(&mut manifest);
    manifest.validate().unwrap();

    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &manifest, &shard);
    let bundle = load_imported_segmentation_bundle(temporary.path()).unwrap();
    assert_eq!(bundle.shards[0].shape[0], 3);
}

#[test]
fn aggregate_extent_tracks_overlapping_window_offsets() {
    let one_chunk_values = vec![0.0_f32; FRAME_COUNT as usize * 11];
    let (mut manifest, _) = make_manifest(
        4,
        2,
        1,
        "scores/000000.npy",
        ScoreRepresentation::Logits,
        &one_chunk_values,
    );
    manifest.audio.sample_count = 134_400;
    manifest.geometry.output_extent.end_samples = 134_800;
    manifest.geometry.window_planning.step_samples = 6_400;
    manifest.geometry.chunks = vec![
        ChunkGeometry {
            index: 0,
            padding_samples: 0,
            start_samples: 0,
            valid_samples: 128_000,
        },
        ChunkGeometry {
            index: 1,
            padding_samples: 0,
            start_samples: 6_400,
            valid_samples: 128_000,
        },
    ];
    let values = vec![0.0_f32; 2 * FRAME_COUNT as usize * 11];
    let shape = [2, FRAME_COUNT as u64, 11];
    let shard = npy(shape, &values);
    let declaration = &mut manifest.tensors.shards[0];
    declaration.bytes = shard.len() as u64;
    declaration.chunk_end = 2;
    declaration.shape = shape;
    declaration.sha256 = file_digest(&shard);
    rebind_identity(&mut manifest);
    manifest.validate().unwrap();

    let temporary = bundle_tempdir();
    write_bundle(temporary.path(), &manifest, &shard);
    let bundle = load_imported_segmentation_bundle(temporary.path()).unwrap();
    assert_eq!(bundle.manifest.geometry.output_extent.end_samples, 134_800);
}
