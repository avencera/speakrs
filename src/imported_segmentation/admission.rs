//! Filesystem, NPY, and score admission for immutable bundles

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use sha2::{Digest, Sha256};

use super::canonical::parse_manifest;
use super::validation::{invalid, validate_member_path};
use super::*;

struct PreparedBundle {
    root: PathBuf,
    manifest: SegmentationManifest,
    shard_paths: Vec<PathBuf>,
}

/// Streams and validates one imported segmentation bundle without retaining
/// its score values
pub fn validate_imported_segmentation_bundle(
    root: impl AsRef<Path>,
) -> Result<SegmentationBundleValidation, SegmentationBundleError> {
    let prepared = prepare_bundle(root)?;
    let classes = prepared.manifest.head.class_to_slot_subsets.len();
    let representation = prepared.manifest.head.score_representation;
    let mut measurements = ScoreMeasurementsAccumulator::default();

    for (declaration, path) in prepared
        .manifest
        .tensors
        .shards
        .iter()
        .zip(prepared.shard_paths.iter())
    {
        let mut validator = ScoreValidator::new(representation, classes)?;
        let scanned = scan_npy_shard(path, declaration, |value| validator.push(value))?;
        let shard_measurements = validator.finish()?;
        measurements.record(declaration, scanned.elements, shard_measurements)?;
    }

    let manifest = prepared.manifest;
    Ok(SegmentationBundleValidation {
        schema_version: VALIDATION_RESULT_SCHEMA_VERSION,
        format_version: manifest.format_version,
        schema_id: manifest.schema_id.clone(),
        audio: manifest.audio.clone(),
        identity: manifest.identity.clone(),
        geometry: manifest.geometry.clone(),
        score_stage: manifest.tensors.score_stage,
        score_representation: representation,
        dimensions: ScoreTensorDimensions {
            chunks: manifest.geometry.chunks.len() as u64,
            frames: u64::from(manifest.geometry.frame_grid.frame_count),
            classes: classes as u64,
        },
        measurements: measurements.finish(),
    })
}

/// Loads and validates a published bundle without loading a segmentation
/// model
pub fn load_imported_segmentation_bundle(
    root: impl AsRef<Path>,
) -> Result<ImportedSegmentationBundle, SegmentationBundleError> {
    let prepared = prepare_bundle(root)?;
    let classes = prepared.manifest.head.class_to_slot_subsets.len();
    let representation = prepared.manifest.head.score_representation;
    let mut shards = Vec::with_capacity(prepared.manifest.tensors.shards.len());

    for (declaration, path) in prepared
        .manifest
        .tensors
        .shards
        .iter()
        .zip(prepared.shard_paths.iter())
    {
        let (shape, values) = read_npy_shard(path, declaration, representation, classes)?;
        shards.push(LoadedTensorShard {
            path: declaration.path.clone(),
            chunk_start: declaration.chunk_start,
            chunk_end: declaration.chunk_end,
            shape,
            values,
        });
    }

    Ok(ImportedSegmentationBundle {
        root: prepared.root,
        manifest: prepared.manifest,
        shards,
    })
}

fn prepare_bundle(root: impl AsRef<Path>) -> Result<PreparedBundle, SegmentationBundleError> {
    let root = root.as_ref();
    let root_metadata = fs::symlink_metadata(root)?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(SegmentationBundleError::Invalid(
            "bundle root must be a real directory".to_owned(),
        ));
    }
    let root = root.canonicalize()?;
    ensure_read_only_directory(&root, "bundle root")?;
    let manifest_path = root.join("manifest.json");
    reject_symlink(&manifest_path, "manifest")?;
    ensure_read_only(&manifest_path, "manifest")?;
    let manifest_bytes = read_bounded_file(&manifest_path, MAX_MANIFEST_BYTES as u64)?;
    let manifest = parse_manifest(&manifest_bytes)?;

    let marker_path = root.join(".complete");
    reject_symlink(&marker_path, "publication marker")?;
    ensure_read_only(&marker_path, "publication marker")?;
    let marker = read_bounded_file(&marker_path, 65)?;
    let expected_digest = manifest.identity.manifest_digest.as_str().as_bytes();
    let marker_matches = marker == expected_digest
        || marker
            .strip_suffix(b"\n")
            .is_some_and(|value| value == expected_digest);
    if !marker_matches {
        return Err(SegmentationBundleError::Invalid(
            "publication marker does not match manifest digest".to_owned(),
        ));
    }

    validate_tree_membership(&root, &manifest)?;

    let mut shard_paths = Vec::with_capacity(manifest.tensors.shards.len());
    for declaration in &manifest.tensors.shards {
        let path = checked_member_path(&root, &declaration.path)?;
        ensure_read_only(&path, "tensor shard")?;
        shard_paths.push(path);
    }

    Ok(PreparedBundle {
        root,
        manifest,
        shard_paths,
    })
}

impl ImportedSegmentationBundle {
    /// Opens and validates a published bundle directory
    pub fn open(root: impl AsRef<Path>) -> Result<Self, SegmentationBundleError> {
        load_imported_segmentation_bundle(root)
    }
}

fn validate_tree_membership(
    root: &Path,
    manifest: &SegmentationManifest,
) -> Result<(), SegmentationBundleError> {
    let mut expected_files =
        BTreeSet::from([PathBuf::from(".complete"), PathBuf::from("manifest.json")]);
    let mut expected_directories = BTreeSet::new();
    for shard in &manifest.tensors.shards {
        let path = PathBuf::from(&shard.path);
        expected_files.insert(path.clone());
        let mut parent = path.parent();
        while let Some(directory) = parent {
            if directory.as_os_str().is_empty() {
                break;
            }
            expected_directories.insert(directory.to_owned());
            parent = directory.parent();
        }
    }

    let mut found_files = BTreeSet::new();
    walk_bundle_tree(
        root,
        Path::new(""),
        &expected_files,
        &expected_directories,
        &mut found_files,
    )?;
    if let Some(missing) = expected_files.difference(&found_files).next() {
        return invalid(format!(
            "bundle is missing expected member {}",
            missing.display()
        ));
    }
    Ok(())
}

fn walk_bundle_tree(
    directory: &Path,
    relative_directory: &Path,
    expected_files: &BTreeSet<PathBuf>,
    expected_directories: &BTreeSet<PathBuf>,
    found_files: &mut BTreeSet<PathBuf>,
) -> Result<(), SegmentationBundleError> {
    ensure_read_only_directory(directory, "bundle directory")?;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let relative = relative_directory.join(&name);
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return invalid(format!(
                "bundle member {} must not be a symlink",
                relative.display()
            ));
        }
        if metadata.is_dir() {
            if !expected_directories.contains(&relative) {
                return invalid(format!(
                    "unexpected bundle directory {}",
                    relative.display()
                ));
            }
            walk_bundle_tree(
                &path,
                &relative,
                expected_files,
                expected_directories,
                found_files,
            )?;
        } else if metadata.is_file() {
            if !expected_files.contains(&relative) {
                return invalid(format!("unexpected bundle member {}", relative.display()));
            }
            ensure_read_only_metadata(&metadata, "bundle member")?;
            found_files.insert(relative);
        } else {
            return invalid(format!(
                "bundle member {} is not a file or directory",
                relative.display()
            ));
        }
    }
    Ok(())
}

fn checked_member_path(root: &Path, member: &str) -> Result<PathBuf, SegmentationBundleError> {
    validate_member_path(member)?;
    ensure_read_only_member_parents(root, member)?;
    let path = root.join(member);
    reject_symlink(&path, "tensor shard")?;
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(root) {
        return invalid(format!("tensor member path escapes bundle root: {member}"));
    }
    ensure_read_only_member_parents(root, member)?;
    if let Some(parent) = canonical.parent() {
        ensure_read_only_directory(parent, "tensor shard parent directory")?;
    }
    Ok(canonical)
}

fn ensure_read_only_member_parents(
    root: &Path,
    member: &str,
) -> Result<(), SegmentationBundleError> {
    ensure_read_only_directory(root, "bundle root")?;
    let relative = Path::new(member);
    let Some(parent) = relative.parent() else {
        return Ok(());
    };
    let mut current = root.to_path_buf();
    for component in parent.components() {
        let Component::Normal(name) = component else {
            return invalid(format!("unsafe tensor member parent path: {member}"));
        };
        current.push(name);
        ensure_read_only_directory(&current, "tensor shard parent directory")?;
    }
    Ok(())
}

fn reject_symlink(path: &Path, context: &str) -> Result<(), SegmentationBundleError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{context} does not exist at {}", path.display()),
            )
        } else {
            error
        }
    })?;
    if metadata.file_type().is_symlink() {
        return invalid(format!("{context} must not be a symlink"));
    }
    Ok(())
}

fn ensure_read_only(path: &Path, context: &str) -> Result<(), SegmentationBundleError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return invalid(format!("{context} must not be a symlink"));
    }
    ensure_read_only_metadata(&metadata, context)
}

fn ensure_read_only_directory(path: &Path, context: &str) -> Result<(), SegmentationBundleError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return invalid(format!("{context} must be a real directory"));
    }
    ensure_read_only_metadata(&metadata, context)
}

fn ensure_read_only_metadata(
    metadata: &fs::Metadata,
    context: &str,
) -> Result<(), SegmentationBundleError> {
    #[cfg(unix)]
    {
        if metadata.permissions().mode() & 0o222 != 0 {
            return invalid(format!("{context} must be read-only"));
        }
    }
    #[cfg(not(unix))]
    {
        // non-Unix metadata cannot expose per-principal write bits
        if !metadata.permissions().readonly() {
            return invalid(format!("{context} must be read-only"));
        }
    }
    Ok(())
}

fn read_bounded_file(path: &Path, maximum: u64) -> Result<Vec<u8>, SegmentationBundleError> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > maximum {
        return invalid(format!("file {} exceeds its bound", path.display()));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        SegmentationBundleError::Invalid(format!(
            "file {} is too large for this host",
            path.display()
        ))
    })?;
    let file = File::open(path)?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() || bytes.len() as u64 > maximum {
        return invalid(format!("file {} changed while being read", path.display()));
    }
    Ok(bytes)
}

fn digest_from_hash(digest: &[u8]) -> Result<Sha256Digest, SegmentationBundleError> {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(text, "{byte:02x}")
            .map_err(|_| SegmentationBundleError::Invalid("digest formatting failed".to_owned()))?;
    }
    Sha256Digest::new(text)
}

fn read_npy_shard(
    path: &Path,
    declaration: &TensorShard,
    representation: ScoreRepresentation,
    classes: usize,
) -> Result<([u64; 3], Vec<f32>), SegmentationBundleError> {
    let mut values = Vec::new();
    let mut validator = ScoreValidator::new(representation, classes)?;
    let scanned = scan_npy_shard(path, declaration, |value| {
        validator.push(value)?;
        values.push(value);
        Ok(())
    })?;
    validator.finish()?;
    Ok((scanned.shape, values))
}

struct ScannedShard {
    shape: [u64; 3],
    elements: u64,
}

fn scan_npy_shard<F>(
    path: &Path,
    declaration: &TensorShard,
    mut consume: F,
) -> Result<ScannedShard, SegmentationBundleError>
where
    F: FnMut(f32) -> Result<(), SegmentationBundleError>,
{
    const PREFIX_LEN: usize = 10;
    let metadata = fs::metadata(path)?;
    if !metadata.is_file()
        || metadata.len() != declaration.bytes
        || metadata.len() > MAX_SHARD_BYTES
    {
        return invalid(format!(
            "shard {} changed size or exceeds its bound",
            declaration.path
        ));
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut prefix = [0_u8; PREFIX_LEN];
    file.read_exact(&mut prefix)?;
    hasher.update(prefix);
    if &prefix[..6] != b"\x93NUMPY" {
        return invalid("NPY magic is invalid".to_owned());
    }
    if prefix[6] != 1 || prefix[7] != 0 {
        return invalid("only NPY format version 1.0 is admitted".to_owned());
    }
    let header_len = usize::from(u16::from_le_bytes([prefix[8], prefix[9]]));
    if header_len > MAX_NPY_HEADER_BYTES {
        return invalid("NPY header exceeds the admission bound".to_owned());
    }
    let data_start = PREFIX_LEN
        .checked_add(header_len)
        .ok_or_else(|| SegmentationBundleError::Invalid("NPY header offset overflow".to_owned()))?;
    if metadata.len() < data_start as u64 {
        return invalid("NPY header is truncated".to_owned());
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)?;
    hasher.update(&header);
    let parsed = parse_npy_header(&header)?;
    if parsed != declaration.shape {
        return invalid(format!(
            "shard {} shape does not match its inventory",
            declaration.path
        ));
    }
    let elements = tensor_elements(&parsed)?;
    let payload_bytes = elements
        .checked_mul(4)
        .ok_or_else(|| SegmentationBundleError::Invalid("NPY byte count overflow".to_owned()))?;
    let expected_bytes = (data_start as u64)
        .checked_add(payload_bytes)
        .ok_or_else(|| SegmentationBundleError::Invalid("NPY byte count overflow".to_owned()))?;
    if declaration.bytes != expected_bytes {
        return invalid("NPY payload is truncated or has trailing bytes".to_owned());
    }

    scan_score_payload(&mut file, payload_bytes, &mut hasher, &mut consume)?;
    let mut trailing = [0_u8; 1];
    if file.read(&mut trailing)? != 0 {
        return invalid("NPY payload has trailing bytes".to_owned());
    }
    let actual_digest = digest_from_hash(&hasher.finalize())?;
    if actual_digest != declaration.sha256 {
        return invalid(format!(
            "shard {} digest does not match its inventory",
            declaration.path
        ));
    }
    Ok(ScannedShard {
        shape: parsed,
        elements,
    })
}

fn scan_score_payload<R, F>(
    reader: &mut R,
    payload_bytes: u64,
    hasher: &mut Sha256,
    consume: &mut F,
) -> Result<(), SegmentationBundleError>
where
    R: Read,
    F: FnMut(f32) -> Result<(), SegmentationBundleError>,
{
    let mut buffer = [0_u8; SCORE_VALIDATION_CHUNK_BYTES];
    let mut remaining = payload_bytes;
    while remaining > 0 {
        let read_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| SegmentationBundleError::Invalid("NPY chunk is too large".to_owned()))?;
        reader.read_exact(&mut buffer[..read_len])?;
        hasher.update(&buffer[..read_len]);
        let (values, remainder) = buffer[..read_len].as_chunks::<4>();
        if !remainder.is_empty() {
            return invalid("NPY payload is not aligned to float32 values".to_owned());
        }
        for value in values {
            let value = f32::from_le_bytes([value[0], value[1], value[2], value[3]]);
            consume(value)?;
        }
        remaining -= read_len as u64;
    }
    Ok(())
}

fn tensor_elements(shape: &[u64; 3]) -> Result<u64, SegmentationBundleError> {
    let elements = shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| SegmentationBundleError::Invalid("NPY element count overflow".to_owned()))?;
    if elements > MAX_TENSOR_ELEMENTS {
        return invalid("NPY element count exceeds the admission bound".to_owned());
    }
    Ok(elements)
}

#[derive(Default)]
struct ScoreMeasurementsAccumulator {
    shard_count: u64,
    serialized_bytes: u64,
    value_count: u64,
    row_count: u64,
    finite_value_count: u64,
    finite_minimum: Option<f32>,
    finite_maximum: Option<f32>,
}

impl ScoreMeasurementsAccumulator {
    fn record(
        &mut self,
        declaration: &TensorShard,
        elements: u64,
        shard: ScoreShardMeasurements,
    ) -> Result<(), SegmentationBundleError> {
        if shard.value_count != elements {
            return invalid("score scanner value count does not match its shape".to_owned());
        }
        self.shard_count = self.shard_count.checked_add(1).ok_or_else(|| {
            SegmentationBundleError::Invalid("score shard count overflow".to_owned())
        })?;
        self.serialized_bytes = self
            .serialized_bytes
            .checked_add(declaration.bytes)
            .ok_or_else(|| {
                SegmentationBundleError::Invalid("score byte count overflow".to_owned())
            })?;
        self.value_count = self
            .value_count
            .checked_add(shard.value_count)
            .ok_or_else(|| {
                SegmentationBundleError::Invalid("score value count overflow".to_owned())
            })?;
        self.row_count = self.row_count.checked_add(shard.row_count).ok_or_else(|| {
            SegmentationBundleError::Invalid("score row count overflow".to_owned())
        })?;
        self.finite_value_count = self
            .finite_value_count
            .checked_add(shard.finite_value_count)
            .ok_or_else(|| {
                SegmentationBundleError::Invalid("finite score count overflow".to_owned())
            })?;
        self.finite_minimum = match (self.finite_minimum, shard.finite_minimum) {
            (Some(left), Some(right)) => Some(if left.total_cmp(&right).is_le() {
                left
            } else {
                right
            }),
            (left, right) => left.or(right),
        };
        self.finite_maximum = match (self.finite_maximum, shard.finite_maximum) {
            (Some(left), Some(right)) => Some(if left.total_cmp(&right).is_ge() {
                left
            } else {
                right
            }),
            (left, right) => left.or(right),
        };
        Ok(())
    }

    fn finish(self) -> ScoreValidationMeasurements {
        ScoreValidationMeasurements {
            shard_count: self.shard_count,
            serialized_bytes: self.serialized_bytes,
            value_count: self.value_count,
            row_count: self.row_count,
            finite_value_count: self.finite_value_count,
            finite_minimum: self.finite_minimum,
            finite_maximum: self.finite_maximum,
        }
    }
}

struct ScoreValidator {
    representation: ScoreRepresentation,
    classes: usize,
    row: Vec<f32>,
    value_count: u64,
    row_count: u64,
    finite_value_count: u64,
    finite_minimum: Option<f32>,
    finite_maximum: Option<f32>,
}

struct ScoreShardMeasurements {
    value_count: u64,
    row_count: u64,
    finite_value_count: u64,
    finite_minimum: Option<f32>,
    finite_maximum: Option<f32>,
}

impl ScoreValidator {
    fn new(
        representation: ScoreRepresentation,
        classes: usize,
    ) -> Result<Self, SegmentationBundleError> {
        if classes == 0 {
            return invalid("score tensor class extent is invalid".to_owned());
        }
        Ok(Self {
            representation,
            classes,
            row: Vec::with_capacity(classes),
            value_count: 0,
            row_count: 0,
            finite_value_count: 0,
            finite_minimum: None,
            finite_maximum: None,
        })
    }

    fn push(&mut self, value: f32) -> Result<(), SegmentationBundleError> {
        self.value_count = self.value_count.checked_add(1).ok_or_else(|| {
            SegmentationBundleError::Invalid("score value count overflow".to_owned())
        })?;
        if value.is_finite() {
            self.finite_value_count = self.finite_value_count.checked_add(1).ok_or_else(|| {
                SegmentationBundleError::Invalid("finite score count overflow".to_owned())
            })?;
            self.finite_minimum = Some(match self.finite_minimum {
                Some(current) if current.total_cmp(&value).is_le() => current,
                _ => value,
            });
            self.finite_maximum = Some(match self.finite_maximum {
                Some(current) if current.total_cmp(&value).is_ge() => current,
                _ => value,
            });
        }
        self.row.push(value);
        if self.row.len() == self.classes {
            validate_score_row(&self.row, self.representation)?;
            self.row.clear();
            self.row_count = self.row_count.checked_add(1).ok_or_else(|| {
                SegmentationBundleError::Invalid("score row count overflow".to_owned())
            })?;
        }
        Ok(())
    }

    fn finish(self) -> Result<ScoreShardMeasurements, SegmentationBundleError> {
        if !self.row.is_empty() {
            return invalid("score tensor class extent is invalid".to_owned());
        }
        Ok(ScoreShardMeasurements {
            value_count: self.value_count,
            row_count: self.row_count,
            finite_value_count: self.finite_value_count,
            finite_minimum: self.finite_minimum,
            finite_maximum: self.finite_maximum,
        })
    }
}

fn parse_npy_header(header: &[u8]) -> Result<[u64; 3], SegmentationBundleError> {
    let mut parser = HeaderParser::new(header);
    parser.parse()
}

struct HeaderParser<'a> {
    input: &'a [u8],
    position: usize,
    seen_descr: bool,
    seen_fortran: bool,
    seen_shape: bool,
    descr: Option<String>,
    fortran: Option<bool>,
    shape: Option<[u64; 3]>,
}

impl<'a> HeaderParser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            position: 0,
            seen_descr: false,
            seen_fortran: false,
            seen_shape: false,
            descr: None,
            fortran: None,
            shape: None,
        }
    }

    fn parse(&mut self) -> Result<[u64; 3], SegmentationBundleError> {
        self.whitespace();
        self.expect(b'{')?;
        loop {
            self.whitespace();
            if self.consume(b'}') {
                break;
            }
            let key = self.quoted()?;
            self.whitespace();
            self.expect(b':')?;
            self.whitespace();
            match key.as_str() {
                "descr" => {
                    if self.seen_descr {
                        return invalid("NPY header repeats descr".to_owned());
                    }
                    self.seen_descr = true;
                    self.descr = Some(self.quoted()?);
                }
                "fortran_order" => {
                    if self.seen_fortran {
                        return invalid("NPY header repeats fortran_order".to_owned());
                    }
                    self.seen_fortran = true;
                    self.fortran = Some(self.boolean()?);
                }
                "shape" => {
                    if self.seen_shape {
                        return invalid("NPY header repeats shape".to_owned());
                    }
                    self.seen_shape = true;
                    self.shape = Some(self.shape()?);
                }
                _ => return invalid(format!("NPY header contains unknown field {key}")),
            }
            self.whitespace();
            if self.consume(b',') {
                continue;
            }
            self.expect(b'}')?;
            break;
        }
        self.whitespace();
        if self.position != self.input.len() {
            return invalid("NPY header has trailing data".to_owned());
        }
        if self.descr.as_deref() != Some("<f4") {
            return invalid("NPY dtype must be explicit little-endian float32".to_owned());
        }
        if self.fortran != Some(false) {
            return invalid("NPY array must use C order".to_owned());
        }
        self.shape
            .ok_or_else(|| SegmentationBundleError::Invalid("NPY shape is missing".to_owned()))
    }

    fn shape(&mut self) -> Result<[u64; 3], SegmentationBundleError> {
        self.expect(b'(')?;
        let mut dimensions = Vec::new();
        loop {
            self.whitespace();
            if self.consume(b')') {
                break;
            }
            let start = self.position;
            while self.position < self.input.len() && self.input[self.position].is_ascii_digit() {
                self.position += 1;
            }
            if start == self.position {
                return invalid("NPY shape contains a non-integer dimension".to_owned());
            }
            let value = std::str::from_utf8(&self.input[start..self.position])
                .map_err(|_| SegmentationBundleError::Invalid("NPY shape is not ASCII".to_owned()))?
                .parse::<u64>()
                .map_err(|_| {
                    SegmentationBundleError::Invalid("NPY shape dimension overflows".to_owned())
                })?;
            dimensions.push(value);
            self.whitespace();
            if self.consume(b',') {
                continue;
            }
            self.expect(b')')?;
            break;
        }
        if dimensions.len() != 3 {
            return invalid("NPY tensor must have rank three".to_owned());
        }
        Ok([dimensions[0], dimensions[1], dimensions[2]])
    }

    fn quoted(&mut self) -> Result<String, SegmentationBundleError> {
        let quote = self.input.get(self.position).copied().ok_or_else(|| {
            SegmentationBundleError::Invalid("NPY header string is truncated".to_owned())
        })?;
        if quote != b'\'' && quote != b'"' {
            return invalid("NPY header expects a quoted string".to_owned());
        }
        self.position += 1;
        let start = self.position;
        while self.position < self.input.len() && self.input[self.position] != quote {
            if self.input[self.position] == b'\\' {
                return invalid("NPY header escapes are not admitted".to_owned());
            }
            self.position += 1;
        }
        if self.position == self.input.len() {
            return invalid("NPY header string is unterminated".to_owned());
        }
        let value = std::str::from_utf8(&self.input[start..self.position])
            .map_err(|_| SegmentationBundleError::Invalid("NPY header is not ASCII".to_owned()))?
            .to_owned();
        self.position += 1;
        Ok(value)
    }

    fn boolean(&mut self) -> Result<bool, SegmentationBundleError> {
        if self
            .input
            .get(self.position..)
            .is_some_and(|value| value.starts_with(b"False"))
        {
            self.position += 5;
            return Ok(false);
        }
        if self
            .input
            .get(self.position..)
            .is_some_and(|value| value.starts_with(b"True"))
        {
            self.position += 4;
            return Ok(true);
        }
        invalid("NPY fortran_order must be True or False".to_owned())
    }

    fn whitespace(&mut self) {
        while self
            .input
            .get(self.position)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), SegmentationBundleError> {
        if self.consume(expected) {
            Ok(())
        } else {
            invalid(format!("NPY header expected {:?}", expected as char))
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.input.get(self.position) == Some(&expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }
}

fn validate_score_row(
    row: &[f32],
    representation: ScoreRepresentation,
) -> Result<(), SegmentationBundleError> {
    match representation {
        ScoreRepresentation::Logits => {
            if row.iter().any(|value| !value.is_finite()) {
                return invalid("logit rows must contain only finite values".to_owned());
            }
        }
        ScoreRepresentation::Probabilities => {
            let mut sum = 0.0_f64;
            for value in row {
                let value = f64::from(*value);
                if !value.is_finite()
                    || !(-SCORE_PROBABILITY_TOLERANCE..=1.0 + SCORE_PROBABILITY_TOLERANCE)
                        .contains(&value)
                {
                    return invalid("probability values are outside [0, 1]".to_owned());
                }
                sum += value;
            }
            if (sum - 1.0).abs() > SCORE_PROBABILITY_TOLERANCE {
                return invalid("probability rows are not normalized".to_owned());
            }
        }
        ScoreRepresentation::LogProbabilities => {
            let mut max = f64::NEG_INFINITY;
            let mut possible = false;
            for value in row {
                let value = f64::from(*value);
                if value.is_nan() || value.is_infinite() && value.is_sign_positive() {
                    return invalid("log-probability rows contain NaN or +inf".to_owned());
                }
                if value.is_finite() {
                    possible = true;
                    max = max.max(value);
                }
            }
            if !possible {
                return invalid("log-probability rows need one possible class".to_owned());
            }
            let sum = row
                .iter()
                .map(|value| (f64::from(*value) - max).exp())
                .sum::<f64>()
                .ln()
                + max;
            if sum.abs() > SCORE_LOG_PROBABILITY_TOLERANCE {
                return invalid("log-probability rows are not normalized".to_owned());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TrackingReader {
        bytes: Vec<u8>,
        offset: usize,
        largest_request: usize,
    }

    impl Read for TrackingReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.largest_request = self.largest_request.max(buffer.len());
            let remaining = self.bytes.len() - self.offset;
            let count = remaining.min(buffer.len());
            buffer[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
            self.offset += count;
            Ok(count)
        }
    }

    fn next_up(value: f32) -> f32 {
        if value.is_sign_negative() {
            f32::from_bits(value.to_bits() - 1)
        } else {
            f32::from_bits(value.to_bits() + 1)
        }
    }

    #[test]
    fn probability_normalization_tolerance_is_inclusive() {
        let mut inside = 0.5_f32;
        let mut outside = next_up(inside);
        while (f64::from(outside) * 2.0 - 1.0).abs() <= SCORE_PROBABILITY_TOLERANCE {
            inside = outside;
            outside = next_up(outside);
        }

        assert!(validate_score_row(&[inside, inside], ScoreRepresentation::Probabilities).is_ok());
        assert!(
            validate_score_row(&[outside, outside], ScoreRepresentation::Probabilities).is_err()
        );
    }

    #[test]
    fn log_probability_normalization_tolerance_is_inclusive() {
        let mut inside = -(2.0_f64).ln() as f32;
        let mut outside = next_up(inside);
        while (f64::from(outside) + 2.0_f64.ln()).abs() <= SCORE_LOG_PROBABILITY_TOLERANCE {
            inside = outside;
            outside = next_up(outside);
        }

        assert!(
            validate_score_row(&[inside, inside], ScoreRepresentation::LogProbabilities).is_ok()
        );
        assert!(
            validate_score_row(&[outside, outside], ScoreRepresentation::LogProbabilities).is_err()
        );
    }

    #[test]
    fn score_payload_is_scanned_in_bounded_chunks() {
        let payload_bytes = SCORE_VALIDATION_CHUNK_BYTES as u64 * 2 + 4;
        let mut reader = TrackingReader {
            bytes: vec![0; payload_bytes as usize],
            offset: 0,
            largest_request: 0,
        };
        let mut hasher = Sha256::new();
        let mut values = 0_u64;
        let mut consume = |_: f32| -> Result<(), SegmentationBundleError> {
            values += 1;
            Ok(())
        };

        scan_score_payload(&mut reader, payload_bytes, &mut hasher, &mut consume).unwrap();

        assert_eq!(values, payload_bytes / 4);
        assert_eq!(reader.offset, payload_bytes as usize);
        assert_eq!(reader.largest_request, SCORE_VALIDATION_CHUNK_BYTES);
    }
}
