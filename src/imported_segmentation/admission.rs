//! Filesystem, NPY, and score admission for immutable bundles

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use sha2::{Digest, Sha256};

use super::canonical::parse_manifest;
use super::validation::{invalid, validate_member_path};
use super::*;

/// Loads and validates a published bundle without loading a segmentation
/// model
pub fn load_imported_segmentation_bundle(
    root: impl AsRef<Path>,
) -> Result<ImportedSegmentationBundle, SegmentationBundleError> {
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

    let mut shards = Vec::with_capacity(manifest.tensors.shards.len());
    for declaration in &manifest.tensors.shards {
        let path = checked_member_path(&root, &declaration.path)?;
        ensure_read_only(&path, "tensor shard")?;
        let (shape, values) = read_npy_shard(&path, declaration)?;
        if shape != declaration.shape {
            return Err(SegmentationBundleError::Invalid(format!(
                "shard {} shape does not match its inventory",
                declaration.path
            )));
        }
        validate_score_values(
            &values,
            manifest.head.score_representation,
            declaration.shape[2] as usize,
        )?;
        shards.push(LoadedTensorShard {
            path: declaration.path.clone(),
            chunk_start: declaration.chunk_start,
            chunk_end: declaration.chunk_end,
            shape,
            values,
        });
    }

    Ok(ImportedSegmentationBundle {
        root,
        manifest,
        shards,
    })
}

impl ImportedSegmentationBundle {
    /// Opens and validates a published bundle directory
    pub fn open(root: impl AsRef<Path>) -> Result<Self, SegmentationBundleError> {
        load_imported_segmentation_bundle(root)
    }
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
) -> Result<([u64; 3], Vec<f32>), SegmentationBundleError> {
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
    let elements = parsed
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| SegmentationBundleError::Invalid("NPY element count overflow".to_owned()))?;
    if elements > MAX_TENSOR_ELEMENTS {
        return invalid("NPY element count exceeds the admission bound".to_owned());
    }
    let payload_bytes = elements
        .checked_mul(4)
        .ok_or_else(|| SegmentationBundleError::Invalid("NPY byte count overflow".to_owned()))?;
    let expected_bytes = (data_start as u64)
        .checked_add(payload_bytes)
        .ok_or_else(|| SegmentationBundleError::Invalid("NPY byte count overflow".to_owned()))?;
    if declaration.bytes != expected_bytes {
        return invalid("NPY payload is truncated or has trailing bytes".to_owned());
    }
    let capacity = usize::try_from(elements).map_err(|_| {
        SegmentationBundleError::Invalid("NPY element count is too large".to_owned())
    })?;
    let mut values = Vec::with_capacity(capacity);
    let mut chunk = [0_u8; 4];
    for _ in 0..elements {
        file.read_exact(&mut chunk)?;
        hasher.update(chunk);
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
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
    Ok((parsed, values))
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

fn validate_score_values(
    values: &[f32],
    representation: ScoreRepresentation,
    classes: usize,
) -> Result<(), SegmentationBundleError> {
    if classes == 0 || !values.len().is_multiple_of(classes) {
        return invalid("score tensor class extent is invalid".to_owned());
    }
    for row in values.chunks_exact(classes) {
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
    }
    Ok(())
}
