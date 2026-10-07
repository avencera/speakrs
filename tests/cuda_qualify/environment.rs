//! Actual shared-library providers, not files found in installation directories

use std::io;
use std::path::Path;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Locate the file mapping that contains an API symbol, preserving spaces in paths
fn mapping_path(line: &str, address: usize) -> Option<&str> {
    let mut text = line;
    let mut fields = Vec::with_capacity(5);
    for _ in 0..5 {
        let (field, rest) = text.split_once(char::is_whitespace)?;
        fields.push(field);
        text = rest.trim_start();
    }
    let (start, end) = fields[0].split_once('-')?;
    let start = usize::from_str_radix(start, 16).ok()?;
    let end = usize::from_str_radix(end, 16).ok()?;
    (start <= address && address < end && fields[1].contains('x') && text.starts_with('/'))
        .then_some(text)
}

fn fingerprints(maps: &str, providers: &[(&str, usize)]) -> io::Result<Value> {
    let mut result = serde_json::Map::new();
    for &(family, address) in providers {
        let matches: Vec<_> = maps
            .lines()
            .filter_map(|line| mapping_path(line, address))
            .collect();
        if matches.len() != 1 {
            return Err(io::Error::other(format!(
                "missing or ambiguous loaded {family} provider"
            )));
        }
        let path = Path::new(matches[0]).canonicalize()?;
        if !path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with(family))
        {
            return Err(io::Error::other(format!(
                "unexpected loaded {family} provider: {}",
                path.display()
            )));
        }
        let mut file = std::fs::File::open(&path)?;
        let mut hash = Sha256::new();
        io::copy(&mut file, &mut hash)?;
        result.insert(
            path.to_string_lossy().into_owned(),
            json!(format!("{:x}", hash.finalize())),
        );
    }
    Ok(Value::Object(result))
}

/// Hash the same loaded libraries cudarc uses for the driver, cuDNN and cuBLAS APIs
pub(super) fn loaded_libraries() -> io::Result<Value> {
    // SAFETY: each library is prepared before this call and kept live by cudarc's
    // process-wide owner. The named symbols are inspected as addresses, never called
    let providers = unsafe {
        [
            (
                "libcuda.so",
                *cudarc::driver::sys::culib()
                    .get::<*const ()>(b"cuDriverGetVersion\0")
                    .map_err(io::Error::other)? as usize,
            ),
            (
                "libcudnn.so",
                *cudarc::cudnn::sys::culib()
                    .get::<*const ()>(b"cudnnGetVersion\0")
                    .map_err(io::Error::other)? as usize,
            ),
            (
                "libcublas.so",
                *cudarc::cublas::sys::culib()
                    .get::<*const ()>(b"cublasGetVersion_v2\0")
                    .map_err(io::Error::other)? as usize,
            ),
        ]
    };
    fingerprints(&std::fs::read_to_string("/proc/self/maps")?, &providers)
}

#[test]
fn symbol_provider_mapping_preserves_paths_and_rejects_non_code_or_wrong_ranges() {
    let line = "7f1000-7f2000 r-xp 00001000 00:20 77   /alternate CUDA/libcublas.so.12";
    assert_eq!(
        mapping_path(line, 0x7f1234),
        Some("/alternate CUDA/libcublas.so.12")
    );
    assert_eq!(mapping_path(line, 0x7f2000), None);
    assert_eq!(mapping_path(line, 0x7f0000), None);
    assert_eq!(mapping_path(&line.replace("r-xp", "r--p"), 0x7f1234), None);
    assert!(fingerprints(line, &[("libcublas.so", 0x7f0000)]).is_err());
    assert!(fingerprints(&format!("{line}\n{line}"), &[("libcublas.so", 0x7f1234)]).is_err());
}

#[test]
fn fingerprint_hashes_the_mapped_provider_not_an_installed_neighbor() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "speakrs-library-provider-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("libcudnn.so.9");
    std::fs::write(&path, b"actually loaded alternative bytes").unwrap();
    let maps = format!("1000-2000 r-xp 00000000 00:20 77 {}", path.display());
    let actual = fingerprints(&maps, &[("libcudnn.so", 0x1200)]).unwrap();
    let expected = format!("{:x}", Sha256::digest(b"actually loaded alternative bytes"));
    assert_eq!(
        actual[path.canonicalize().unwrap().to_string_lossy().as_ref()],
        expected
    );
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
