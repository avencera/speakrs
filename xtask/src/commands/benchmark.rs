use std::path::Path;

use color_eyre::eyre::{Result, bail};

mod der;
mod jobs;
mod report;
mod run_store;
mod runner;
mod selection;
mod types;

pub use der::{DerArgs, der};
pub use run_store::{BenchmarkRun, convert_legacy_results};

pub fn score(run_dir: &Path) -> Result<()> {
    let results = run_dir.join("results.json");
    if !results.exists() {
        bail!(
            "schema version 2 results.json not found in {}",
            run_dir.display()
        );
    }
    let payload: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&results)?)?;
    let version = payload
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    if version != u64::from(run_store::SCHEMA_VERSION) {
        bail!(
            "{} is schema version {version}; score requires version {}",
            results.display(),
            run_store::SCHEMA_VERSION
        );
    }
    println!(
        "Scored schema version {version} run at {}",
        run_dir.display()
    );
    Ok(())
}

pub use jobs::{
    BenchmarkJobConfig, BenchmarkJobResult, GpuBenchmarkSuiteConfig, ProgressUpdate, gpu_impls,
    run_benchmark_job, run_gpu_benchmark_suite, run_speakrs_gpu, validate_gpu_impls,
};
pub use report::{DerResultsWriter, format_eta, now_stamp};
pub(crate) use selection::discover_files;
pub(crate) use types::{BatchCommandRunner, PREFLIGHT_TIMEOUT, PyannoteRsFileRunner};
pub use types::{
    BenchmarkMetadata, DerAccumulation, DerImplResult, DerImplStatus, ImplType, PerFileDerResult,
    PyannoteBatchSizes,
};

#[cfg(test)]
mod score_tests {
    use super::score;

    #[test]
    fn score_requires_schema_version_2() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("results.json"), r#"{"schema_version":1}"#).unwrap();
        assert!(score(dir.path()).is_err());
        std::fs::write(dir.path().join("results.json"), r#"{"schema_version":2}"#).unwrap();
        assert!(score(dir.path()).is_ok());
    }
}
