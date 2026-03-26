use std::collections::HashMap;
use std::path::PathBuf;

use super::*;

mod gpu;
mod preflight;
mod run;

pub use gpu::{gpu_impls, validate_gpu_impls};
pub use run::{run_benchmark_job, run_gpu_benchmark_suite, run_speakrs_gpu};

pub struct GpuBenchmarkSuiteConfig {
    pub dataset: String,
    pub impls: Vec<String>,
    pub max_files: u32,
    pub max_minutes: u32,
    pub description: Option<String>,
    pub no_preflight: bool,
    pub models_dir: PathBuf,
    pub datasets_dir: PathBuf,
    pub root: PathBuf,
    pub results_dir: PathBuf,
    pub pyannote_batch_sizes: PyannoteBatchSizes,
}

pub struct ProgressUpdate {
    pub impl_name: String,
    pub file_index: u32,
    pub total_files: u32,
    pub file_id: String,
    pub elapsed_secs: f64,
}

pub struct BenchmarkJobConfig {
    pub models_dir: PathBuf,
    pub datasets_dir: PathBuf,
    pub root: PathBuf,
    pub results_dir: PathBuf,
    pub dataset: crate::datasets::Dataset,
    pub implementations: Vec<(&'static str, ImplType)>,
    pub max_files: u32,
    pub max_minutes: u32,
    pub description: Option<String>,
    pub multi_dataset: bool,
    pub pyannote_batch_sizes: PyannoteBatchSizes,
}

pub struct BenchmarkJobResult {
    pub run_id: String,
    pub run_dir: PathBuf,
    pub total_audio_minutes: f64,
    pub results: HashMap<String, DerImplResult>,
}
