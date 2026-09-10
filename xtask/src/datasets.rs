mod aishell4;
mod alimeeting;
mod ami;
mod catalog;
mod earnings21;
mod voxconverse;

pub use catalog::{
    DATASET_MANIFEST_FILE, DatasetCatalog, DatasetFile, DatasetId, DatasetSnapshot, DatasetSpec,
    select_alimeeting_far_field,
};

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{Result, bail, eyre};

use crate::cmd::run_cmd;

#[derive(Clone)]
pub struct Dataset {
    pub id: String,
    pub display_name: String,
    source: Source,
}

#[derive(Clone)]
enum Source {
    VoxConverseDev,
    VoxConverseTest,
    AmiIhm,
    AmiSdm,
    Aishell4,
    Earnings21,
    AliMeeting,
    Hf {
        repo: String,
    },
    #[cfg(test)]
    Local {
        root: PathBuf,
        provenance: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InstallationState {
    Missing,
    Unmarked,
    Published,
}

impl Dataset {
    fn new(id: &str, display_name: &str, source: Source) -> Self {
        Self {
            id: id.to_string(),
            display_name: display_name.to_string(),
            source,
        }
    }

    /// Return the final installation directory for this dataset
    pub fn dataset_dir(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(&self.id)
    }

    /// Ensure a complete, validated dataset installation exists
    pub fn ensure(&self, base_dir: &Path) -> Result<()> {
        fs::create_dir_all(base_dir)?;
        let dataset = self.catalog_id()?;
        let _lock = InstallationLock::acquire_exclusive(base_dir, self.id.as_str())?;
        recover_interrupted_installation(base_dir, &self.id, dataset)?;
        remove_stale_staging(base_dir, &self.id)?;

        match self.installation_state(base_dir) {
            InstallationState::Published => return Ok(()),
            InstallationState::Missing | InstallationState::Unmarked => {}
        }

        if self.source.try_tigris() {
            let staging = create_staging_directory(base_dir, &self.id)?;
            let result = S5cmd::try_download(&self.id, &staging).and_then(|downloaded| {
                if !downloaded {
                    return Ok(false);
                }
                self.publish_staged(
                    dataset,
                    &staging,
                    &self.source.tigris_provenance(&self.id),
                    base_dir,
                    true,
                )?;
                Ok(true)
            });
            match result {
                Ok(true) => return Ok(()),
                Ok(false) => remove_staging_directory(&staging)?,
                Err(error) => {
                    eprintln!(
                        "{}: Tigris acquisition was not published ({error}); trying direct source",
                        self.id
                    );
                    remove_staging_directory(&staging)?;
                }
            }
        }

        if matches!(self.source, Source::AmiSdm) {
            let ihm = find_dataset("ami-ihm")
                .ok_or_else(|| eyre!("AMI IHM is missing from the dataset catalog"))?;
            ihm.ensure(base_dir)?;
        }

        // keep the IHM inventory stable while SDM copies its RTTM source
        let _ihm_lock = matches!(self.source, Source::AmiSdm)
            .then(|| InstallationLock::acquire_shared(base_dir, "ami-ihm"))
            .transpose()?;

        let staging = create_staging_directory(base_dir, &self.id)?;
        let provenance = self.source.provenance(&self.id);
        let result = self.acquire(&staging, base_dir);
        if let Err(error) = result {
            if let Err(cleanup_error) = remove_staging_directory(&staging) {
                return Err(eyre!(
                    "dataset {} acquisition failed ({error}); staging cleanup failed ({cleanup_error})",
                    self.id
                ));
            }
            return Err(error);
        }
        let publish_result = self.publish_staged(dataset, &staging, &provenance, base_dir, false);
        if let Err(error) = publish_result {
            if let Err(cleanup_error) = remove_staging_directory(&staging) {
                return Err(eyre!(
                    "dataset {} publication failed ({error}); staging cleanup failed ({cleanup_error})",
                    self.id
                ));
            }
            return Err(error);
        }
        Ok(())
    }

    /// Resolve this dataset's canonical catalog identity
    pub fn catalog_id(&self) -> Result<DatasetId> {
        DatasetId::parse_cli(&self.id)
            .ok_or_else(|| eyre!("dataset {} is not in the catalog", self.id))
    }

    /// Read a complete validated snapshot while coordinating with installation
    pub fn snapshot(&self, base_dir: &Path) -> Result<DatasetSnapshot> {
        let dataset = self.catalog_id()?;
        let _lock = InstallationLock::acquire_shared(base_dir, &self.id)?;
        DatasetSnapshot::from_paired_directory(dataset, &self.dataset_dir(base_dir))
    }

    fn snapshot_unlocked(&self, base_dir: &Path) -> Result<DatasetSnapshot> {
        DatasetSnapshot::from_paired_directory(self.catalog_id()?, &self.dataset_dir(base_dir))
    }

    fn installation_state(&self, base_dir: &Path) -> InstallationState {
        let dir = self.dataset_dir(base_dir);
        if !dir.exists() {
            return InstallationState::Missing;
        }
        if self.snapshot_unlocked(base_dir).is_ok() {
            InstallationState::Published
        } else {
            InstallationState::Unmarked
        }
    }

    fn acquire(&self, staging: &Path, base_dir: &Path) -> Result<()> {
        match &self.source {
            Source::VoxConverseDev => voxconverse::ensure_dev(staging, base_dir),
            Source::VoxConverseTest => voxconverse::ensure_test(staging),
            Source::AmiIhm => ami::ensure_ihm(staging),
            Source::AmiSdm => ami::ensure_sdm(staging, base_dir),
            Source::Aishell4 => aishell4::ensure(staging),
            Source::Earnings21 => earnings21::ensure(staging),
            Source::AliMeeting => alimeeting::ensure(staging),
            Source::Hf { repo } => ensure_hf(&self.display_name, repo, staging),
            #[cfg(test)]
            Source::Local { root, .. } => copy_directory_contents(root, staging),
        }
    }

    fn publish_staged(
        &self,
        dataset: DatasetId,
        staging: &Path,
        provenance: &str,
        base_dir: &Path,
        require_source_manifest: bool,
    ) -> Result<()> {
        validate_source_manifest(dataset, staging, require_source_manifest)?;
        let snapshot = DatasetSnapshot::validate_staged(dataset, staging, provenance)?;
        snapshot.write_manifest(staging)?;
        // re-read the staged manifest so publication only receives a complete checked directory
        let _ = DatasetSnapshot::from_paired_directory(dataset, staging)?;
        publish_staging_directory(staging, &self.dataset_dir(base_dir), &self.id)
    }
}

impl Source {
    fn try_tigris(&self) -> bool {
        if matches!(self, Self::VoxConverseDev) {
            return false;
        }
        #[cfg(test)]
        if matches!(self, Self::Local { .. }) {
            return false;
        }
        true
    }

    fn provenance(&self, dataset_id: &str) -> String {
        let provenance = match self {
            Self::VoxConverseDev => {
                "voxconverse-dev: https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip; https://github.com/joonson/voxconverse".to_string()
            }
            Self::VoxConverseTest => {
                "voxconverse-test: https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip; https://github.com/joonson/voxconverse".to_string()
            }
            Self::AmiIhm => {
                "ami-ihm: https://github.com/BUTSpeechFIT/AMI-diarization-setup; https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus (Mix-Headset)".to_string()
            }
            Self::AmiSdm => {
                "ami-sdm: AMI IHM RTTMs; https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus (Array1-01)".to_string()
            }
            Self::Aishell4 => {
                "aishell4: https://openslr.trmal.net/resources/111/test.tar.gz".to_string()
            }
            Self::Earnings21 => {
                "earnings21: https://github.com/revdotcom/speech-datasets (earnings21)".to_string()
            }
            Self::AliMeeting => "alimeeting: https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz; selection=lexicographically-first matching far-field recording; channel=0".to_string(),
            Self::Hf { repo } => format!("huggingface dataset {repo}; split=test extraction"),
            #[cfg(test)]
            Self::Local { provenance, .. } => provenance.clone(),
        };
        format!("{} (dataset={dataset_id})", provenance.trim())
    }

    fn tigris_provenance(&self, dataset_id: &str) -> String {
        let endpoint = std::env::var("S3_ENDPOINT_URL")
            .or_else(|_| std::env::var("AWS_ENDPOINT_URL"))
            .unwrap_or_else(|_| TIGRIS_ENDPOINT.to_string());
        format!(
            "tigris dataset s3://speakrs/datasets/{dataset_id}; endpoint={endpoint}; source={}",
            self.provenance(dataset_id)
        )
    }
}

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

struct InstallationLock {
    file: File,
}

impl InstallationLock {
    fn acquire_exclusive(base_dir: &Path, dataset_id: &str) -> Result<Self> {
        Self::acquire(base_dir, dataset_id, true)
    }

    fn acquire_shared(base_dir: &Path, dataset_id: &str) -> Result<Self> {
        Self::acquire(base_dir, dataset_id, false)
    }

    fn acquire(base_dir: &Path, dataset_id: &str, exclusive: bool) -> Result<Self> {
        let path = installation_lock_path(base_dir, dataset_id);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        if exclusive {
            match file.try_lock() {
                Ok(()) => {}
                Err(std::fs::TryLockError::WouldBlock) => {
                    bail!("dataset {dataset_id} installation is already in progress")
                }
                Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
            }
        } else {
            file.lock_shared()?;
        }
        Ok(Self { file })
    }
}

impl Drop for InstallationLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn installation_lock_path(base_dir: &Path, dataset_id: &str) -> PathBuf {
    base_dir.join(format!(".{dataset_id}.install.lock"))
}

fn create_staging_directory(base_dir: &Path, dataset_id: &str) -> Result<PathBuf> {
    for _ in 0..100 {
        let serial = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path = base_dir.join(format!(
            ".{dataset_id}.staging-{}-{timestamp}-{serial}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    bail!("could not allocate a unique staging directory for {dataset_id}")
}

fn remove_staging_directory(path: &Path) -> Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path)?;
    } else if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn remove_stale_staging(base_dir: &Path, dataset_id: &str) -> Result<()> {
    let prefix = format!(".{dataset_id}.staging-");
    for entry in fs::read_dir(base_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(&prefix) && entry.path().is_dir() {
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

fn recover_interrupted_installation(
    base_dir: &Path,
    dataset_id: &str,
    dataset: DatasetId,
) -> Result<()> {
    let final_dir = base_dir.join(dataset_id);
    let prefix = format!(".{dataset_id}.previous-");
    let mut previous = Vec::new();
    for entry in fs::read_dir(base_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(&prefix) && entry.path().is_dir() {
            previous.push(entry.path());
        }
    }
    previous.sort();

    if !final_dir.exists() {
        while let Some(recovery) = previous.pop() {
            if DatasetSnapshot::from_paired_directory(dataset, &recovery).is_ok() {
                fs::rename(&recovery, &final_dir)?;
                break;
            }
            remove_staging_directory(&recovery)?;
        }
    } else if DatasetSnapshot::from_paired_directory(dataset, &final_dir).is_err() {
        while let Some(recovery) = previous.pop() {
            if DatasetSnapshot::from_paired_directory(dataset, &recovery).is_err() {
                remove_staging_directory(&recovery)?;
                continue;
            }

            let invalid = unique_recovery_path(base_dir, dataset_id);
            fs::rename(&final_dir, &invalid)?;
            if let Err(error) = fs::rename(&recovery, &final_dir) {
                let _ = fs::rename(&invalid, &final_dir);
                return Err(error.into());
            }
            remove_staging_directory(&invalid)?;
            break;
        }
    }
    for stale in previous {
        let _ = fs::remove_dir_all(stale);
    }
    Ok(())
}

fn unique_recovery_path(base_dir: &Path, dataset_id: &str) -> PathBuf {
    let serial = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    base_dir.join(format!(
        ".{dataset_id}.recovery-{}-{timestamp}-{serial}",
        std::process::id()
    ))
}

fn validate_source_manifest(dataset: DatasetId, staging: &Path, required: bool) -> Result<()> {
    let manifest_path = staging.join(DATASET_MANIFEST_FILE);
    if !manifest_path.exists() {
        if required {
            bail!(
                "dataset {} source acquisition has no completion manifest at {}",
                dataset.as_str(),
                manifest_path.display()
            );
        }
        return Ok(());
    }
    DatasetSnapshot::from_paired_directory(dataset, staging).map_err(|error| {
        eyre!(
            "dataset {} source completion manifest is invalid at {}: {error}",
            dataset.as_str(),
            manifest_path.display()
        )
    })?;
    Ok(())
}

fn publish_staging_directory(staging: &Path, final_dir: &Path, dataset_id: &str) -> Result<()> {
    let base_dir = final_dir.parent().ok_or_else(|| {
        eyre!(
            "dataset final directory has no parent: {}",
            final_dir.display()
        )
    })?;
    let previous = base_dir.join(format!(
        ".{dataset_id}.previous-{}-{}",
        std::process::id(),
        STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let had_previous = final_dir.exists();
    if had_previous {
        fs::rename(final_dir, &previous)?;
    }

    if let Err(error) = fs::rename(staging, final_dir) {
        if had_previous && let Err(restore_error) = fs::rename(&previous, final_dir) {
            return Err(eyre!(
                "publishing dataset {} failed ({error}); restoring previous installation also failed ({restore_error}); recovery directory: {}",
                dataset_id,
                previous.display()
            ));
        }
        return Err(error.into());
    }

    if had_previous && let Err(error) = fs::remove_dir_all(&previous) {
        eprintln!(
            "dataset {dataset_id} published, but stale previous installation could not be removed: {error}"
        );
    }
    Ok(())
}

#[cfg(test)]
fn copy_directory_contents(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            fs::create_dir_all(&destination_path)?;
            copy_directory_contents(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, destination_path)?;
        }
    }
    Ok(())
}

pub fn all_datasets() -> Vec<Dataset> {
    vec![
        Dataset::new("voxconverse-dev", "VoxConverse Dev", Source::VoxConverseDev),
        Dataset::new(
            "voxconverse-test",
            "VoxConverse Test",
            Source::VoxConverseTest,
        ),
        Dataset::new("ami-ihm", "AMI IHM", Source::AmiIhm),
        Dataset::new("ami-sdm", "AMI SDM", Source::AmiSdm),
        Dataset::new("aishell4", "AISHELL-4", Source::Aishell4),
        Dataset::new("earnings21", "Earnings-21", Source::Earnings21),
        Dataset::new("alimeeting", "AliMeeting", Source::AliMeeting),
        Dataset::new(
            "ava-avd",
            "AVA-AVD",
            Source::Hf {
                repo: "argmaxinc/ava-avd".into(),
            },
        ),
        Dataset::new(
            "icsi",
            "ICSI",
            Source::Hf {
                repo: "argmaxinc/icsi-meetings".into(),
            },
        ),
    ]
}

pub fn find_dataset(id: &str) -> Option<Dataset> {
    let canonical = DatasetCatalog::parse_cli(id)?.id.as_str();
    all_datasets().into_iter().find(|d| d.id == canonical)
}

pub fn list_dataset_ids() -> Vec<String> {
    DatasetCatalog::all()
        .iter()
        .map(|spec| spec.id.as_str().to_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// s5cmd -- parallel S3 downloads from Tigris
// ---------------------------------------------------------------------------

const S3_BUCKET: &str = "s3://speakrs/datasets";
const S3_BENCHMARKS: &str = "s3://speakrs/benchmarks";
const TIGRIS_ENDPOINT: &str = "https://t3.storage.dev";

pub struct S5cmd;

impl S5cmd {
    /// Check if s5cmd binary is installed and S3 credentials are configured
    pub fn available() -> bool {
        Command::new("s5cmd")
            .arg("version")
            .output()
            .is_ok_and(|o| o.status.success())
            && std::env::var("AWS_ACCESS_KEY_ID").is_ok()
    }

    fn base_cmd() -> Command {
        let mut cmd = Command::new("s5cmd");
        let endpoint = std::env::var("S3_ENDPOINT_URL")
            .or_else(|_| std::env::var("AWS_ENDPOINT_URL"))
            .unwrap_or_else(|_| TIGRIS_ENDPOINT.to_string());
        cmd.arg("--endpoint-url").arg(endpoint);
        cmd
    }

    /// Copy a dataset from Tigris S3 to a local directory
    pub fn sync(dataset_id: &str, local_dir: &Path) -> Result<()> {
        let s3_path = format!("{S3_BUCKET}/{dataset_id}/*");
        fs::create_dir_all(local_dir)?;
        run_cmd(
            Self::base_cmd()
                .args(["cp", "--concurrency", "20", "--part-size", "25"])
                .arg(&s3_path)
                .arg(local_dir),
        )
    }

    /// Upload a local dataset directory to Tigris S3
    pub fn upload(dataset_id: &str, local_dir: &Path) -> Result<()> {
        let s3_path = format!("{S3_BUCKET}/{dataset_id}/");
        run_cmd(
            Self::base_cmd()
                .args(["sync", "--concurrency", "20", "--part-size", "25"])
                .args(["--exclude", "*__MACOSX*", "--exclude", "*.DS_Store"])
                .arg(format!("{}/", local_dir.display()))
                .arg(&s3_path),
        )
    }

    /// Upload a benchmark run directory to S3
    pub fn upload_benchmarks(run_id: &str, run_dir: &Path) -> Result<()> {
        let s3_path = format!("{S3_BENCHMARKS}/{run_id}/");
        run_cmd(
            Self::base_cmd()
                .args(["sync", "--concurrency", "20", "--part-size", "25"])
                .arg(format!("{}/*", run_dir.display()))
                .arg(&s3_path),
        )
    }

    /// Verify that an S3 upload contains the expected number of files
    pub fn verify_upload(run_id: &str, expected_count: usize) -> Result<bool> {
        let s3_path = format!("{S3_BENCHMARKS}/{run_id}/");
        let output = Self::base_cmd()
            .args(["ls", &s3_path])
            .output()
            .map_err(|e| color_eyre::eyre::eyre!("s5cmd ls failed: {e}"))?;

        if !output.status.success() {
            return Ok(false);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let remote_count = stdout.lines().filter(|l| !l.trim().is_empty()).count();
        Ok(remote_count >= expected_count)
    }

    /// Download benchmark results from S3
    pub fn download_benchmarks(run_id: &str, local_dir: &Path) -> Result<()> {
        let s3_path = format!("{S3_BENCHMARKS}/{run_id}/*");
        fs::create_dir_all(local_dir)?;
        run_cmd(
            Self::base_cmd()
                .args(["cp", "--concurrency", "20", "--part-size", "25"])
                .arg(&s3_path)
                .arg(local_dir),
        )
    }

    /// Try s5cmd download into a staging directory
    fn try_download(dataset_id: &str, staging_dir: &Path) -> Result<bool> {
        if !Self::available() {
            return Ok(false);
        }

        println!("=== Downloading {dataset_id} via s5cmd from Tigris ===");
        match Self::sync(dataset_id, staging_dir) {
            Ok(()) => Ok(true),
            Err(e) => {
                println!("{dataset_id}: s5cmd failed ({e}), falling back to direct download");
                Ok(false)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// hf download fallback for Tigris-sourced datasets
// ---------------------------------------------------------------------------

fn hf_download(repo: &str, local_dir: &Path) -> Result<()> {
    fs::create_dir_all(local_dir)?;
    run_cmd(
        Command::new("uv")
            .args(["tool", "run", "--from", "huggingface-hub", "hf", "download"])
            .arg(repo)
            .args(["--repo-type", "dataset", "--local-dir"])
            .arg(local_dir),
    )
}

fn ensure_hf(display_name: &str, repo: &str, dir: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");
    println!("=== Downloading {display_name} from HuggingFace ===");
    let tmp_name = repo.replace('/', "-");
    let tmp_dir = dir.join(format!(
        ".{tmp_name}-hf-{}-{}",
        std::process::id(),
        STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&tmp_dir);
    if let Err(error) = hf_download(repo, &tmp_dir) {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Err(error);
    }

    // hf datasets use parquet format with embedded audio
    let parquet_dir = tmp_dir.join("data");
    let extract_script = crate::cmd::project_root().join("scripts/extract_hf_dataset.py");

    println!("Extracting parquet to wav + rttm...");
    let extraction_result = run_cmd(
        Command::new("uv")
            .args(["run", "--script"])
            .arg(&extract_script)
            .arg(&parquet_dir)
            .arg(&wav_dir)
            .arg(&rttm_dir)
            .args(["--split", "test"]),
    );

    let _ = fs::remove_dir_all(&tmp_dir);
    extraction_result?;
    println!("{display_name} setup complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use std::sync::Arc;
    use std::thread;

    use hound::{SampleFormat, WavSpec, WavWriter};
    use tempfile::TempDir;

    #[test]
    fn local_source_publishes_manifest_and_checked_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        write_fake_dataset(&source_dir, "sample", valid_rttm("sample"));
        let install_dir = temp_dir.path().join("installed");
        fs::create_dir_all(&install_dir).unwrap();
        fs::create_dir(install_dir.join(".icsi.staging-stale")).unwrap();

        let dataset = local_dataset(&source_dir);
        dataset.ensure(&install_dir).unwrap();

        let final_dir = dataset.dataset_dir(&install_dir);
        assert!(final_dir.join(DATASET_MANIFEST_FILE).is_file());
        let snapshot = dataset.snapshot(&install_dir).unwrap();
        assert_eq!(snapshot.files().len(), 1);
        assert_eq!(snapshot.files()[0].file_id(), "sample");
        assert!(snapshot.source_provenance().contains("fake-source"));
    }

    #[test]
    fn snapshot_rejects_changed_published_files() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        write_fake_dataset(&source_dir, "sample", valid_rttm("sample"));
        let install_dir = temp_dir.path().join("installed");
        let dataset = local_dataset(&source_dir);
        dataset.ensure(&install_dir).unwrap();

        fs::OpenOptions::new()
            .append(true)
            .open(dataset.dataset_dir(&install_dir).join("wav/sample.wav"))
            .unwrap()
            .write_all(b"changed")
            .unwrap();
        let error = dataset.snapshot(&install_dir).unwrap_err().to_string();
        assert!(error.contains("WAV changed"), "{error}");
    }

    #[test]
    fn interrupted_source_does_not_publish_partial_installation() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        fs::create_dir_all(source_dir.join("wav")).unwrap();
        fs::create_dir_all(source_dir.join("rttm")).unwrap();
        write_wav(&source_dir.join("wav/sample.wav"));
        fs::write(source_dir.join("rttm/sample.rttm"), b"not RTTM\n").unwrap();
        let install_dir = temp_dir.path().join("installed");

        let dataset = local_dataset(&source_dir);
        let error = dataset.ensure(&install_dir).unwrap_err().to_string();
        assert!(error.contains("RTTM") || error.contains("rttm"), "{error}");
        assert!(!dataset.dataset_dir(&install_dir).exists());
        assert!(
            staging_directories(&install_dir, &dataset.id)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn failed_replacement_leaves_previous_unmarked_installation_untouched() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        write_fake_dataset(&source_dir, "sample", valid_rttm("sample"));
        let install_dir = temp_dir.path().join("installed");
        let dataset = local_dataset(&source_dir);
        fs::create_dir_all(dataset.dataset_dir(&install_dir).join("wav")).unwrap();
        fs::create_dir_all(dataset.dataset_dir(&install_dir).join("rttm")).unwrap();
        write_wav(&dataset.dataset_dir(&install_dir).join("wav/sample.wav"));
        fs::write(
            dataset.dataset_dir(&install_dir).join("rttm/sample.rttm"),
            valid_rttm("sample"),
        )
        .unwrap();
        let old_wav = fs::read(dataset.dataset_dir(&install_dir).join("wav/sample.wav")).unwrap();

        fs::write(source_dir.join("rttm/sample.rttm"), b"broken\n").unwrap();
        assert!(dataset.ensure(&install_dir).is_err());

        let current_wav =
            fs::read(dataset.dataset_dir(&install_dir).join("wav/sample.wav")).unwrap();
        assert_eq!(current_wav, old_wav);
        assert!(
            !dataset
                .dataset_dir(&install_dir)
                .join(DATASET_MANIFEST_FILE)
                .exists()
        );
    }

    #[test]
    fn one_pair_without_completion_manifest_is_not_accepted() {
        let temp_dir = TempDir::new().unwrap();
        let install_dir = temp_dir.path().join("installed");
        let dataset = local_dataset(temp_dir.path());
        fs::create_dir_all(dataset.dataset_dir(&install_dir).join("wav")).unwrap();
        fs::create_dir_all(dataset.dataset_dir(&install_dir).join("rttm")).unwrap();
        write_wav(&dataset.dataset_dir(&install_dir).join("wav/sample.wav"));
        fs::write(
            dataset.dataset_dir(&install_dir).join("rttm/sample.rttm"),
            valid_rttm("sample"),
        )
        .unwrap();

        assert!(dataset.snapshot(&install_dir).is_err());
    }

    #[test]
    fn existing_unmarked_installation_is_republished_through_staging() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        write_fake_dataset(&source_dir, "sample", valid_rttm("sample"));
        let install_dir = temp_dir.path().join("installed");
        let dataset = local_dataset(&source_dir);
        let final_dir = dataset.dataset_dir(&install_dir);
        fs::create_dir_all(final_dir.join("wav")).unwrap();
        fs::create_dir_all(final_dir.join("rttm")).unwrap();
        write_wav(&final_dir.join("wav/sample.wav"));
        fs::write(final_dir.join("rttm/sample.rttm"), valid_rttm("sample")).unwrap();

        dataset.ensure(&install_dir).unwrap();
        assert!(final_dir.join(DATASET_MANIFEST_FILE).is_file());
        assert!(dataset.snapshot(&install_dir).is_ok());
    }

    #[test]
    fn source_completion_manifest_rejects_a_partial_copy() {
        let temp_dir = TempDir::new().unwrap();
        let complete_source = temp_dir.path().join("complete-source");
        write_fake_dataset(&complete_source, "first", valid_rttm("first"));
        write_fake_dataset(&complete_source, "second", valid_rttm("second"));

        let source_install = temp_dir.path().join("source-install");
        let source_dataset = local_dataset(&complete_source);
        source_dataset.ensure(&source_install).unwrap();

        let partial_source = temp_dir.path().join("partial-source");
        copy_directory_contents(
            &source_dataset.dataset_dir(&source_install),
            &partial_source,
        )
        .unwrap();
        fs::remove_file(partial_source.join("wav/second.wav")).unwrap();
        fs::remove_file(partial_source.join("rttm/second.rttm")).unwrap();

        let target = local_dataset(&partial_source);
        let install_dir = temp_dir.path().join("target-install");
        let error = target.ensure(&install_dir).unwrap_err().to_string();
        assert!(
            error.contains("completion manifest") || error.contains("inventory"),
            "{error}"
        );
        assert!(!target.dataset_dir(&install_dir).exists());
    }

    #[test]
    fn one_pair_legacy_voxconverse_directory_is_not_published() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("datasets");
        let legacy_dir = base_dir.join("voxconverse");
        write_fake_dataset(&legacy_dir, "sample", valid_rttm("sample"));
        let old_wav = fs::read(legacy_dir.join("wav/sample.wav")).unwrap();
        let old_rttm = fs::read(legacy_dir.join("rttm/sample.rttm")).unwrap();

        let staging = base_dir.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let dataset = Dataset::new("voxconverse-dev", "VoxConverse Dev", Source::VoxConverseDev);
        assert!(!voxconverse::migrate_verified_legacy_directory(&legacy_dir, &staging).unwrap());

        let error = dataset
            .publish_staged(
                DatasetId::VoxconverseDev,
                &staging,
                &dataset.source.provenance("voxconverse-dev"),
                &base_dir,
                false,
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("missing wav") || error.contains("installation has no"),
            "{error}"
        );
        assert!(!dataset.dataset_dir(&base_dir).exists());
        assert_eq!(
            fs::read(legacy_dir.join("wav/sample.wav")).unwrap(),
            old_wav
        );
        assert_eq!(
            fs::read(legacy_dir.join("rttm/sample.rttm")).unwrap(),
            old_rttm
        );
    }

    #[test]
    fn verified_legacy_voxconverse_directory_preserves_source_bytes() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("datasets");
        let legacy_dir = base_dir.join("voxconverse");
        write_fake_dataset(&legacy_dir, "first", valid_rttm("first"));
        write_fake_dataset(&legacy_dir, "second", valid_rttm("second"));
        let dataset = Dataset::new("voxconverse-dev", "VoxConverse Dev", Source::VoxConverseDev);
        let provenance = dataset.source.provenance("voxconverse-dev");
        let snapshot =
            DatasetSnapshot::validate_staged(DatasetId::VoxconverseDev, &legacy_dir, &provenance)
                .unwrap();
        snapshot.write_manifest(&legacy_dir).unwrap();
        let old_wav = fs::read(legacy_dir.join("wav/first.wav")).unwrap();
        let old_rttm = fs::read(legacy_dir.join("rttm/second.rttm")).unwrap();

        let staging = base_dir.join("staging");
        fs::create_dir_all(&staging).unwrap();
        assert!(voxconverse::migrate_verified_legacy_directory(&legacy_dir, &staging).unwrap());
        dataset
            .publish_staged(
                DatasetId::VoxconverseDev,
                &staging,
                &provenance,
                &base_dir,
                false,
            )
            .unwrap();

        let installed_dir = dataset.dataset_dir(&base_dir);
        assert_eq!(
            fs::read(installed_dir.join("wav/first.wav")).unwrap(),
            old_wav
        );
        assert_eq!(
            fs::read(installed_dir.join("rttm/second.rttm")).unwrap(),
            old_rttm
        );
        assert!(dataset.snapshot(&base_dir).is_ok());
        assert!(
            DatasetSnapshot::from_paired_directory(DatasetId::VoxconverseDev, &legacy_dir).is_ok()
        );
    }

    #[test]
    fn stale_published_installation_is_recovered_before_acquisition() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        write_fake_dataset(&source_dir, "sample", valid_rttm("sample"));
        let install_dir = temp_dir.path().join("installed");
        let dataset = local_dataset(&source_dir);
        dataset.ensure(&install_dir).unwrap();

        let final_dir = dataset.dataset_dir(&install_dir);
        let previous_dir = install_dir.join(".icsi.previous-stale");
        fs::rename(&final_dir, &previous_dir).unwrap();
        fs::write(source_dir.join("rttm/sample.rttm"), b"broken\n").unwrap();

        dataset.ensure(&install_dir).unwrap();
        assert!(dataset.snapshot(&install_dir).is_ok());
        assert!(!previous_dir.exists());
    }

    #[test]
    fn stale_previous_installation_replaces_an_invalid_final_directory() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        write_fake_dataset(&source_dir, "sample", valid_rttm("sample"));
        let install_dir = temp_dir.path().join("installed");
        let dataset = local_dataset(&source_dir);
        dataset.ensure(&install_dir).unwrap();

        let final_dir = dataset.dataset_dir(&install_dir);
        let previous_dir = install_dir.join(".icsi.previous-stale");
        copy_directory_contents(&final_dir, &previous_dir).unwrap();
        fs::write(final_dir.join("rttm/sample.rttm"), b"broken\n").unwrap();
        fs::write(source_dir.join("rttm/sample.rttm"), b"broken\n").unwrap();

        dataset.ensure(&install_dir).unwrap();
        assert!(dataset.snapshot(&install_dir).is_ok());
        assert!(!previous_dir.exists());
    }

    #[test]
    fn concurrent_installation_attempts_have_one_owner() {
        let temp_dir = TempDir::new().unwrap();
        let source_dir = temp_dir.path().join("source");
        write_fake_dataset(&source_dir, "sample", valid_rttm("sample"));
        let install_dir = Arc::new(temp_dir.path().join("installed"));
        let dataset = Arc::new(local_dataset(&source_dir));
        let first_dataset = Arc::clone(&dataset);
        let first_dir = Arc::clone(&install_dir);
        let first = thread::spawn(move || first_dataset.ensure(&first_dir));
        let second_dataset = Arc::clone(&dataset);
        let second_dir = Arc::clone(&install_dir);
        let second = thread::spawn(move || second_dataset.ensure(&second_dir));

        let first_result = first.join().unwrap();
        let second_result = second.join().unwrap();
        assert!(first_result.is_ok() || second_result.is_ok());
        assert!(dataset.snapshot(&install_dir).is_ok());
        assert!(
            staging_directories(&install_dir, &dataset.id)
                .unwrap()
                .is_empty()
        );
    }

    fn local_dataset(source_dir: &Path) -> Dataset {
        Dataset::new(
            "icsi",
            "ICSI",
            Source::Local {
                root: source_dir.to_owned(),
                provenance: "fake-source".to_string(),
            },
        )
    }

    fn write_fake_dataset(root: &Path, file_id: &str, rttm: String) {
        fs::create_dir_all(root.join("wav")).unwrap();
        fs::create_dir_all(root.join("rttm")).unwrap();
        write_wav(&root.join(format!("wav/{file_id}.wav")));
        fs::write(root.join(format!("rttm/{file_id}.rttm")), rttm).unwrap();
    }

    fn write_wav(path: &Path) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for _ in 0..1_600 {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn valid_rttm(file_id: &str) -> String {
        format!("SPEAKER {file_id} 1 0.000 0.050 <NA> <NA> speaker <NA> <NA>\n")
    }

    fn staging_directories(base_dir: &Path, dataset_id: &str) -> Result<Vec<PathBuf>> {
        if !base_dir.exists() {
            return Ok(Vec::new());
        }
        let prefix = format!(".{dataset_id}.staging-");
        Ok(fs::read_dir(base_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name().to_string_lossy().starts_with(&prefix) && entry.path().is_dir()
            })
            .map(|entry| entry.path())
            .collect())
    }
}
