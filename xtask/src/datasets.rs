mod aishell4;
mod alimeeting;
mod ami;
mod earnings21;
mod voxconverse;

use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;

pub trait Dataset {
    /// Unique identifier used in --dataset flag and fixtures/ directory name
    fn id(&self) -> &'static str;

    /// Human-readable name for display
    fn display_name(&self) -> &'static str;

    /// Ensure wav/ and rttm/ subdirectories are populated.
    /// Idempotent — safe to call if already downloaded
    fn ensure(&self, base_dir: &Path) -> Result<()>;

    /// Returns the base dir for this dataset
    fn dataset_dir(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(self.id())
    }
}

pub fn all_datasets() -> Vec<Box<dyn Dataset>> {
    vec![
        Box::new(voxconverse::VoxConverseDev),
        Box::new(voxconverse::VoxConverseTest),
        Box::new(ami::AmiIhm),
        Box::new(aishell4::Aishell4),
        Box::new(earnings21::Earnings21),
        Box::new(alimeeting::AliMeeting),
    ]
}

pub fn find_dataset(id: &str) -> Option<Box<dyn Dataset>> {
    all_datasets().into_iter().find(|d| d.id() == id)
}

pub fn list_dataset_ids() -> Vec<&'static str> {
    vec![
        "voxconverse-dev",
        "voxconverse-test",
        "ami-ihm",
        "aishell4",
        "earnings21",
        "alimeeting",
    ]
}
