use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::{Result, bail};

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;

/// RAII temp directory that cleans up on drop
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!("xtask-{}", std::process::id()));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn is_youtube_url(source: &str) -> bool {
    source.starts_with("http://www.youtube.com")
        || source.starts_with("https://www.youtube.com")
        || source.starts_with("http://youtube.com")
        || source.starts_with("https://youtube.com")
        || source.starts_with("http://youtu.be")
        || source.starts_with("https://youtu.be")
}

fn is_http_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

enum AudioSource {
    YouTube,
    HttpDownload,
    LocalFile,
}

impl AudioSource {
    fn new(source: &str) -> Self {
        if is_youtube_url(source) {
            Self::YouTube
        } else if is_http_url(source) {
            Self::HttpDownload
        } else {
            Self::LocalFile
        }
    }
}

/// Prepare audio from a URL, YouTube link, or local file path into 16kHz mono WAV
///
/// Returns the path to the WAV file and a TempDir guard (drop to clean up)
pub fn prepare_audio(source: &str) -> Result<(PathBuf, TempDir)> {
    let tmp = TempDir::new()?;
    let wav = tmp.path().join("audio.wav");

    println!("=== Preparing audio ===");

    match AudioSource::new(source) {
        AudioSource::YouTube => {
            run_cmd(
                Command::new("yt-dlp")
                    .args(["-x", "--audio-format", "wav"])
                    .arg("--postprocessor-args")
                    .arg("ffmpeg:-ar 16000 -ac 1")
                    .arg("-o")
                    .arg(tmp.path().join("audio.%(ext)s"))
                    .arg(source),
            )?;
        }
        AudioSource::HttpDownload => {
            let input = tmp.path().join("input");
            run_cmd(
                Command::new("curl")
                    .args(["--fail", "--location", "--silent", "--show-error"])
                    .arg(source)
                    .arg("-o")
                    .arg(&input),
            )?;
            convert_to_16k_mono(&input, &wav)?;
        }
        AudioSource::LocalFile => {
            if !Path::new(source).is_file() {
                bail!("Input does not exist: {source}");
            }
            convert_to_16k_mono(Path::new(source), &wav)?;
        }
    }

    Ok((wav, tmp))
}
