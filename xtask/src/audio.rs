use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::{Result, bail};
use tempfile::TempDir;

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;

pub struct PreparedAudio {
    wav_path: PathBuf,
    temp_dir: TempDir,
}

impl PreparedAudio {
    pub fn wav_path(&self) -> &Path {
        &self.wav_path
    }

    pub fn into_parts(self) -> (PathBuf, TempDir) {
        (self.wav_path, self.temp_dir)
    }
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
/// Returns the converted WAV path plus the temp dir guard that keeps it alive
pub fn prepare_audio(source: &str) -> Result<PreparedAudio> {
    let temp_dir = TempDir::new()?;
    let wav_path = temp_dir.path().join("audio.wav");

    println!("=== Preparing audio ===");

    match AudioSource::new(source) {
        AudioSource::YouTube => {
            run_cmd(
                Command::new("yt-dlp")
                    .args(["-x", "--audio-format", "wav"])
                    .arg("--postprocessor-args")
                    .arg("ffmpeg:-ar 16000 -ac 1")
                    .arg("-o")
                    .arg(temp_dir.path().join("audio.%(ext)s"))
                    .arg(source),
            )?;
        }
        AudioSource::HttpDownload => {
            let input = temp_dir.path().join("input");
            run_cmd(
                Command::new("curl")
                    .args(["--fail", "--location", "--silent", "--show-error"])
                    .arg(source)
                    .arg("-o")
                    .arg(&input),
            )?;
            convert_to_16k_mono(&input, &wav_path)?;
        }
        AudioSource::LocalFile => {
            if !Path::new(source).is_file() {
                bail!("Input does not exist: {source}");
            }
            convert_to_16k_mono(Path::new(source), &wav_path)?;
        }
    }

    Ok(PreparedAudio { wav_path, temp_dir })
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
