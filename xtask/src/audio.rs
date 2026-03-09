use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::{Result, bail};

use crate::cmd::run_cmd;

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

/// Prepare audio from a URL, YouTube link, or local file path into 16kHz mono WAV
///
/// Returns the path to the WAV file and a TempDir guard (drop to clean up)
pub fn prepare_audio(source: &str) -> Result<(PathBuf, TempDir)> {
    let tmp = TempDir::new()?;
    let wav = tmp.path().join("audio.wav");

    println!("=== Preparing audio ===");

    if is_http_url(source) {
        if is_youtube_url(source) {
            run_cmd(
                Command::new("yt-dlp")
                    .args(["-x", "--audio-format", "wav"])
                    .arg("--postprocessor-args")
                    .arg("ffmpeg:-ar 16000 -ac 1")
                    .arg("-o")
                    .arg(tmp.path().join("audio.%(ext)s"))
                    .arg(source),
            )?;
        } else {
            let input = tmp.path().join("input");
            run_cmd(
                Command::new("curl")
                    .args(["--fail", "--location", "--silent", "--show-error"])
                    .arg(source)
                    .arg("-o")
                    .arg(&input),
            )?;
            convert_to_wav(&input, &wav)?;
        }
    } else {
        if !Path::new(source).is_file() {
            bail!("Input does not exist: {source}");
        }
        convert_to_wav(Path::new(source), &wav)?;
    }

    Ok((wav, tmp))
}

fn convert_to_wav(input: &Path, output: &Path) -> Result<()> {
    run_cmd(
        Command::new("ffmpeg")
            .args(["-y", "-i"])
            .arg(input)
            .args(["-ar", "16000", "-ac", "1"])
            .arg(output)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null()),
    )
}
