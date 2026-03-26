use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use color_eyre::eyre::{Result, bail, ensure};

fn read_u16(bytes: &[u8], context: &str) -> Result<u16> {
    let raw: [u8; 2] = bytes
        .try_into()
        .map_err(|_| color_eyre::eyre::eyre!("{context}: expected 2 bytes"))?;
    Ok(u16::from_le_bytes(raw))
}

fn read_u32(bytes: &[u8], context: &str) -> Result<u32> {
    let raw: [u8; 4] = bytes
        .try_into()
        .map_err(|_| color_eyre::eyre::eyre!("{context}: expected 4 bytes"))?;
    Ok(u32::from_le_bytes(raw))
}

/// Load 16-bit PCM mono WAV samples as f32 in [-1.0, 1.0]
pub fn load_wav_samples(path: &str) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut riff_header = [0u8; 12];
    reader.read_exact(&mut riff_header)?;
    ensure!(&riff_header[0..4] == b"RIFF", "expected RIFF WAV");
    ensure!(&riff_header[8..12] == b"WAVE", "expected WAVE file");

    let mut sample_rate = None;
    let mut channels = None;
    let mut bits_per_sample = None;

    loop {
        let mut chunk_header = [0u8; 8];
        if reader.read_exact(&mut chunk_header).is_err() {
            break;
        }

        let chunk_id = &chunk_header[0..4];
        let chunk_size = read_u32(&chunk_header[4..8], "wav chunk size")? as usize;

        match chunk_id {
            b"fmt " => {
                let mut fmt = vec![0u8; chunk_size];
                reader.read_exact(&mut fmt)?;
                let audio_format = read_u16(&fmt[0..2], "wav fmt audio format")?;
                let chunk_channels = read_u16(&fmt[2..4], "wav fmt channels")?;
                let chunk_sample_rate = read_u32(&fmt[4..8], "wav fmt sample rate")?;
                let chunk_bits_per_sample = read_u16(&fmt[14..16], "wav fmt bits per sample")?;

                ensure!(audio_format == 1, "expected PCM WAV");
                channels = Some(chunk_channels);
                sample_rate = Some(chunk_sample_rate);
                bits_per_sample = Some(chunk_bits_per_sample);
            }
            b"data" => {
                let sample_rate = sample_rate.ok_or_else(|| {
                    color_eyre::eyre::eyre!("fmt chunk must appear before data chunk")
                })?;
                let channels =
                    channels.ok_or_else(|| color_eyre::eyre::eyre!("missing channel count"))?;
                let bits_per_sample = bits_per_sample
                    .ok_or_else(|| color_eyre::eyre::eyre!("missing bits per sample"))?;
                ensure!(channels == 1, "expected mono WAV");
                ensure!(bits_per_sample == 16, "expected 16-bit PCM WAV");

                let mut samples = Vec::with_capacity(chunk_size / 2);
                let mut remaining = chunk_size;
                let mut buffer = [0u8; 8192];

                while remaining > 0 {
                    let to_read = remaining.min(buffer.len());
                    reader.read_exact(&mut buffer[..to_read])?;
                    for bytes in buffer[..to_read].chunks_exact(2) {
                        samples.push(i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0);
                    }
                    remaining -= to_read;
                }

                if chunk_size % 2 == 1 {
                    reader.seek(SeekFrom::Current(1))?;
                }

                return Ok((samples, sample_rate));
            }
            _ => {
                reader.seek(SeekFrom::Current(chunk_size as i64))?;
            }
        }

        if chunk_size % 2 == 1 {
            reader.seek(SeekFrom::Current(1))?;
        }
    }

    bail!("no data chunk found in WAV")
}
