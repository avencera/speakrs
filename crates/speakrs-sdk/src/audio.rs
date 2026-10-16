//! Decoding audio files to the 16 kHz mono samples the pipeline takes

use std::path::Path;

use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use thiserror::Error;

/// Sample rate the diarization pipeline expects
pub const SDK_SAMPLE_RATE_HZ: u32 = 16_000;

/// Mono audio samples at a known sample rate
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Mono samples in file order
    pub samples: Vec<f32>,
    /// Sample rate of [`Self::samples`] in Hz
    pub sample_rate_hz: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct InterleavedAudio {
    samples: Vec<f32>,
    sample_rate_hz: u32,
    channels: usize,
}

/// Failure to decode, downmix, or resample an audio file
#[derive(Debug, Error)]
pub enum AudioDecodeError {
    /// The container reported a zero sample rate
    #[error("invalid sample rate {0}")]
    InvalidSampleRate(u32),

    /// A decoded packet had no channels
    #[error("invalid channel count {0}")]
    InvalidChannelCount(usize),

    /// The interleaved buffer does not hold whole frames
    #[error("audio sample count {samples} is not divisible by channel count {channels}")]
    InvalidInterleavedLength {
        /// Samples in the buffer
        samples: usize,
        /// Channels per frame
        channels: usize,
    },

    /// The file could not be opened
    #[error("failed to open audio file `{path}`: {message}")]
    Open {
        /// The audio file
        path: String,
        /// Why it failed
        message: String,
    },

    /// The container format was not recognized
    #[error("failed to probe audio file `{path}`: {message}")]
    Probe {
        /// The audio file
        path: String,
        /// Why it failed
        message: String,
    },

    /// The container has no audio track
    #[error("audio file `{path}` has no decodable audio track")]
    MissingAudioTrack {
        /// The audio file
        path: String,
    },

    /// The audio codec is not compiled in
    #[error("unsupported audio codec in `{path}`: {message}")]
    UnsupportedCodec {
        /// The audio file
        path: String,
        /// Why it failed
        message: String,
    },

    /// Reading a packet from the container failed
    #[error("failed to read audio packet from `{path}`: {message}")]
    ReadPacket {
        /// The audio file
        path: String,
        /// Why it failed
        message: String,
    },

    /// Decoding a packet failed
    #[error("failed to decode audio packet from `{path}`: {message}")]
    Decode {
        /// The audio file
        path: String,
        /// Why it failed
        message: String,
    },

    /// The file decoded to zero samples
    #[error("decoded audio file `{path}` did not contain samples")]
    EmptyAudio {
        /// The audio file
        path: String,
    },

    /// The resampler rejected the input
    #[error("failed to resample audio: {0}")]
    Resample(String),
}

/// Decode an audio file, downmix it to mono, and resample it to 16 kHz
pub fn decode_file_to_mono_16khz(path: impl AsRef<Path>) -> Result<DecodedAudio, AudioDecodeError> {
    let decoded = decode_file_to_mono(path)?;
    Ok(DecodedAudio {
        samples: resample_mono_to_16khz(decoded.samples, decoded.sample_rate_hz)?,
        sample_rate_hz: SDK_SAMPLE_RATE_HZ,
    })
}

/// Decode an audio file and downmix it to mono at its native sample rate
///
/// WAV, FLAC, MP3, Ogg Vorbis, and AAC in MP4/M4A are supported
pub fn decode_file_to_mono(path: impl AsRef<Path>) -> Result<DecodedAudio, AudioDecodeError> {
    let interleaved = decode_interleaved(path.as_ref())?;
    Ok(DecodedAudio {
        samples: downmix_interleaved_to_mono(interleaved.samples, interleaved.channels)?,
        sample_rate_hz: interleaved.sample_rate_hz,
    })
}

/// Average interleaved channels into one; mono input is returned unchanged
pub fn downmix_interleaved_to_mono(
    samples: Vec<f32>,
    channels: usize,
) -> Result<Vec<f32>, AudioDecodeError> {
    if channels == 0 {
        return Err(AudioDecodeError::InvalidChannelCount(channels));
    }

    if !samples.len().is_multiple_of(channels) {
        return Err(AudioDecodeError::InvalidInterleavedLength {
            samples: samples.len(),
            channels,
        });
    }

    if channels == 1 {
        return Ok(samples);
    }

    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect())
}

/// Resample mono audio to 16 kHz; 16 kHz input is returned unchanged
pub fn resample_mono_to_16khz(
    samples: Vec<f32>,
    sample_rate_hz: u32,
) -> Result<Vec<f32>, AudioDecodeError> {
    if sample_rate_hz == 0 {
        return Err(AudioDecodeError::InvalidSampleRate(sample_rate_hz));
    }

    if sample_rate_hz == SDK_SAMPLE_RATE_HZ || samples.is_empty() {
        return Ok(samples);
    }

    let input_len = samples.len();
    let input = InterleavedOwned::new_from(samples, 1, input_len)
        .map_err(|err| AudioDecodeError::Resample(err.to_string()))?;
    // the FFT resampler low-passes before decimating, so content above 8 kHz
    // cannot fold back into the band the models hear
    let mut resampler = Fft::<f32>::new(
        sample_rate_hz as usize,
        SDK_SAMPLE_RATE_HZ as usize,
        1024,
        1,
        FixedSync::Input,
    )
    .map_err(|err| AudioDecodeError::Resample(err.to_string()))?;
    let output_capacity = resampler.process_all_needed_output_len(input_len);
    let mut output = InterleavedOwned::new(0.0, 1, output_capacity);
    let (_, output_len) = resampler
        .process_all_into_buffer(&input, &mut output, input_len, None)
        .map_err(|err| AudioDecodeError::Resample(err.to_string()))?;

    let mut samples = output.take_data();
    samples.truncate(output_len);
    Ok(samples)
}

fn decode_interleaved(path: &Path) -> Result<InterleavedAudio, AudioDecodeError> {
    let path_display = path.display().to_string();
    let src = std::fs::File::open(path).map_err(|err| AudioDecodeError::Open {
        path: path_display.clone(),
        message: err.to_string(),
    })?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    // the extension only speeds up probing; symphonia still checks the container
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| AudioDecodeError::Probe {
            path: path_display.clone(),
            message: err.to_string(),
        })?;
    let track = format.default_track(TrackType::Audio).ok_or_else(|| {
        AudioDecodeError::MissingAudioTrack {
            path: path_display.clone(),
        }
    })?;
    let track_id = track.id;
    let codec_params =
        track
            .codec_params
            .as_ref()
            .ok_or_else(|| AudioDecodeError::MissingAudioTrack {
                path: path_display.clone(),
            })?;
    let audio_params = codec_params
        .audio()
        .ok_or_else(|| AudioDecodeError::MissingAudioTrack {
            path: path_display.clone(),
        })?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|err| AudioDecodeError::UnsupportedCodec {
            path: path_display.clone(),
            message: err.to_string(),
        })?;

    let mut samples = Vec::new();
    let mut sample_rate_hz = None;
    let mut channels = None;
    while let Some(packet) = next_packet(&mut *format, &path_display)? {
        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                let current_channels = spec.channels().count();
                if current_channels == 0 {
                    return Err(AudioDecodeError::InvalidChannelCount(current_channels));
                }

                sample_rate_hz.get_or_insert(spec.rate());
                channels.get_or_insert(current_channels);

                let start = samples.len();
                samples.resize(start + decoded.samples_interleaved(), 0.0);
                decoded.copy_to_slice_interleaved(&mut samples[start..]);
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => {
                return Err(AudioDecodeError::Decode {
                    path: path_display,
                    message: err.to_string(),
                });
            }
        }
    }

    if samples.is_empty() {
        return Err(AudioDecodeError::EmptyAudio { path: path_display });
    }

    Ok(InterleavedAudio {
        samples,
        sample_rate_hz: sample_rate_hz.unwrap_or(SDK_SAMPLE_RATE_HZ),
        channels: channels.unwrap_or(1),
    })
}

fn next_packet(
    format: &mut dyn symphonia::core::formats::FormatReader,
    path: &str,
) -> Result<Option<symphonia::core::packet::Packet>, AudioDecodeError> {
    match format.next_packet() {
        Ok(packet) => Ok(packet),
        Err(SymphoniaError::ResetRequired) => Err(AudioDecodeError::ReadPacket {
            path: path.to_string(),
            message: "stream reset required".to_string(),
        }),
        Err(SymphoniaError::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
            Ok(None)
        }
        Err(err) => Err(AudioDecodeError::ReadPacket {
            path: path.to_string(),
            message: err.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use super::*;

    #[test]
    fn resamples_mono_audio_to_sdk_rate_without_aliasing() {
        let tone = |rate: u32, hz: f32, len: usize| -> Vec<f32> {
            (0..len)
                .map(|idx| (std::f32::consts::TAU * hz * idx as f32 / rate as f32).sin())
                .collect()
        };
        // skip the filter's edges, where a tone has not fully ramped in or out
        let rms = |samples: &[f32]| {
            let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
            (middle.iter().map(|value| value * value).sum::<f32>() / middle.len() as f32).sqrt()
        };

        let kept = resample_mono_to_16khz(tone(48_000, 1_000.0, 48_000), 48_000).unwrap();
        assert_eq!(kept.len(), 16_000);
        assert!((rms(&kept) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02);

        // 12 kHz is above the 8 kHz output Nyquist and would alias to 4 kHz unfiltered
        let removed = resample_mono_to_16khz(tone(48_000, 12_000.0, 48_000), 48_000).unwrap();
        assert!(rms(&removed) < 0.01, "aliased energy {}", rms(&removed));

        let from_cd = resample_mono_to_16khz(tone(44_100, 1_000.0, 44_100), 44_100).unwrap();
        assert_eq!(from_cd.len(), 16_000);
    }

    #[test]
    fn decodes_stereo_48khz_wav_to_mono_16khz() {
        let wav = temp_audio_path("decode-stereo-48k", "wav");
        write_stereo_wav(&wav, 48_000, 4_800);

        let native = decode_file_to_mono(&wav).unwrap();
        let decoded = decode_file_to_mono_16khz(&wav).unwrap();

        assert_eq!(native.sample_rate_hz, 48_000);
        assert_eq!(native.samples.len(), 4_800);
        assert_eq!(decoded.sample_rate_hz, SDK_SAMPLE_RATE_HZ);
        assert_eq!(decoded.samples.len(), 1_600);
        let _ = std::fs::remove_file(wav);
    }

    #[test]
    fn decodes_flac_and_mp3_when_ffmpeg_is_available() {
        let wav = temp_audio_path("decode-transcode-source", "wav");
        write_stereo_wav(&wav, 16_000, 16_000);

        for extension in ["flac", "mp3", "ogg", "m4a"] {
            let encoded = temp_audio_path("decode-transcoded", extension);
            if transcode_with_ffmpeg(&wav, &encoded) {
                let decoded = decode_file_to_mono_16khz(&encoded).unwrap();
                // lossy codecs pad by up to a frame or two
                let len = decoded.samples.len() as i64;
                assert!((len - 16_000).abs() < 4_096, "{extension}: {len} samples");
                let _ = std::fs::remove_file(encoded);
            }
        }

        let _ = std::fs::remove_file(wav);
    }

    #[test]
    fn reports_missing_files_as_open_errors() {
        let err = decode_file_to_mono_16khz("/nonexistent/speakrs/audio.wav").unwrap_err();

        assert!(matches!(err, AudioDecodeError::Open { .. }));
    }

    fn write_stereo_wav(path: &Path, sample_rate_hz: u32, frames: usize) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for idx in 0..frames {
            let sample = ((idx as f32 / 10.0).sin() * i16::MAX as f32 * 0.5) as i16;
            writer.write_sample(sample).unwrap();
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn transcode_with_ffmpeg(source: &Path, destination: &Path) -> bool {
        Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(source)
            .arg(destination)
            .status()
            .is_ok_and(|status| status.success())
    }

    fn temp_audio_path(name: &str, extension: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "speakrs-sdk-{name}-{}.{extension}",
            std::process::id()
        ))
    }
}
