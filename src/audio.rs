use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::num::{NonZeroU16, NonZeroU32};
use std::path::Path;

use thiserror::Error;

use crate::imported_segmentation::Sha256Digest;
use crate::pipeline::canonical_waveform_digest;

const CHUNK_HEADER_BYTES: u64 = 8;
const RIFF_HEADER_BYTES: u64 = 12;
const SAMPLE_BYTES: u64 = 2;
const READ_BUFFER_BYTES: usize = 8192;

/// Errors returned while loading or validating a WAV file
#[derive(Debug, Error)]
pub enum WavError {
    /// A filesystem operation failed while reading the WAV file
    #[error("WAV I/O error: {0}")]
    Io(#[from] io::Error),

    /// The file ended before the declared WAV structure was complete
    #[error("WAV file is truncated")]
    Truncated,

    /// The file does not start with a RIFF container
    #[error("expected RIFF WAV")]
    InvalidRiff,

    /// The RIFF size field does not describe the complete file
    #[error(
        "RIFF size is {declared} bytes, but the file declares {actual} bytes after the RIFF header"
    )]
    RiffSizeMismatch {
        /// The size declared by the RIFF header
        declared: u64,
        /// The number of bytes present after the RIFF header
        actual: u64,
    },

    /// The RIFF size is too small to contain the WAVE form type
    #[error("invalid RIFF size {size}")]
    InvalidRiffSize {
        /// The invalid RIFF body size
        size: u64,
    },

    /// The RIFF form type is not WAVE
    #[error("expected WAVE file")]
    InvalidWave,

    /// The remaining RIFF bytes cannot contain a complete chunk header
    #[error("WAV chunk header is truncated with {remaining} bytes remaining")]
    TruncatedChunkHeader {
        /// The number of RIFF bytes left after the incomplete header
        remaining: u64,
    },

    /// A chunk extends beyond the RIFF container
    #[error("WAV chunk {chunk:?} with size {size} exceeds the RIFF bounds")]
    InvalidChunkSize {
        /// The four-byte chunk identifier
        chunk: [u8; 4],
        /// The size declared by the chunk header
        size: u32,
    },

    /// The format chunk is shorter than the required PCM header
    #[error("WAV fmt chunk is {size} bytes; expected at least 16")]
    InvalidFormatChunkSize {
        /// The size declared by the format chunk
        size: u32,
    },

    /// More than one format chunk was found
    #[error("WAV contains more than one fmt chunk")]
    DuplicateFormatChunk,

    /// Audio data appeared before a format chunk
    #[error("WAV fmt chunk must appear before data chunk")]
    MissingFormatChunk,

    /// No audio data chunk was found
    #[error("no data chunk found in WAV")]
    MissingDataChunk,

    /// More than one audio data chunk was found
    #[error("WAV contains more than one data chunk")]
    DuplicateDataChunk,

    /// The WAV encoding is not supported
    #[error("unsupported WAV encoding {format}; expected PCM")]
    UnsupportedEncoding {
        /// The unsupported WAV format code
        format: u16,
    },

    /// The WAV channel count is not the supported mono layout
    #[error("invalid WAV channel count {channels}; expected mono")]
    InvalidChannels {
        /// The channel count declared by the WAV format
        channels: u16,
    },

    /// The WAV sample rate is zero
    #[error("invalid WAV sample rate {sample_rate}")]
    InvalidSampleRate {
        /// The sample rate declared by the WAV format
        sample_rate: u32,
    },

    /// The WAV sample width is not the supported 16-bit representation
    #[error("unsupported WAV sample width {bits_per_sample} bits; expected 16")]
    UnsupportedSampleWidth {
        /// The sample width declared by the WAV format
        bits_per_sample: u16,
    },

    /// The format chunk block alignment is inconsistent with its sample layout
    #[error("WAV block alignment is {actual}; expected {expected}")]
    InconsistentBlockAlignment {
        /// The alignment implied by the supported sample layout
        expected: u16,
        /// The alignment declared by the WAV format
        actual: u16,
    },

    /// The format chunk byte rate is inconsistent with its sample layout
    #[error("WAV byte rate is {actual}; expected {expected}")]
    InconsistentByteRate {
        /// The byte rate implied by the supported sample layout
        expected: u64,
        /// The byte rate declared by the WAV format
        actual: u32,
    },

    /// The data chunk ends in an incomplete sample frame
    #[error("WAV data chunk has {bytes} bytes; expected a multiple of {block_alignment}")]
    IncompleteSamples {
        /// The number of bytes declared by the data chunk
        bytes: u32,
        /// The required number of bytes in one sample frame
        block_alignment: u16,
    },

    /// The decoded sample vector cannot be represented or allocated safely
    #[error("WAV contains too many samples to allocate: {sample_count}")]
    Allocation {
        /// The number of samples that could not be allocated
        sample_count: u64,
    },
}

/// A validated non-zero WAV sample rate in hertz
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampleRate(NonZeroU32);

impl SampleRate {
    /// Return the sample rate in hertz
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// A validated non-zero WAV channel count
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChannelCount(NonZeroU16);

impl ChannelCount {
    /// Return the number of channels
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

/// The sample representation supported by the WAV loader
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SampleRepresentation {
    /// Signed little-endian 16-bit PCM samples converted to `f32`
    Pcm16,
}

/// A validated WAV recording decoded into exact `f32` sample values
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedWav {
    sample_rate: SampleRate,
    channels: ChannelCount,
    sample_representation: SampleRepresentation,
    samples: Vec<f32>,
    waveform_sha256: Sha256Digest,
}

impl DecodedWav {
    /// Return the validated sample rate
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Return the validated channel count
    #[must_use]
    pub const fn channels(&self) -> ChannelCount {
        self.channels
    }

    /// Return the validated sample representation
    #[must_use]
    pub const fn sample_representation(&self) -> SampleRepresentation {
        self.sample_representation
    }

    /// Return the decoded samples without copying them
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Consume the decoded WAV and return its samples
    #[must_use]
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }

    /// Return the canonical SHA-256 identity of the decoded waveform
    #[must_use]
    pub fn waveform_sha256(&self) -> &Sha256Digest {
        &self.waveform_sha256
    }
}

/// Load and validate a mono 16-bit PCM WAV file
pub fn load_wav(path: impl AsRef<Path>) -> Result<DecodedWav, WavError> {
    let file = File::open(path)?;
    let file_length = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut riff_header = [0u8; RIFF_HEADER_BYTES as usize];
    read_exact(&mut reader, &mut riff_header)?;
    if &riff_header[0..4] != b"RIFF" {
        return Err(WavError::InvalidRiff);
    }
    if &riff_header[8..12] != b"WAVE" {
        return Err(WavError::InvalidWave);
    }

    let riff_size = u64::from(u32::from_le_bytes(riff_header[4..8].try_into().unwrap()));
    if riff_size < 4 {
        return Err(WavError::InvalidRiffSize { size: riff_size });
    }
    let actual_riff_size = file_length.checked_sub(8).ok_or(WavError::Truncated)?;
    if riff_size != actual_riff_size {
        return Err(WavError::RiffSizeMismatch {
            declared: riff_size,
            actual: actual_riff_size,
        });
    }

    let riff_end = RIFF_HEADER_BYTES
        .checked_add(riff_size - 4)
        .ok_or(WavError::InvalidRiffSize { size: riff_size })?;
    let mut position = RIFF_HEADER_BYTES;
    let mut format = None;
    let mut samples = None;

    while position < riff_end {
        let remaining = riff_end - position;
        if remaining < CHUNK_HEADER_BYTES {
            return Err(WavError::TruncatedChunkHeader { remaining });
        }

        let mut chunk_header = [0u8; CHUNK_HEADER_BYTES as usize];
        read_exact(&mut reader, &mut chunk_header)?;
        let chunk = chunk_header[0..4].try_into().unwrap();
        let chunk_size = u32::from_le_bytes(chunk_header[4..8].try_into().unwrap());
        position += CHUNK_HEADER_BYTES;

        let chunk_data_end =
            position
                .checked_add(u64::from(chunk_size))
                .ok_or(WavError::InvalidChunkSize {
                    chunk,
                    size: chunk_size,
                })?;
        let chunk_end = chunk_data_end
            .checked_add(u64::from(chunk_size % 2))
            .ok_or(WavError::InvalidChunkSize {
                chunk,
                size: chunk_size,
            })?;
        if chunk_end > riff_end {
            return Err(WavError::InvalidChunkSize {
                chunk,
                size: chunk_size,
            });
        }

        match &chunk {
            b"fmt " => {
                if format.is_some() {
                    return Err(WavError::DuplicateFormatChunk);
                }
                if chunk_size < 16 {
                    return Err(WavError::InvalidFormatChunkSize { size: chunk_size });
                }

                let mut format_header = [0u8; 16];
                read_exact(&mut reader, &mut format_header)?;

                let encoding = u16::from_le_bytes(format_header[0..2].try_into().unwrap());
                if encoding != 1 {
                    return Err(WavError::UnsupportedEncoding { format: encoding });
                }

                let channel_count = u16::from_le_bytes(format_header[2..4].try_into().unwrap());
                let channels = NonZeroU16::new(channel_count).ok_or(WavError::InvalidChannels {
                    channels: channel_count,
                })?;
                if channels.get() != 1 {
                    return Err(WavError::InvalidChannels {
                        channels: channel_count,
                    });
                }

                let sample_rate = u32::from_le_bytes(format_header[4..8].try_into().unwrap());
                let sample_rate = NonZeroU32::new(sample_rate)
                    .ok_or(WavError::InvalidSampleRate { sample_rate })?;
                let byte_rate = u32::from_le_bytes(format_header[8..12].try_into().unwrap());
                let block_alignment = u16::from_le_bytes(format_header[12..14].try_into().unwrap());
                let bits_per_sample = u16::from_le_bytes(format_header[14..16].try_into().unwrap());
                if bits_per_sample != 16 {
                    return Err(WavError::UnsupportedSampleWidth { bits_per_sample });
                }

                let expected_block_alignment = channels.get() * 2;
                if block_alignment != expected_block_alignment {
                    return Err(WavError::InconsistentBlockAlignment {
                        expected: expected_block_alignment,
                        actual: block_alignment,
                    });
                }
                let expected_byte_rate =
                    u64::from(sample_rate.get()) * u64::from(expected_block_alignment);
                if u64::from(byte_rate) != expected_byte_rate {
                    return Err(WavError::InconsistentByteRate {
                        expected: expected_byte_rate,
                        actual: byte_rate,
                    });
                }

                format = Some(WavFormat {
                    sample_rate: SampleRate(sample_rate),
                    channels: ChannelCount(channels),
                    sample_representation: SampleRepresentation::Pcm16,
                    block_alignment,
                });
            }
            b"data" => {
                let format = format.as_ref().ok_or(WavError::MissingFormatChunk)?;
                if samples.is_some() {
                    return Err(WavError::DuplicateDataChunk);
                }
                if !u64::from(chunk_size).is_multiple_of(u64::from(format.block_alignment)) {
                    return Err(WavError::IncompleteSamples {
                        bytes: chunk_size,
                        block_alignment: format.block_alignment,
                    });
                }

                let sample_count = u64::from(chunk_size) / SAMPLE_BYTES;
                let sample_count_usize = usize::try_from(sample_count)
                    .map_err(|_| WavError::Allocation { sample_count })?;
                let mut decoded = Vec::new();
                decoded
                    .try_reserve_exact(sample_count_usize)
                    .map_err(|_| WavError::Allocation { sample_count })?;
                let mut remaining = u64::from(chunk_size);
                let mut buffer = [0u8; READ_BUFFER_BYTES];

                while remaining > 0 {
                    let read_length =
                        usize::try_from(remaining.min(u64::try_from(buffer.len()).unwrap()))
                            .unwrap();
                    read_exact(&mut reader, &mut buffer[..read_length])?;
                    for sample_bytes in buffer[..read_length].as_chunks::<2>().0 {
                        let sample = i16::from_le_bytes(*sample_bytes);
                        decoded.push(f32::from(sample) / 32_768.0);
                    }
                    remaining -= u64::try_from(read_length).unwrap();
                }
                samples = Some(decoded);
            }
            _ => {}
        }

        seek_to(&mut reader, chunk_end)?;
        position = chunk_end;
    }

    let format = format.ok_or(WavError::MissingFormatChunk)?;
    let samples = samples.ok_or(WavError::MissingDataChunk)?;
    let waveform_sha256 = canonical_waveform_digest(&samples);

    Ok(DecodedWav {
        sample_rate: format.sample_rate,
        channels: format.channels,
        sample_representation: format.sample_representation,
        samples,
        waveform_sha256,
    })
}

struct WavFormat {
    sample_rate: SampleRate,
    channels: ChannelCount,
    sample_representation: SampleRepresentation,
    block_alignment: u16,
}

fn read_exact(reader: &mut impl Read, bytes: &mut [u8]) -> Result<(), WavError> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            WavError::Truncated
        } else {
            WavError::Io(error)
        }
    })
}

fn seek_to(reader: &mut impl Seek, position: u64) -> Result<(), WavError> {
    reader.seek(SeekFrom::Start(position))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn valid_pcm_wav_exposes_typed_format_samples_and_identity() {
        let data = samples(&[0, 16_384, -16_384]);
        let (_directory, path) = write_wav(wav([
            (*b"fmt ", format_chunk(1, 1, 16_000, 32_000, 2, 16)),
            (*b"data", data),
        ]));

        let decoded = load_wav(&path).unwrap();

        assert_eq!(decoded.sample_rate().get(), 16_000);
        assert_eq!(decoded.channels().get(), 1);
        assert_eq!(decoded.sample_representation(), SampleRepresentation::Pcm16);
        assert_eq!(decoded.samples(), &[0.0, 0.5, -0.5]);
        assert_eq!(
            decoded.waveform_sha256(),
            &canonical_waveform_digest(decoded.samples())
        );
    }

    #[test]
    fn riff_padding_before_data_is_skipped() {
        let (_directory, path) = write_wav(wav([
            (*b"JUNK", vec![0x7f]),
            (*b"fmt ", format_chunk(1, 1, 16_000, 32_000, 2, 16)),
            (*b"data", samples(&[1234])),
        ]));

        let decoded = load_wav(&path).unwrap();

        assert_eq!(decoded.samples(), &[1234.0 / 32_768.0]);
    }

    #[test]
    fn malformed_riff_size_is_rejected() {
        let mut bytes = wav([
            (*b"fmt ", format_chunk(1, 1, 16_000, 32_000, 2, 16)),
            (*b"data", samples(&[0])),
        ]);
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        let (_directory, path) = write_wav(bytes);

        assert!(matches!(
            load_wav(&path),
            Err(WavError::InvalidRiffSize { .. }) | Err(WavError::RiffSizeMismatch { .. })
        ));
    }

    #[test]
    fn malformed_chunk_size_is_rejected() {
        let mut bytes = wav([
            (*b"fmt ", format_chunk(1, 1, 16_000, 32_000, 2, 16)),
            (*b"data", samples(&[0])),
        ]);
        let data_size_offset = 12 + 8 + 16 + 4;
        bytes[data_size_offset..data_size_offset + 4].copy_from_slice(&100u32.to_le_bytes());
        let (_directory, path) = write_wav(bytes);

        assert!(matches!(
            load_wav(&path),
            Err(WavError::InvalidChunkSize { chunk, .. }) if chunk == *b"data"
        ));
    }

    #[test]
    fn truncated_chunk_header_is_rejected() {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend((5u32).to_le_bytes());
        bytes.extend(b"WAVE");
        bytes.push(0);
        let (_directory, path) = write_wav(bytes);

        assert!(matches!(
            load_wav(&path),
            Err(WavError::TruncatedChunkHeader { remaining: 1 })
        ));
    }

    #[test]
    fn truncated_format_chunk_is_rejected() {
        let (_directory, path) = write_wav(wav([(*b"fmt ", vec![0; 8]), (*b"data", Vec::new())]));

        assert!(matches!(
            load_wav(&path),
            Err(WavError::InvalidFormatChunkSize { size: 8 })
        ));
    }

    #[test]
    fn incomplete_samples_are_rejected() {
        let (_directory, path) = write_wav(wav([
            (*b"fmt ", format_chunk(1, 1, 16_000, 32_000, 2, 16)),
            (*b"data", vec![0]),
        ]));

        assert!(matches!(
            load_wav(&path),
            Err(WavError::IncompleteSamples {
                bytes: 1,
                block_alignment: 2
            })
        ));
    }

    #[test]
    fn inconsistent_block_alignment_is_rejected() {
        let (_directory, path) = write_wav(wav([
            (*b"fmt ", format_chunk(1, 1, 16_000, 64_000, 4, 16)),
            (*b"data", Vec::new()),
        ]));

        assert!(matches!(
            load_wav(&path),
            Err(WavError::InconsistentBlockAlignment {
                expected: 2,
                actual: 4
            })
        ));
    }

    #[test]
    fn inconsistent_byte_rate_is_rejected() {
        let (_directory, path) = write_wav(wav([
            (*b"fmt ", format_chunk(1, 1, 16_000, 16_000, 2, 16)),
            (*b"data", Vec::new()),
        ]));

        assert!(matches!(
            load_wav(&path),
            Err(WavError::InconsistentByteRate {
                expected: 32_000,
                actual: 16_000
            })
        ));
    }

    #[test]
    fn unsupported_encoding_is_rejected() {
        let (_directory, path) = write_wav(wav([
            (*b"fmt ", format_chunk(3, 1, 16_000, 32_000, 2, 16)),
            (*b"data", Vec::new()),
        ]));

        assert!(matches!(
            load_wav(&path),
            Err(WavError::UnsupportedEncoding { format: 3 })
        ));
    }

    #[test]
    fn invalid_channel_count_is_rejected() {
        for channels in [0, 2] {
            let (_directory, path) = write_wav(wav([
                (*b"fmt ", format_chunk(1, channels, 16_000, 32_000, 2, 16)),
                (*b"data", Vec::new()),
            ]));

            assert!(matches!(
                load_wav(&path),
                Err(WavError::InvalidChannels { channels: actual }) if actual == channels
            ));
        }
    }

    #[test]
    fn invalid_sample_rate_is_rejected() {
        let (_directory, path) = write_wav(wav([
            (*b"fmt ", format_chunk(1, 1, 0, 0, 2, 16)),
            (*b"data", Vec::new()),
        ]));

        assert!(matches!(
            load_wav(&path),
            Err(WavError::InvalidSampleRate { sample_rate: 0 })
        ));
    }

    #[test]
    fn unsupported_sample_width_is_rejected() {
        let (_directory, path) = write_wav(wav([
            (*b"fmt ", format_chunk(1, 1, 16_000, 16_000, 1, 8)),
            (*b"data", Vec::new()),
        ]));

        assert!(matches!(
            load_wav(&path),
            Err(WavError::UnsupportedSampleWidth { bits_per_sample: 8 })
        ));
    }

    fn write_wav(bytes: Vec<u8>) -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("audio.wav");
        fs::write(&path, bytes).unwrap();
        (directory, path)
    }

    fn wav(chunks: impl IntoIterator<Item = ([u8; 4], Vec<u8>)>) -> Vec<u8> {
        let mut body = b"WAVE".to_vec();
        for (id, payload) in chunks {
            body.extend(id);
            body.extend(u32::try_from(payload.len()).unwrap().to_le_bytes());
            body.extend(payload);
            if body.len() % 2 == 1 {
                body.push(0);
            }
        }

        let mut file = b"RIFF".to_vec();
        file.extend(u32::try_from(body.len()).unwrap().to_le_bytes());
        file.extend(body);
        file
    }

    fn format_chunk(
        encoding: u16,
        channels: u16,
        sample_rate: u32,
        byte_rate: u32,
        block_alignment: u16,
        bits_per_sample: u16,
    ) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend(encoding.to_le_bytes());
        bytes.extend(channels.to_le_bytes());
        bytes.extend(sample_rate.to_le_bytes());
        bytes.extend(byte_rate.to_le_bytes());
        bytes.extend(block_alignment.to_le_bytes());
        bytes.extend(bits_per_sample.to_le_bytes());
        bytes
    }

    fn samples(values: &[i16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}
