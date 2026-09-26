//! Minimal protobuf wire-format access for ONNX graph surgery
//!
//! The split only drops and adds whole fields, so it walks fields and copies their
//! encoded bytes unchanged instead of decoding the full ONNX schema

/// Errors raised while walking protobuf bytes
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    /// A field or length prefix runs past the end of its message
    #[error("protobuf message is truncated at byte {offset}")]
    Truncated {
        /// Byte offset inside the message
        offset: usize,
    },
    /// A varint has more than ten bytes
    #[error("protobuf varint at byte {offset} is too long")]
    VarintOverflow {
        /// Byte offset inside the message
        offset: usize,
    },
    /// The field uses a deprecated group or unknown wire type
    #[error("protobuf wire type {wire_type} at byte {offset} is not supported")]
    UnsupportedWireType {
        /// Wire type from the field tag
        wire_type: u8,
        /// Byte offset inside the message
        offset: usize,
    },
    /// A field number is zero or out of range
    #[error("protobuf field number {number} at byte {offset} is invalid")]
    InvalidFieldNumber {
        /// Field number from the tag
        number: u64,
        /// Byte offset inside the message
        offset: usize,
    },
    /// A string field is not UTF-8
    #[error("protobuf string field {field} is not UTF-8")]
    InvalidString {
        /// Field number of the string
        field: u32,
    },
}

/// Decoded value of one field
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Payload<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32,
}

/// One field of a message, with its complete encoding for verbatim copies
#[derive(Clone, Copy, Debug)]
pub(crate) struct Field<'a> {
    pub(crate) number: u32,
    pub(crate) payload: Payload<'a>,
    pub(crate) encoded: &'a [u8],
}

impl<'a> Field<'a> {
    /// Length-delimited payload, or `None` for scalar fields
    pub(crate) const fn bytes(&self) -> Option<&'a [u8]> {
        match self.payload {
            Payload::Bytes(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Length-delimited payload as UTF-8 text
    pub(crate) fn string(&self) -> Result<Option<&'a str>, WireError> {
        self.bytes()
            .map(|bytes| {
                std::str::from_utf8(bytes)
                    .map_err(|_| WireError::InvalidString { field: self.number })
            })
            .transpose()
    }
}

/// Iterator over the top-level fields of one message
pub(crate) struct Fields<'a> {
    bytes: &'a [u8],
    offset: usize,
}

/// Walk the fields of one encoded message in order
pub(crate) const fn fields(bytes: &[u8]) -> Fields<'_> {
    Fields { bytes, offset: 0 }
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<Field<'a>, WireError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.bytes.len() {
            return None;
        }
        let field = self.read_field();
        if field.is_err() {
            // stop after the first error so callers see it once
            self.offset = self.bytes.len();
        }
        Some(field)
    }
}

impl<'a> Fields<'a> {
    fn read_field(&mut self) -> Result<Field<'a>, WireError> {
        let start = self.offset;
        let tag = read_varint(self.bytes, &mut self.offset)?;
        let number = tag >> 3;
        if number == 0 || number > u64::from(u32::MAX >> 3) {
            return Err(WireError::InvalidFieldNumber {
                number,
                offset: start,
            });
        }
        let wire_type = (tag & 0b111) as u8;
        let payload = match wire_type {
            0 => Payload::Varint(read_varint(self.bytes, &mut self.offset)?),
            1 => {
                self.skip(8)?;
                Payload::Fixed64
            }
            2 => {
                let length = read_varint(self.bytes, &mut self.offset)?;
                let length = usize::try_from(length).map_err(|_| WireError::Truncated {
                    offset: self.offset,
                })?;
                let payload_start = self.offset;
                self.skip(length)?;
                Payload::Bytes(&self.bytes[payload_start..self.offset])
            }
            5 => {
                self.skip(4)?;
                Payload::Fixed32
            }
            _ => {
                return Err(WireError::UnsupportedWireType {
                    wire_type,
                    offset: start,
                });
            }
        };

        Ok(Field {
            number: number as u32,
            payload,
            encoded: &self.bytes[start..self.offset],
        })
    }

    fn skip(&mut self, length: usize) -> Result<(), WireError> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(WireError::Truncated {
                offset: self.offset,
            })?;
        self.offset = end;
        Ok(())
    }
}

fn read_varint(bytes: &[u8], offset: &mut usize) -> Result<u64, WireError> {
    let start = *offset;
    let mut value = 0_u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*offset)
            .ok_or(WireError::Truncated { offset: *offset })?;
        *offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(WireError::VarintOverflow { offset: start })
}

/// Decode packed or unpacked repeated varints from one field occurrence
pub(crate) fn push_varints(field: &Field<'_>, values: &mut Vec<u64>) -> Result<(), WireError> {
    match field.payload {
        Payload::Varint(value) => values.push(value),
        Payload::Bytes(bytes) => {
            let mut offset = 0;
            while offset < bytes.len() {
                values.push(read_varint(bytes, &mut offset)?);
            }
        }
        Payload::Fixed32 | Payload::Fixed64 => {}
    }
    Ok(())
}

/// Append one length-delimited field
pub(crate) fn write_bytes_field(out: &mut Vec<u8>, number: u32, payload: &[u8]) {
    write_varint(out, (u64::from(number) << 3) | 2);
    write_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_all_wire_types_and_keeps_encodings() {
        let mut message = vec![0x08, 0x96, 0x01];
        message.extend([0x11, 1, 2, 3, 4, 5, 6, 7, 8]);
        write_bytes_field(&mut message, 3, b"abc");
        message.extend([0x25, 1, 2, 3, 4]);

        let fields = fields(&message).collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(fields.len(), 4);
        assert_eq!(fields[0].payload, Payload::Varint(150));
        assert_eq!(fields[1].payload, Payload::Fixed64);
        assert_eq!(fields[2].string().unwrap(), Some("abc"));
        assert_eq!(fields[3].payload, Payload::Fixed32);
        let rebuilt = fields
            .iter()
            .flat_map(|field| field.encoded.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(rebuilt, message);
    }

    #[test]
    fn decodes_packed_and_unpacked_varints() {
        let mut packed = Vec::new();
        write_bytes_field(&mut packed, 1, &[0x01, 0x96, 0x01, 0x03]);
        let unpacked = [0x08, 0x07];
        let mut values = Vec::new();

        for field in fields(&packed).chain(fields(&unpacked)) {
            push_varints(&field.unwrap(), &mut values).unwrap();
        }

        assert_eq!(values, [1, 150, 3, 7]);
    }

    #[test]
    fn rejects_truncated_and_unsupported_fields() {
        let truncated_length = [0x0a, 0x05, b'a'];
        let truncated_varint = [0x08, 0x80];
        let group = [0x0b];
        let field_zero = [0x00, 0x01];

        let first_error = |bytes: &[u8]| fields(bytes).find_map(Result::err);

        assert!(matches!(
            first_error(&truncated_length),
            Some(WireError::Truncated { .. })
        ));
        assert!(matches!(
            first_error(&truncated_varint),
            Some(WireError::Truncated { .. })
        ));
        assert!(matches!(
            first_error(&group),
            Some(WireError::UnsupportedWireType { wire_type: 3, .. })
        ));
        assert!(matches!(
            first_error(&field_zero),
            Some(WireError::InvalidFieldNumber { number: 0, .. })
        ));
        assert_eq!(fields(&truncated_length).count(), 1);
    }
}
