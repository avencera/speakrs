//! Canonical parsing and identity projections for imported manifests

use std::collections::BTreeSet;
use std::fmt;

use serde::Serialize;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use sha2::{Digest, Sha256};

use super::*;

/// Parses and validates one strict manifest without reading tensor files
pub fn parse_manifest(bytes: &[u8]) -> Result<SegmentationManifest, SegmentationBundleError> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(SegmentationBundleError::Invalid(format!(
            "manifest exceeds {MAX_MANIFEST_BYTES} bytes"
        )));
    }
    reject_duplicate_json_keys(bytes)?;
    let manifest: SegmentationManifest = serde_json::from_slice(bytes)?;
    manifest.validate()?;
    Ok(manifest)
}

fn reject_duplicate_json_keys(bytes: &[u8]) -> Result<(), SegmentationBundleError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    deserializer
        .deserialize_any(DuplicateKeyVisitor)
        .map_err(SegmentationBundleError::Json)?;
    deserializer.end()?;
    Ok(())
}

struct DuplicateKeySeed;

impl<'de> DeserializeSeed<'de> for DuplicateKeySeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateKeyVisitor)
    }
}

struct DuplicateKeyVisitor;

impl<'de> Visitor<'de> for DuplicateKeyVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key) {
                return Err(de::Error::custom("JSON object contains a duplicate key"));
            }
            map.next_value_seed(DuplicateKeySeed)?;
        }
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element_seed(DuplicateKeySeed)?.is_some() {}
        Ok(())
    }

    fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(())
    }

    fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(())
    }

    fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(())
    }

    fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(())
    }

    fn visit_str<E>(self, _: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(())
    }
}

/// Computes the deterministic canonical identity digest for a manifest
pub fn canonical_manifest_digest(
    manifest: &SegmentationManifest,
) -> Result<Sha256Digest, SegmentationBundleError> {
    manifest.canonical_digest()
}

/// Computes the derived content identity for a manifest
pub fn canonical_bundle_id(
    manifest: &SegmentationManifest,
) -> Result<Sha256Digest, SegmentationBundleError> {
    manifest.canonical_bundle_id()
}

impl SegmentationManifest {
    /// Parses and validates one strict manifest document
    pub fn from_json(bytes: &[u8]) -> Result<Self, SegmentationBundleError> {
        parse_manifest(bytes)
    }

    /// Returns compact UTF-8 canonical identity bytes
    pub fn canonical_identity_bytes(&self) -> Result<Vec<u8>, SegmentationBundleError> {
        self.canonical_manifest_projection(true)
    }

    /// Returns compact UTF-8 bytes used to derive the content bundle ID
    pub fn canonical_bundle_id_bytes(&self) -> Result<Vec<u8>, SegmentationBundleError> {
        self.canonical_manifest_projection(false)
    }

    fn canonical_manifest_projection(
        &self,
        include_bundle_id: bool,
    ) -> Result<Vec<u8>, SegmentationBundleError> {
        let projection = CanonicalManifest {
            audio: &self.audio,
            format_version: self.format_version,
            geometry: &self.geometry,
            head: &self.head,
            identity: CanonicalIdentity {
                bundle_id: include_bundle_id.then_some(&self.identity.bundle_id),
                components: &self.identity.components,
                config: &self.identity.config,
                environment: &self.identity.environment,
                model: &self.identity.model,
                precision: self.identity.precision,
                producer: &self.identity.producer,
                source: &self.identity.source,
            },
            policy: &self.policy,
            schema_id: &self.schema_id,
            tensors: CanonicalTensorInventory {
                score_stage: self.tensors.score_stage,
                shards: self
                    .tensors
                    .shards
                    .iter()
                    .map(|shard| CanonicalTensorShard {
                        bytes: shard.bytes,
                        chunk_end: shard.chunk_end,
                        chunk_start: shard.chunk_start,
                        sha256: &shard.sha256,
                        shape: shard.shape,
                    })
                    .collect(),
            },
        };
        Ok(serde_json::to_vec(&projection)?)
    }

    /// Computes the SHA-256 digest over canonical identity bytes
    pub fn canonical_digest(&self) -> Result<Sha256Digest, SegmentationBundleError> {
        digest_bytes(&self.canonical_identity_bytes()?)
    }

    /// Computes the derived content identity for this manifest
    pub fn canonical_bundle_id(&self) -> Result<Sha256Digest, SegmentationBundleError> {
        digest_bytes(&self.canonical_bundle_id_bytes()?)
    }

    /// Serializes the complete manifest as compact deterministic JSON
    pub fn to_json(&self) -> Result<Vec<u8>, SegmentationBundleError> {
        Ok(serde_json::to_vec(self)?)
    }
}

#[derive(Serialize)]
struct CanonicalManifest<'a> {
    audio: &'a AudioIdentity,
    format_version: u32,
    geometry: &'a SegmentationGeometry,
    head: &'a SegmentationHead,
    identity: CanonicalIdentity<'a>,
    policy: &'a SegmentationPolicy,
    schema_id: &'a str,
    tensors: CanonicalTensorInventory<'a>,
}

#[derive(Serialize)]
struct CanonicalIdentity<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle_id: Option<&'a Sha256Digest>,
    components: &'a Vec<ComponentIdentity>,
    config: &'a IdentityReference,
    environment: &'a IdentityReference,
    model: &'a IdentityReference,
    precision: Precision,
    producer: &'a IdentityReference,
    source: &'a IdentityReference,
}

#[derive(Serialize)]
struct CanonicalTensorInventory<'a> {
    score_stage: ScoreStage,
    shards: Vec<CanonicalTensorShard<'a>>,
}

#[derive(Serialize)]
struct CanonicalTensorShard<'a> {
    bytes: u64,
    chunk_end: u64,
    chunk_start: u64,
    sha256: &'a Sha256Digest,
    shape: [u64; 3],
}

fn digest_bytes(bytes: &[u8]) -> Result<Sha256Digest, SegmentationBundleError> {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest_from_hash(&digest)
}

fn digest_from_hash(digest: &[u8]) -> Result<Sha256Digest, SegmentationBundleError> {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(text, "{byte:02x}")
            .map_err(|_| SegmentationBundleError::Invalid("digest formatting failed".to_owned()))?;
    }
    Sha256Digest::new(text)
}
