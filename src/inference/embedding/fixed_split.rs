//! Host fbank frontend plus ONNX tail for the fixed embedding model
//!
//! The fixed model computes its fbank inside `Session::run`, and ONNX Runtime has no
//! CUDA kernel for `DFT`, so each call spends most of its time in a single-threaded
//! host FFT while the GPU waits. This module cuts the verified model at the frontend
//! output: the frontend runs in Rust once per window, and the unchanged rest of the
//! graph runs as its own session. Deriving the tail from the pinned model, instead of
//! shipping a second file, keeps the model bundle and its digests unchanged

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ndarray::ArrayView2;
use ort::session::Session;

use super::wespeaker_fbank::{
    FRAME_SAMPLES, FbankFrontendError, MEL_BANDS, SPECTRUM_BINS, WeSpeakerFbank,
};

mod wire;

pub use wire::WireError;
use wire::{Field, fields, push_varints, write_bytes_field};

const MODEL_GRAPH: u32 = 7;
const GRAPH_NODE: u32 = 1;
const GRAPH_INITIALIZER: u32 = 5;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;
const GRAPH_VALUE_INFO: u32 = 13;
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const NODE_METADATA: u32 = 9;
const ENTRY_KEY: u32 = 1;
const ENTRY_VALUE: u32 = 2;
const VALUE_INFO_NAME: u32 = 1;
const VALUE_INFO_TYPE: u32 = 2;
const TYPE_TENSOR: u32 = 1;
const TENSOR_TYPE_ELEMENT: u32 = 1;
const TENSOR_TYPE_SHAPE: u32 = 2;
const SHAPE_DIMENSION: u32 = 1;
const DIMENSION_VALUE: u32 = 1;
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
const TENSOR_FLOAT_DATA: u32 = 4;
const TENSOR_NAME: u32 = 8;
const TENSOR_RAW_DATA: u32 = 9;
const TENSOR_DATA_LOCATION: u32 = 14;
const FLOAT_ELEMENT: u64 = 1;
const EXTERNAL_DATA_LOCATION: u64 = 1;

const NAME_SCOPES_KEY: &str = "pkg.torch.onnx.name_scopes";
// the exporter records each node's module path; `FixedEmbeddingWrapper` in
// `scripts/export_models.py` names its frontend module `fbank_model`
const FRONTEND_SCOPE_PREFIX: &str = "['', 'fbank_model'";

/// Errors raised while cutting the fixed embedding model at its frontend output
#[derive(Debug, thiserror::Error)]
pub enum FixedSplitError {
    /// The model file could not be read
    #[error("failed to read fixed embedding model `{path}`: {source}")]
    Read {
        /// Model path
        path: PathBuf,
        /// Underlying I/O error
        source: std::io::Error,
    },
    /// The feature frames of the model do not match its waveform window
    #[error(
        "fixed embedding model has {actual} feature frames, expected {expected} for its window"
    )]
    FeatureFrames {
        /// Frames implied by the waveform window
        expected: usize,
        /// Frames of the cut tensor
        actual: usize,
    },
    /// The model bytes are not a well-formed protobuf message
    #[error("fixed embedding model is not valid protobuf: {0}")]
    Wire(#[from] WireError),
    /// The model has no graph
    #[error("fixed embedding model has no graph")]
    MissingGraph,
    /// No node carries the frontend name scope
    #[error("fixed embedding model has no nodes in the `fbank_model` scope")]
    MissingFrontend,
    /// The frontend does not read exactly one graph input
    #[error("fixed embedding frontend must read one graph input, found {inputs:?}")]
    FrontendInputs {
        /// Graph inputs read by frontend nodes
        inputs: Vec<String>,
    },
    /// The frontend does not hand exactly one tensor to the rest of the graph
    #[error("fixed embedding frontend must produce one tail input, found {tensors:?}")]
    FrontendOutputs {
        /// Frontend tensors read outside the frontend
        tensors: Vec<String>,
    },
    /// A frontend node reads a value that the tail produces
    #[error("fixed embedding frontend reads tail tensor `{tensor}`")]
    FrontendReadsTail {
        /// Tail tensor name
        tensor: String,
    },
    /// A tail node reads the waveform directly
    #[error("fixed embedding tail reads frontend input `{input}`")]
    TailReadsWaveform {
        /// Graph input name
        input: String,
    },
    /// The frontend output tensor has no recorded static shape
    #[error("fixed embedding feature tensor `{tensor}` has no static shape")]
    MissingFeatureShape {
        /// Feature tensor name
        tensor: String,
    },
    /// The frontend output tensor is not `[1, frames, 80]` float
    #[error(
        "fixed embedding feature tensor `{tensor}` has shape {shape:?}, expected [1, frames, 80]"
    )]
    FeatureShape {
        /// Feature tensor name
        tensor: String,
        /// Recorded dimensions
        shape: Vec<u64>,
    },
    /// The frontend does not have exactly one constant of the expected shape
    #[error("fixed embedding frontend must have one {constant} constant, found {count}")]
    FrontendConstant {
        /// Constant role
        constant: &'static str,
        /// Matching initializers
        count: usize,
    },
    /// A frontend constant cannot be read as inline float data
    #[error("fixed embedding constant `{name}` is not usable: {reason}")]
    ConstantData {
        /// Initializer name
        name: String,
        /// Why the data was rejected
        reason: &'static str,
    },
    /// The host frontend rejected the constants
    #[error(transparent)]
    Frontend(#[from] FbankFrontendError),
}

/// Fixed embedding model split into a host frontend and an ONNX tail
pub(crate) struct FixedEmbeddingSplit {
    /// Serialized ONNX model that starts at the fbank features
    pub(crate) tail_model: Vec<u8>,
    /// Tail input that receives `[1, frames, 80]` fbank features
    pub(crate) feature_input: String,
    /// Fbank frames per waveform window
    pub(crate) feature_frames: usize,
    /// Host frontend built from the model's own window and mel matrix
    pub(crate) frontend: WeSpeakerFbank,
}

/// Tail session and host frontend of a split fixed embedding model
pub(crate) struct FixedSplitSession {
    pub(crate) tail: Session,
    pub(crate) feature_input: String,
    pub(crate) feature_frames: usize,
    pub(crate) frontend: WeSpeakerFbank,
}

impl FixedSplitSession {
    /// Split the model file and load its tail with `build_tail`
    pub(crate) fn load<E>(
        model_path: &Path,
        window_samples: usize,
        build_tail: impl FnOnce(&[u8]) -> Result<Session, E>,
    ) -> Result<Self, E>
    where
        E: From<FixedSplitError>,
    {
        let model = std::fs::read(model_path).map_err(|source| FixedSplitError::Read {
            path: model_path.to_path_buf(),
            source,
        })?;
        let split = split_fixed_embedding(&model)?;
        let expected = WeSpeakerFbank::frame_count(window_samples);
        if split.feature_frames != expected {
            return Err(FixedSplitError::FeatureFrames {
                expected,
                actual: split.feature_frames,
            }
            .into());
        }

        Ok(Self {
            tail: build_tail(&split.tail_model)?,
            feature_input: split.feature_input,
            feature_frames: split.feature_frames,
            frontend: split.frontend,
        })
    }
}

/// Cut a fixed embedding model at its frontend output
pub(crate) fn split_fixed_embedding(model: &[u8]) -> Result<FixedEmbeddingSplit, FixedSplitError> {
    let graph = nested(model, MODEL_GRAPH)?.ok_or(FixedSplitError::MissingGraph)?;
    let parsed = ParsedGraph::parse(graph)?;
    let cut = parsed.frontend_cut()?;
    let feature_info = parsed
        .value_infos
        .iter()
        .find(|info| info.name == cut.feature)
        .ok_or_else(|| FixedSplitError::MissingFeatureShape {
            tensor: cut.feature.to_owned(),
        })?;
    let feature_frames = feature_frames(cut.feature, feature_info.payload)?;
    let window = parsed.frontend_constant(&cut, "window", |dims| {
        dims.iter().product::<u64>() == FRAME_SAMPLES as u64
            && dims.last() == Some(&(FRAME_SAMPLES as u64))
    })?;
    let mel = parsed.frontend_constant(&cut, "mel", |dims| {
        dims == [SPECTRUM_BINS as u64, MEL_BANDS as u64]
    })?;
    let mel = ArrayView2::from_shape((SPECTRUM_BINS, MEL_BANDS), &mel)
        .expect("mel constant length was checked against its shape");
    let frontend = WeSpeakerFbank::new(&window, mel)?;

    let mut tail_graph = Vec::with_capacity(graph.len());
    let mut node_index = 0;
    for field in fields(graph) {
        let field = field?;
        match field.number {
            GRAPH_NODE => {
                if !parsed.nodes[node_index].frontend {
                    tail_graph.extend_from_slice(field.encoded);
                }
                node_index += 1;
            }
            GRAPH_INPUT if field_name(&field, VALUE_INFO_NAME)? == Some(cut.waveform) => {
                write_bytes_field(&mut tail_graph, GRAPH_INPUT, feature_info.payload);
            }
            GRAPH_INITIALIZER => {
                if field_name(&field, TENSOR_NAME)?
                    .is_some_and(|name| cut.tail_reads.contains(name))
                {
                    tail_graph.extend_from_slice(field.encoded);
                }
            }
            GRAPH_VALUE_INFO => {
                if !field_name(&field, VALUE_INFO_NAME)?
                    .is_some_and(|name| cut.frontend_values.contains(name))
                {
                    tail_graph.extend_from_slice(field.encoded);
                }
            }
            _ => tail_graph.extend_from_slice(field.encoded),
        }
    }

    let mut tail_model = Vec::with_capacity(model.len());
    for field in fields(model) {
        let field = field?;
        if field.number == MODEL_GRAPH {
            write_bytes_field(&mut tail_model, MODEL_GRAPH, &tail_graph);
        } else {
            tail_model.extend_from_slice(field.encoded);
        }
    }

    Ok(FixedEmbeddingSplit {
        tail_model,
        feature_input: cut.feature.to_owned(),
        feature_frames,
        frontend,
    })
}

struct Node<'a> {
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    frontend: bool,
}

struct ValueInfo<'a> {
    name: &'a str,
    payload: &'a [u8],
}

struct Initializer<'a> {
    name: &'a str,
    payload: &'a [u8],
}

struct ParsedGraph<'a> {
    nodes: Vec<Node<'a>>,
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    initializers: Vec<Initializer<'a>>,
    value_infos: Vec<ValueInfo<'a>>,
}

/// Tensors on each side of the frontend boundary
struct FrontendCut<'a> {
    waveform: &'a str,
    feature: &'a str,
    frontend_values: HashSet<&'a str>,
    frontend_reads: HashSet<&'a str>,
    tail_reads: HashSet<&'a str>,
}

impl<'a> ParsedGraph<'a> {
    fn parse(graph: &'a [u8]) -> Result<Self, FixedSplitError> {
        let mut parsed = Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            value_infos: Vec::new(),
        };
        for field in fields(graph) {
            let field = field?;
            let Some(payload) = field.bytes() else {
                continue;
            };
            match field.number {
                GRAPH_NODE => parsed.nodes.push(parse_node(payload)?),
                GRAPH_INPUT => parsed.inputs.extend(field_name(&field, VALUE_INFO_NAME)?),
                GRAPH_OUTPUT => parsed.outputs.extend(field_name(&field, VALUE_INFO_NAME)?),
                GRAPH_INITIALIZER => {
                    if let Some(name) = field_name(&field, TENSOR_NAME)? {
                        parsed.initializers.push(Initializer { name, payload });
                    }
                }
                GRAPH_VALUE_INFO => {
                    if let Some(name) = field_name(&field, VALUE_INFO_NAME)? {
                        parsed.value_infos.push(ValueInfo { name, payload });
                    }
                }
                _ => {}
            }
        }
        Ok(parsed)
    }

    fn frontend_cut(&self) -> Result<FrontendCut<'a>, FixedSplitError> {
        let (frontend, tail): (Vec<_>, Vec<_>) = self.nodes.iter().partition(|node| node.frontend);
        if frontend.is_empty() {
            return Err(FixedSplitError::MissingFrontend);
        }
        let frontend_values: HashSet<&str> = frontend
            .iter()
            .flat_map(|node| node.outputs.iter().copied())
            .collect();
        let frontend_reads: HashSet<&str> = frontend
            .iter()
            .flat_map(|node| node.inputs.iter().copied())
            .filter(|name| !name.is_empty())
            .collect();
        let tail_reads: HashSet<&str> = tail
            .iter()
            .flat_map(|node| node.inputs.iter().copied())
            .filter(|name| !name.is_empty())
            .collect();
        let tail_values: HashSet<&str> = tail
            .iter()
            .flat_map(|node| node.outputs.iter().copied())
            .collect();

        if let Some(tensor) = frontend_reads
            .iter()
            .find(|name| tail_values.contains(*name))
        {
            return Err(FixedSplitError::FrontendReadsTail {
                tensor: (*tensor).to_owned(),
            });
        }
        let waveform_inputs = self
            .inputs
            .iter()
            .copied()
            .filter(|input| frontend_reads.contains(input))
            .collect::<Vec<_>>();
        let [waveform] = waveform_inputs.as_slice() else {
            return Err(FixedSplitError::FrontendInputs {
                inputs: waveform_inputs
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect(),
            });
        };
        if tail_reads.contains(waveform) {
            return Err(FixedSplitError::TailReadsWaveform {
                input: (*waveform).to_owned(),
            });
        }

        let mut crossing = frontend_values
            .iter()
            .copied()
            .filter(|value| tail_reads.contains(value) || self.outputs.contains(value))
            .collect::<Vec<_>>();
        crossing.sort_unstable();
        let [feature] = crossing.as_slice() else {
            return Err(FixedSplitError::FrontendOutputs {
                tensors: crossing.iter().map(|name| (*name).to_owned()).collect(),
            });
        };
        if self.outputs.contains(feature) {
            return Err(FixedSplitError::FrontendOutputs {
                tensors: vec![(*feature).to_owned()],
            });
        }

        Ok(FrontendCut {
            waveform,
            feature,
            frontend_values,
            frontend_reads,
            tail_reads,
        })
    }

    /// Read the one float initializer of the frontend whose shape matches
    fn frontend_constant(
        &self,
        cut: &FrontendCut<'_>,
        constant: &'static str,
        shape_matches: impl Fn(&[u64]) -> bool,
    ) -> Result<Vec<f32>, FixedSplitError> {
        let mut matches = Vec::new();
        for initializer in &self.initializers {
            if !cut.frontend_reads.contains(initializer.name) {
                continue;
            }
            let tensor = parse_tensor(initializer)?;
            // integer frame indices can share the window's element count
            if tensor.data_type == Some(FLOAT_ELEMENT) && shape_matches(&tensor.dims) {
                matches.push(tensor);
            }
        }
        if matches.len() != 1 {
            return Err(FixedSplitError::FrontendConstant {
                constant,
                count: matches.len(),
            });
        }
        matches.remove(0).float_values()
    }
}

struct Tensor<'a> {
    name: &'a str,
    dims: Vec<u64>,
    data_type: Option<u64>,
    external: bool,
    raw_data: Option<&'a [u8]>,
    float_data: Vec<&'a [u8]>,
}

impl Tensor<'_> {
    fn float_values(&self) -> Result<Vec<f32>, FixedSplitError> {
        let reject = |reason| FixedSplitError::ConstantData {
            name: self.name.to_owned(),
            reason,
        };
        if self.data_type != Some(FLOAT_ELEMENT) {
            return Err(reject("element type is not float32"));
        }
        if self.external {
            return Err(reject("data is stored outside the model"));
        }
        let expected = usize::try_from(self.dims.iter().product::<u64>())
            .map_err(|_| reject("element count does not fit in memory"))?;
        let bytes = match (self.raw_data, self.float_data.as_slice()) {
            (Some(raw), []) => raw.to_vec(),
            (None, packed) => packed.concat(),
            (Some(_), _) => return Err(reject("data is stored twice")),
        };
        if bytes.len() != expected * size_of::<f32>() {
            return Err(reject("data length does not match its shape"));
        }
        let (chunks, _) = bytes.as_chunks::<{ size_of::<f32>() }>();
        Ok(chunks.iter().copied().map(f32::from_le_bytes).collect())
    }
}

fn parse_tensor<'a>(initializer: &Initializer<'a>) -> Result<Tensor<'a>, FixedSplitError> {
    let mut tensor = Tensor {
        name: initializer.name,
        dims: Vec::new(),
        data_type: None,
        external: false,
        raw_data: None,
        float_data: Vec::new(),
    };
    for field in fields(initializer.payload) {
        let field = field?;
        match (field.number, field.payload) {
            (TENSOR_DIMS, _) => push_varints(&field, &mut tensor.dims)?,
            (TENSOR_DATA_TYPE, wire::Payload::Varint(value)) => tensor.data_type = Some(value),
            (TENSOR_DATA_LOCATION, wire::Payload::Varint(value)) => {
                tensor.external = value == EXTERNAL_DATA_LOCATION;
            }
            (TENSOR_RAW_DATA, wire::Payload::Bytes(bytes)) => tensor.raw_data = Some(bytes),
            (TENSOR_FLOAT_DATA, wire::Payload::Bytes(bytes)) => tensor.float_data.push(bytes),
            (TENSOR_FLOAT_DATA, _) => {
                return Err(FixedSplitError::ConstantData {
                    name: initializer.name.to_owned(),
                    reason: "float data is not packed",
                });
            }
            _ => {}
        }
    }
    Ok(tensor)
}

fn parse_node(payload: &[u8]) -> Result<Node<'_>, FixedSplitError> {
    let mut node = Node {
        inputs: Vec::new(),
        outputs: Vec::new(),
        frontend: false,
    };
    for field in fields(payload) {
        let field = field?;
        match field.number {
            NODE_INPUT => node.inputs.extend(field.string()?),
            NODE_OUTPUT => node.outputs.extend(field.string()?),
            NODE_METADATA => node.frontend |= is_frontend_scope(&field)?,
            _ => {}
        }
    }
    Ok(node)
}

fn is_frontend_scope(entry: &Field<'_>) -> Result<bool, FixedSplitError> {
    let Some(entry) = entry.bytes() else {
        return Ok(false);
    };
    let mut key = None;
    let mut value = None;
    for field in fields(entry) {
        let field = field?;
        match field.number {
            ENTRY_KEY => key = field.string()?,
            ENTRY_VALUE => value = field.string()?,
            _ => {}
        }
    }
    Ok(key == Some(NAME_SCOPES_KEY)
        && value.is_some_and(|value| value.starts_with(FRONTEND_SCOPE_PREFIX)))
}

/// Read the name field of a nested message
fn field_name<'a>(field: &Field<'a>, name_field: u32) -> Result<Option<&'a str>, FixedSplitError> {
    let Some(payload) = field.bytes() else {
        return Ok(None);
    };
    for inner in fields(payload) {
        let inner = inner?;
        if inner.number == name_field {
            return Ok(inner.string()?);
        }
    }
    Ok(None)
}

/// Frames of a `[1, frames, 80]` float feature tensor
fn feature_frames(tensor: &str, value_info: &[u8]) -> Result<usize, FixedSplitError> {
    let missing = || FixedSplitError::MissingFeatureShape {
        tensor: tensor.to_owned(),
    };
    let tensor_type = nested(value_info, VALUE_INFO_TYPE)?
        .and_then(|type_proto| nested(type_proto, TYPE_TENSOR).transpose())
        .transpose()?
        .ok_or_else(missing)?;
    let mut element = None;
    let mut dims = Vec::new();
    for field in fields(tensor_type) {
        let field = field?;
        match (field.number, field.payload) {
            (TENSOR_TYPE_ELEMENT, wire::Payload::Varint(value)) => element = Some(value),
            (TENSOR_TYPE_SHAPE, wire::Payload::Bytes(shape)) => {
                for dimension in fields(shape) {
                    let dimension = dimension?;
                    if dimension.number != SHAPE_DIMENSION {
                        continue;
                    }
                    let value = dimension
                        .bytes()
                        .map(static_dimension)
                        .transpose()?
                        .flatten()
                        .ok_or_else(missing)?;
                    dims.push(value);
                }
            }
            _ => {}
        }
    }

    match (element, dims.as_slice()) {
        (Some(FLOAT_ELEMENT), [1, frames, bands]) if *frames > 0 && *bands == MEL_BANDS as u64 => {
            usize::try_from(*frames).map_err(|_| FixedSplitError::FeatureShape {
                tensor: tensor.to_owned(),
                shape: dims.clone(),
            })
        }
        _ => Err(FixedSplitError::FeatureShape {
            tensor: tensor.to_owned(),
            shape: dims,
        }),
    }
}

fn static_dimension(dimension: &[u8]) -> Result<Option<u64>, FixedSplitError> {
    for field in fields(dimension) {
        let field = field?;
        if let (DIMENSION_VALUE, wire::Payload::Varint(value)) = (field.number, field.payload) {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn nested(message: &[u8], number: u32) -> Result<Option<&[u8]>, FixedSplitError> {
    for field in fields(message) {
        let field = field?;
        if field.number == number {
            return Ok(field.bytes());
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    fn fixture_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    fn toy_model() -> Vec<u8> {
        std::fs::read(fixture_path("fixed_embedding_toy.onnx")).unwrap()
    }

    fn graph_of(model: &[u8]) -> ParsedGraph<'_> {
        let graph = fields(model)
            .map(Result::unwrap)
            .find(|field| field.number == MODEL_GRAPH)
            .and_then(|field| field.bytes())
            .unwrap();
        ParsedGraph::parse(graph).unwrap()
    }

    #[test]
    fn cuts_the_toy_model_at_the_fbank_tensor() {
        let model = toy_model();
        let split = split_fixed_embedding(&model).unwrap();

        assert_eq!(split.feature_frames, 48);
        let tail = graph_of(&split.tail_model);
        assert_eq!(tail.inputs, [split.feature_input.as_str(), "weights"]);
        assert_eq!(tail.outputs, ["output"]);
        assert!(tail.nodes.iter().all(|node| !node.frontend));
        assert!(tail.initializers.iter().all(|initializer| {
            tail.nodes
                .iter()
                .any(|node| node.inputs.contains(&initializer.name))
        }));
        let original = graph_of(&model);
        assert_eq!(
            tail.nodes.len(),
            original.nodes.iter().filter(|node| !node.frontend).count()
        );
    }

    #[test]
    fn host_frontend_uses_the_model_constants() {
        let model = toy_model();
        let mut split = split_fixed_embedding(&model).unwrap();
        let waveform: ndarray::Array1<f32> =
            ndarray_npy::read_npy(fixture_path("wespeaker_fbank_input.npy")).unwrap();
        let expected: ndarray::Array2<f32> =
            ndarray_npy::read_npy(fixture_path("wespeaker_fbank_expected.npy")).unwrap();
        let mut actual = ndarray::Array2::zeros(expected.dim());

        split
            .frontend
            .compute_into(waveform.as_slice().unwrap(), actual.view_mut())
            .unwrap();

        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 2e-4, "max fbank error {max_error}");
    }

    #[test]
    fn rejects_models_without_a_frontend_scope() {
        let model = toy_model();
        let renamed = replace_all(
            &model,
            FRONTEND_SCOPE_PREFIX.as_bytes(),
            b"['', 'fbank_modex'",
        );

        assert!(matches!(
            split_fixed_embedding(&renamed),
            Err(FixedSplitError::MissingFrontend)
        ));
        assert!(matches!(
            split_fixed_embedding(&model[..model.len() / 2]),
            Err(FixedSplitError::Wire(_)) | Err(FixedSplitError::MissingGraph)
        ));
    }

    fn replace_all(haystack: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
        assert_eq!(needle.len(), replacement.len());
        let mut out = haystack.to_vec();
        let mut index = 0;
        while index + needle.len() <= out.len() {
            if &out[index..index + needle.len()] == needle {
                out[index..index + needle.len()].copy_from_slice(replacement);
                index += needle.len();
            } else {
                index += 1;
            }
        }
        out
    }
}

#[cfg(test)]
mod runtime_tests {
    use std::path::Path;

    use ndarray::{Array1, Array2, Array3, Axis};
    use ort::session::Session;
    use ort::value::TensorRef;

    use super::*;

    const TOY_MASK_FRAMES: usize = 24;

    fn session(model: &[u8]) -> Session {
        crate::inference::ensure_ort_ready().unwrap();
        Session::builder()
            .unwrap()
            .commit_from_memory(model)
            .unwrap()
    }

    fn run(session: &mut Session, inputs: Vec<(&str, TensorRef<'_, f32>)>) -> Vec<f32> {
        let inputs = inputs
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor.into()))
            .collect::<Vec<(String, ort::session::SessionInputValue<'_>)>>();
        let outputs = session.run(inputs).unwrap();
        let (_, data) = outputs[0].try_extract_tensor::<f32>().unwrap();
        data.to_vec()
    }

    #[test]
    fn host_frontend_and_tail_match_the_fused_model() {
        let model = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/fixed_embedding_toy.onnx"),
        )
        .unwrap();
        let waveform: Array1<f32> = ndarray_npy::read_npy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/wespeaker_fbank_input.npy"),
        )
        .unwrap();
        let mut split = split_fixed_embedding(&model).unwrap();
        let mut fused = session(&model);
        let mut tail = session(&split.tail_model);
        let waveform = waveform.into_shape_with_order((1, 1, 8_000)).unwrap();
        let mut features = Array3::zeros((1, split.feature_frames, MEL_BANDS));
        split
            .frontend
            .compute_into(
                waveform.as_slice().unwrap(),
                features.index_axis_mut(Axis(0), 0),
            )
            .unwrap();

        for active in [TOY_MASK_FRAMES, 5] {
            let mut weights = Array2::<f32>::zeros((1, TOY_MASK_FRAMES));
            weights.slice_mut(ndarray::s![.., ..active]).fill(1.0);
            let expected = run(
                &mut fused,
                vec![
                    (
                        "waveform",
                        TensorRef::from_array_view(waveform.view()).unwrap(),
                    ),
                    (
                        "weights",
                        TensorRef::from_array_view(weights.view()).unwrap(),
                    ),
                ],
            );
            let actual = run(
                &mut tail,
                vec![
                    (
                        split.feature_input.as_str(),
                        TensorRef::from_array_view(features.view()).unwrap(),
                    ),
                    (
                        "weights",
                        TensorRef::from_array_view(weights.view()).unwrap(),
                    ),
                ],
            );

            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "split {actual} differs from fused {expected}"
                );
            }
        }
    }
}
