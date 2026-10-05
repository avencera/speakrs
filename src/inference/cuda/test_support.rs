//! Locked, test-only selection, trace scopes, call trackers and device mutation support
//!
//! Only locked code calls into this module, and the harness's static scan refuses any
//! reference to it from candidate files. Every harness NVTX range carries the per-run
//! nonce that the locked driver passes in `SPEAKRS_QUALIFY_NONCE`, which candidate code
//! cannot read, so a range a candidate pushes itself never counts

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ffi::{CStr, CString, c_char};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Mutex, OnceLock, PoisonError};

use cudarc::driver::sys::{
    self, CUdevice_attribute, CUfunction_attribute, CUgraphNodeType, CUmemorytype,
    CUstreamCaptureStatus,
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::candidate::Direction;
use super::implementation::{Choice, Selection};
use super::{CudaError, CudaRuntime, CudaSegmentation, ResNetEmbedding};

/// The live planted faults share the real implementation seam
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mutant {
    Precision,
    Shape,
    Fallback,
    Tail,
    Atomic,
    Slow,
    /// Computes correctly except during timing, where it skips the work
    PhaseCheat,
    /// Launches harness work on the stream after its candidate scope closes
    Unscoped,
    /// Launches a kernel from PTX the recorder never saw
    Unlisted,
    /// Answers from the first weights or inputs it saw, like a baked lookup table
    Lookup,
    /// Launches an entry that loads shared memory no thread stored
    UninitShared,
}

impl Mutant {
    /// Parses only the fixed mutation inventory
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "Precision" => Some(Self::Precision),
            "Shape" => Some(Self::Shape),
            "Fallback" => Some(Self::Fallback),
            "Tail" => Some(Self::Tail),
            "Atomic" => Some(Self::Atomic),
            "Slow" => Some(Self::Slow),
            "PhaseCheat" => Some(Self::PhaseCheat),
            "Unscoped" => Some(Self::Unscoped),
            "Unlisted" => Some(Self::Unlisted),
            "Lookup" => Some(Self::Lookup),
            "UninitShared" => Some(Self::UninitShared),
            _ => None,
        }
    }

    /// Whether the planted fault skips its work in this process
    pub(crate) fn skips(self) -> bool {
        self == Self::PhaseCheat && phase() == Some("timing")
    }
}

/// Resolves the fixed implementation names; candidates register a plan type, not a name
pub(crate) fn choice(name: &str) -> Choice {
    match name {
        "Library" => Choice::Library,
        "Oxide" => Choice::Oxide(Selection::Explicit),
        other => Choice::Mutant(Mutant::parse(other).expect("fixed implementation inventory")),
    }
}

/// Qualification controls every boundary itself; production defaults must not enter
/// its Library inputs or unrelated operator stages
pub(crate) fn default_choice(production: Choice) -> Choice {
    if harness().is_some() {
        Choice::Library
    } else {
        production
    }
}

/// Selects one eligible convolution before a batch shares the model
pub(crate) fn select_conv(model: &mut ResNetEmbedding, name: &str, choice_name: &str) -> bool {
    model.select_conv(name, choice(choice_name))
}

/// Selects the Sinc family, plans a covered candidate and invalidates any captured graph
pub(crate) fn select_sinc(
    model: &mut CudaSegmentation,
    runtime: &CudaRuntime,
    shape: [usize; 2],
    choice_name: &str,
) -> Result<(), CudaError> {
    model.select_sinc(runtime, shape[0], shape[1], choice(choice_name))
}

/// Selects the complete stack, plans a covered candidate and invalidates any captured graph
pub(crate) fn select_lstm(
    model: &mut CudaSegmentation,
    runtime: &CudaRuntime,
    shape: [usize; 2],
    choice_name: &str,
) -> Result<(), CudaError> {
    model.select_lstm(runtime, shape, choice(choice_name))
}

#[path = "../../../tests/cuda_qualify/probe.rs"]
pub(crate) mod qualify;

/// The harness environment, read once and only by this locked module
struct Harness {
    nonce: String,
    phase: String,
    nvtx: Option<Nvtx>,
}

struct Nvtx {
    _library: libloading::Library,
    push: unsafe extern "C" fn(*const c_char),
    pop: unsafe extern "C" fn(),
}

fn harness() -> Option<&'static Harness> {
    static HARNESS: OnceLock<Option<Harness>> = OnceLock::new();
    HARNESS
        .get_or_init(|| {
            let phase = std::env::var("SPEAKRS_QUALIFY_PHASE").ok()?;
            let nonce = std::env::var("SPEAKRS_QUALIFY_NONCE").expect("qualification nonce");
            assert!(
                nonce.len() == 32 && nonce.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "the locked driver passes a 128-bit hex nonce"
            );
            let nvtx = std::env::var_os("SPEAKRS_QUALIFY_NVTX").map(|path| {
                // SAFETY: the harness builds this shim from its locked source; both
                // symbols have the exact ABI below and the library stays loaded
                let library = unsafe { libloading::Library::new(path) }
                    .expect("load qualification NVTX shim");
                // SAFETY: the locked shim exports this exact function signature
                let push = *unsafe {
                    library.get::<unsafe extern "C" fn(*const c_char)>(b"qualify_push\0")
                }
                .expect("qualification NVTX push");
                // SAFETY: the locked shim exports this exact function signature
                let pop = *unsafe { library.get::<unsafe extern "C" fn()>(b"qualify_pop\0") }
                    .expect("qualification NVTX pop");
                Nvtx {
                    _library: library,
                    push,
                    pop,
                }
            });
            Some(Harness { nonce, phase, nvtx })
        })
        .as_ref()
}

/// The qualification phase of this process, if the locked driver started it
pub(crate) fn phase() -> Option<&'static str> {
    harness().map(|harness| harness.phase.as_str())
}

/// The scope kinds the trace parser recognizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The locked driver's interval between an input upload and its output download
    Window,
    /// A candidate's `enqueue`, for one declared layer and batch pair
    Candidate,
    /// The Library path of a boundary: the Library choice or an undeclared pair
    Library,
    /// One locked cuDNN or cuBLAS call site
    Call,
    /// The locked LSTM input-projection helper
    Projection,
    /// One locked launch of a Library-owned kernel
    Fixed,
    /// A candidate's `plan`, at batch setup
    Plan,
    /// A locked sub-scope a candidate opens through its phase handle
    Phase,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Self::Window => "window",
            Self::Candidate => "candidate",
            Self::Library => "library",
            Self::Call => "call",
            Self::Projection => "projection",
            Self::Fixed => "fixed",
            Self::Plan => "plan",
            Self::Phase => "phase",
        }
    }
}

struct Frame {
    kind: Kind,
    name: String,
    capture: Option<Capture>,
}

/// The graph a stream is capturing into and its nodes when a scope opened
struct Capture {
    graph: sys::CUgraph,
    before: BTreeSet<usize>,
}

/// Library projection permissions belong to one capture ID, never a node address lifetime
struct ProjectionNodes {
    capture_id: Option<u64>,
    nodes: BTreeSet<usize>,
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
    static CALL_VIOLATIONS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static GRAPH_VIOLATIONS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static PROJECTION_NODES: RefCell<ProjectionNodes> = const { RefCell::new(ProjectionNodes { capture_id: None, nodes: BTreeSet::new() }) };
    static GRAPH_EVIDENCE: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
    static LABEL: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Labels the graph captures that follow with the driver's case id, so each case's
/// captured kernels can be compared with that case's eager profile
pub(crate) fn set_label(label: Option<String>) {
    LABEL.with(|cell| *cell.borrow_mut() = label);
}

/// One open harness scope; closes its NVTX range and checks captured nodes on drop
pub(crate) struct Scope {
    active: bool,
    _thread: PhantomData<Rc<()>>,
}

fn open(kind: Kind, name: &str, stream: Option<&CudaStream>) -> Scope {
    let Some(harness) = harness() else {
        return Scope {
            active: false,
            _thread: PhantomData,
        };
    };

    if kind == Kind::Call {
        check_call(name);
    }

    let capture = stream.and_then(capture_of);
    STACK.with(|stack| {
        stack.borrow_mut().push(Frame {
            kind,
            name: name.to_owned(),
            capture,
        })
    });
    if let Some(nvtx) = &harness.nvtx {
        let text = CString::new(format!("qualify.{}.{}.{name}", harness.nonce, kind.label()))
            .expect("scope names have no nul bytes");
        // SAFETY: NVTX copies the terminated message during the call
        unsafe { (nvtx.push)(text.as_ptr()) };
    }

    Scope {
        active: true,
        _thread: PhantomData,
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        if let Some(nvtx) = harness().and_then(|harness| harness.nvtx.as_ref()) {
            // SAFETY: this guard closes exactly one range on the thread that opened it
            unsafe { (nvtx.pop)() };
        }
        let frame = STACK.with(|stack| stack.borrow_mut().pop());
        if let Some(frame) = frame {
            close_capture(frame);
        }
    }
}

/// Every library call made while a candidate scope is open is a violation
fn check_call(name: &str) {
    let candidate = STACK.with(|stack| {
        stack
            .borrow()
            .iter()
            .rev()
            .find(|frame| frame.kind == Kind::Candidate)
            .map(|frame| frame.name.clone())
    });
    let Some(candidate) = candidate else {
        return;
    };

    CALL_VIOLATIONS.with(|violations| {
        violations
            .borrow_mut()
            .push(format!("{name} inside candidate {candidate}"))
    });
}

/// Library calls made from candidate scopes, eager or during graph capture
pub(crate) fn library_call_violations() -> Vec<String> {
    CALL_VIOLATIONS.with(|violations| violations.borrow().clone())
}

/// Captured graph nodes that a candidate scope may not contain
pub(crate) fn graph_violations() -> Vec<String> {
    GRAPH_VIOLATIONS.with(|violations| violations.borrow().clone())
}

/// Kernel names of every captured candidate and Library scope
pub(crate) fn graph_evidence() -> Vec<Value> {
    GRAPH_EVIDENCE.with(|evidence| evidence.borrow().clone())
}

/// The driver's interval between an upload and its download
pub(crate) fn window(name: &str) -> Scope {
    open(Kind::Window, name, None)
}

/// A candidate's `enqueue` for one declared pair
pub(crate) fn candidate(stream: &CudaStream, layer: &str) -> Scope {
    open(Kind::Candidate, layer, Some(stream))
}

/// The Library path of a boundary
pub(crate) fn library(stream: &CudaStream, layer: &str) -> Scope {
    open(Kind::Library, layer, Some(stream))
}

/// One locked cuDNN or cuBLAS call site
pub(crate) fn call(name: &str) -> Scope {
    open(Kind::Call, name, None)
}

/// The locked LSTM input-projection helper
// unused until an LSTM candidate calls the helper
#[allow(dead_code)]
pub(crate) fn projection(stream: &CudaStream, layer: usize, direction: Direction) -> Scope {
    open(
        Kind::Projection,
        &format!("L{layer}.{}", direction.name()),
        Some(stream),
    )
}

/// One locked launch of a Library-owned kernel
pub(crate) fn fixed(kernel: &str) -> Scope {
    open(Kind::Fixed, kernel, None)
}

/// A candidate's `plan` for one boundary
pub(crate) fn plan(layer: &str) -> Scope {
    open(Kind::Plan, layer, None)
}

/// A locked candidate sub-scope such as `input_proj.L0.forward` or `op.pack`
#[allow(dead_code)]
pub(crate) fn sub_scope(name: &str) -> Scope {
    open(Kind::Phase, name, None)
}

fn side_streams() -> &'static Mutex<Vec<u64>> {
    static STREAMS: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    STREAMS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Records a candidate side stream's driver ID for the result
#[allow(dead_code)]
pub(crate) fn register_side_stream(stream: &CudaStream) {
    let mut id = 0;
    // SAFETY: the stream is live and `id` is a valid out parameter
    let result = unsafe { sys::cuStreamGetId(stream.cu_stream(), &mut id) };
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "side stream ID");
    side_streams()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(id);
}

/// Driver IDs of every registered candidate side stream
pub(crate) fn registered_side_streams() -> Vec<u64> {
    side_streams()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

fn capture_of(stream: &CudaStream) -> Option<Capture> {
    let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id = 0;
    let mut graph = std::ptr::null_mut();
    let mut dependencies = std::ptr::null();
    let mut count = 0;
    // SAFETY: every out pointer is valid; the stream is live for the call
    let result = unsafe {
        sys::cuStreamGetCaptureInfo_v2(
            stream.cu_stream(),
            &mut status,
            &mut id,
            &mut graph,
            &mut dependencies,
            &mut count,
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS
        || status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
        || graph.is_null()
    {
        return None;
    }

    // capture IDs do not reuse destroyed graph addresses
    PROJECTION_NODES.with(|cell| {
        let mut tracking = cell.borrow_mut();
        if tracking.capture_id != Some(id) {
            tracking.capture_id = Some(id);
            tracking.nodes.clear();
        }
    });

    Some(Capture {
        graph,
        before: graph_nodes(graph),
    })
}

fn graph_nodes(graph: sys::CUgraph) -> BTreeSet<usize> {
    let mut count = 0;
    // SAFETY: a null node array asks only for the count
    let result = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) };
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "count captured nodes");
    // the driver refuses a node list for a graph that has no nodes yet
    if count == 0 {
        return BTreeSet::new();
    }

    let mut nodes = vec![std::ptr::null_mut(); count];
    // SAFETY: the array holds exactly `count` node handles
    let result = unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) };
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "list captured nodes");
    nodes.truncate(count);
    nodes.into_iter().map(|node| node as usize).collect()
}

/// What one captured node is, for the scope rules
enum Node {
    Kernel(String),
    DeviceCopy,
    HostCopy,
    Memset,
    Other(String),
}

fn describe(node: usize) -> Node {
    let node = node as sys::CUgraphNode;
    let mut kind = CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
    // SAFETY: the node belongs to a live graph under capture
    let result = unsafe { sys::cuGraphNodeGetType(node, &mut kind) };
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "captured node type");
    match kind {
        CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => Node::Kernel(kernel_name(node)),
        CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => Node::Memset,
        CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => {
            let mut copy = std::mem::MaybeUninit::<sys::CUDA_MEMCPY3D>::uninit();
            // SAFETY: the node is a copy node and `copy` is a valid out parameter
            let result = unsafe { sys::cuGraphMemcpyNodeGetParams(node, copy.as_mut_ptr()) };
            assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "captured copy");
            // SAFETY: the driver wrote the whole description on success
            let copy = unsafe { copy.assume_init() };
            let device = |kind: CUmemorytype| {
                matches!(
                    kind,
                    CUmemorytype::CU_MEMORYTYPE_DEVICE | CUmemorytype::CU_MEMORYTYPE_ARRAY
                )
            };
            if device(copy.srcMemoryType) && device(copy.dstMemoryType) {
                Node::DeviceCopy
            } else {
                Node::HostCopy
            }
        }
        other => Node::Other(format!("{other:?}")),
    }
}

fn kernel_name(node: sys::CUgraphNode) -> String {
    let mut params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
    // SAFETY: the node is a kernel node and `params` is a valid out parameter
    let result = unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) };
    assert_eq!(
        result,
        sys::CUresult::CUDA_SUCCESS,
        "captured kernel parameters"
    );
    // SAFETY: the driver wrote the whole parameter block on success
    let params = unsafe { params.assume_init() };
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: the handles come from the driver; it writes a static C string
    let result = unsafe {
        if params.func.is_null() {
            sys::cuKernelGetName(&mut name, params.kern)
        } else {
            sys::cuFuncGetName(&mut name, params.func)
        }
    };
    if result != sys::CUresult::CUDA_SUCCESS || name.is_null() {
        return "<unnamed>".to_owned();
    }

    // SAFETY: the driver returned a nul-terminated name it owns
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

fn close_capture(frame: Frame) {
    let Some(capture) = frame.capture else {
        return;
    };

    let added: BTreeSet<usize> = graph_nodes(capture.graph)
        .difference(&capture.before)
        .copied()
        .collect();
    if frame.kind == Kind::Projection {
        PROJECTION_NODES.with(|nodes| nodes.borrow_mut().nodes.extend(added));
        return;
    }

    // only Library controls may omit their projection nodes from boundary evidence
    let permitted = if frame.kind == Kind::Candidate {
        BTreeSet::new()
    } else {
        PROJECTION_NODES.with(|nodes| nodes.borrow().nodes.clone())
    };
    let mut kernels = Vec::new();
    let mut violations = Vec::new();
    for node in added.difference(&permitted) {
        match describe(*node) {
            Node::Kernel(name) => {
                let refused = match frame.kind {
                    Kind::Candidate => !allowed(&name),
                    _ => candidate_area_entry(&name),
                };
                if refused {
                    violations.push(format!("kernel {name}"));
                }
                kernels.push(name);
            }
            Node::DeviceCopy | Node::Memset => {}
            Node::HostCopy if frame.kind == Kind::Candidate => {
                violations.push("host copy".to_owned())
            }
            Node::HostCopy => {}
            Node::Other(kind) if frame.kind == Kind::Candidate => {
                violations.push(format!("node {kind}"))
            }
            Node::Other(_) => {}
        }
    }

    let label = frame.kind.label();
    let case = LABEL.with(|cell| cell.borrow().clone());
    GRAPH_EVIDENCE.with(|evidence| {
        evidence
            .borrow_mut()
            .push(json!({"scope": label, "name": frame.name, "kernels": kernels, "case": case}))
    });
    GRAPH_VIOLATIONS.with(|all| {
        all.borrow_mut().extend(
            violations
                .into_iter()
                .map(|violation| format!("{label} {}: {violation}", frame.name)),
        )
    });
}

/// One PTX module this process loaded, with the exact bytes' hash and entry names
#[derive(Debug, Clone)]
struct LoadedModule {
    area: String,
    tier: String,
    sha256: String,
    entries: Vec<String>,
}

fn modules() -> &'static Mutex<Vec<LoadedModule>> {
    static MODULES: OnceLock<Mutex<Vec<LoadedModule>>> = OnceLock::new();
    MODULES.get_or_init(|| Mutex::new(Vec::new()))
}

/// The `.entry` names of a PTX module
pub(crate) fn entries(ptx: &str) -> Vec<String> {
    ptx.split(".entry")
        .skip(1)
        .filter_map(|rest| {
            let name: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '$')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Records the exact PTX bytes a locked loader hands to the driver
pub(crate) fn record_module(area: &str, tier: &str, ptx: &str) {
    let module = LoadedModule {
        area: area.to_owned(),
        tier: tier.to_owned(),
        sha256: format!("{:x}", Sha256::digest(ptx.as_bytes())),
        entries: entries(ptx),
    };
    let mut loaded = modules().lock().unwrap_or_else(PoisonError::into_inner);
    if !loaded
        .iter()
        .any(|known| known.area == module.area && known.sha256 == module.sha256)
    {
        loaded.push(module);
    }
}

/// Every recorded module, for the result and the harness allow-list
pub(crate) fn loaded_modules() -> Value {
    let loaded = modules().lock().unwrap_or_else(PoisonError::into_inner);
    Value::Array(
        loaded
            .iter()
            .map(|module| {
                json!({"area": module.area, "tier": module.tier, "sha256": module.sha256, "entries": module.entries})
            })
            .collect(),
    )
}

fn allowed(name: &str) -> bool {
    let loaded = modules().lock().unwrap_or_else(PoisonError::into_inner);
    loaded
        .iter()
        .any(|module| module.entries.iter().any(|entry| entry == name))
}

/// The candidate PTX areas; their entries never run on a Library path
pub(crate) const CANDIDATE_AREAS: [&str; 3] = ["resnet", "lstm", "sincnet"];

fn candidate_area_entry(name: &str) -> bool {
    let loaded = modules().lock().unwrap_or_else(PoisonError::into_inner);
    loaded.iter().any(|module| {
        CANDIDATE_AREAS.contains(&module.area.as_str())
            && module.entries.iter().any(|entry| entry == name)
    })
}

/// Loads harness PTX through the recorder, so its entries join the allow-list
fn load_recorded(
    runtime: &CudaRuntime,
    area: &str,
    source: &str,
) -> Result<std::sync::Arc<CudaModule>, CudaError> {
    record_module(area, "sm75", source);
    Ok(runtime
        .context()
        .load_module(Ptx::from_src(source.to_owned()))?)
}

/// Persistent buffers and functions are prepared before CUDA graph capture
struct MutationKernels {
    round: CudaFunction,
    pool: CudaFunction,
    fault: CudaFunction,
    clear: CudaFunction,
    atomic: CudaFunction,
    add: CudaFunction,
    poison: CudaFunction,
    perturb: CudaFunction,
    poison_bytes: u32,
    poison_blocks: u32,
    unlisted: CudaFunction,
    uninit_shared: Option<CudaFunction>,
    bins: CudaSlice<f32>,
}

thread_local! {
    static DEVICE: RefCell<Option<MutationKernels>> = const { RefCell::new(None) };
}

/// A kernel no recorded module contains; only the `Unlisted` mutant launches it
const UNLISTED_PTX: &str = ".version 6.3\n.target sm_75\n.address_size 64\n\n.visible .entry qualify_unlisted()\n{\n\tret;\n}\n";

/// Loads only the locked test PTX, with no path selected by candidate code
pub(crate) fn prepare(runtime: &CudaRuntime) -> Result<(), CudaError> {
    DEVICE.with(|cell| {
        let mut state = cell.borrow_mut();
        if state.is_some() {
            return Ok(());
        }

        let faults = load_recorded(
            runtime,
            "qualify",
            include_str!("../../../tests/cuda_qualify/device/qualify.sm75.ptx"),
        )?;
        let controls = load_recorded(
            runtime,
            "controls",
            include_str!("../../../tests/cuda_qualify/device/controls.sm75.ptx"),
        )?;
        // deliberately not recorded: the profile rule must refuse this name
        let unlisted = runtime
            .context()
            .load_module(Ptx::from_src(UNLISTED_PTX.to_owned()))?;

        let context = runtime.context();
        let optin = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?;
        let sms =
            context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let poison = controls.load_function("qualify_poison")?;
        poison.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            optin,
        )?;
        poison.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            100,
        )?;
        *state = Some(MutationKernels {
            round: faults.load_function("qualify_round")?,
            pool: faults.load_function("qualify_pool")?,
            fault: faults.load_function("qualify_fault")?,
            clear: faults.load_function("qualify_clear")?,
            atomic: faults.load_function("qualify_atomic")?,
            add: faults.load_function("qualify_add")?,
            poison,
            perturb: controls.load_function("qualify_perturb")?,
            poison_bytes: u32::try_from(optin).unwrap_or(0),
            // two blocks per SM, so every SM runs at least one even if one is busy
            poison_blocks: u32::try_from(2 * sms.max(1)).unwrap_or(2),
            unlisted: unlisted.load_function("qualify_unlisted")?,
            uninit_shared: load_uninit_shared(runtime)?,
            bins: runtime.stream().alloc_zeros(256)?,
        });
        Ok(())
    })
}

/// Loads the uninitialized-shared entry only for its planted fault, so every other run
/// passes the shared-initialization PTX check on its loaded modules
fn load_uninit_shared(runtime: &CudaRuntime) -> Result<Option<CudaFunction>, CudaError> {
    if std::env::var("SPEAKRS_QUALIFY_IMPL").as_deref() != Ok("UninitShared") {
        return Ok(None);
    }

    let module = load_recorded(
        runtime,
        "uninit_shared",
        include_str!("../../../tests/cuda_qualify/device/uninit_shared.sm75.ptx"),
    )?;
    Ok(Some(module.load_function("qualify_uninit_shared")?))
}

/// Fills the largest dynamic shared-memory allocation on every SM with a NaN pattern
/// before a candidate launch in the numeric phase
///
/// Best effort: the hardware does not promise that a later kernel sees this residue,
/// because blocks may land on other SMs or carve-out sizes may change, so a read of
/// never-written shared memory is likely but not certain to turn the output into NaN.
/// Shared-memory initialization therefore stays an explicit kernel-audit item
pub(crate) fn poison(runtime: &CudaRuntime) -> Result<(), CudaError> {
    if phase() != Some("numeric") {
        return Ok(());
    }

    DEVICE.with(|cell| {
        let state = cell.borrow();
        let k = state
            .as_ref()
            .expect("prepare test kernels before poisoning");
        let words = k.poison_bytes / 4;
        let mut launch = runtime.stream().launch_builder(&k.poison);
        launch.arg(&words);
        // SAFETY: each thread writes only dynamic shared words below `words`, which
        // fit the requested dynamic shared allocation
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (k.poison_blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: k.poison_bytes,
            })
        }?;
        Ok(())
    })
}

thread_local! {
    static BAND: RefCell<Option<(u32, Vec<String>)>> = const { RefCell::new(None) };
}

/// Turns the TF32 stage noise band on with a seed and the declared layers, or off
pub(crate) fn set_band(band: Option<(u32, Vec<String>)>) {
    BAND.with(|cell| *cell.borrow_mut() = band);
}

/// Moves a seeded random two thirds of a Library output by one unit in the last place,
/// when the noise band is on and lists `layer`
pub(crate) fn perturb<Y: DevicePtrMut<f32>>(
    runtime: &CudaRuntime,
    layer: &str,
    output: &mut Y,
) -> Result<(), CudaError> {
    let seed = BAND.with(|cell| {
        let band = cell.borrow();
        let (seed, layers) = band.as_ref()?;
        layers.iter().any(|name| name == layer).then(|| {
            // FNV-1a of the layer name keeps each layer's pattern independent
            layer.bytes().fold(*seed ^ 0x811c_9dc5, |hash, byte| {
                (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
            })
        })
    });
    let Some(seed) = seed else {
        return Ok(());
    };

    DEVICE.with(|cell| {
        let state = cell.borrow();
        let k = state
            .as_ref()
            .expect("prepare test kernels before the noise band");
        let len = output.len() as u64;
        let count = u32::try_from(output.len()).expect("noise band output fits a grid");
        let (pointer, _record) = output.device_ptr_mut(runtime.stream());
        let mut launch = runtime.stream().launch_builder(&k.perturb);
        launch.arg(&pointer).arg(&len).arg(&seed);
        // SAFETY: one guarded thread per element of the exact output slice
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }?;
        Ok(())
    })
}

/// The precision mutant changes the real input in place; the driver restores it
/// before every isolated numeric run, outside the measured operator interval
pub(crate) fn round_input<X: DevicePtr<f32>>(
    runtime: &CudaRuntime,
    input: &X,
) -> Result<(), CudaError> {
    DEVICE.with(|cell| {
        let state = cell.borrow();
        let k = state.as_ref().expect("prepare test kernels before capture");
        let len = input.len() as u64;
        let mut launch = runtime.stream().launch_builder(&k.round);
        let (pointer, _record) = input.device_ptr(runtime.stream());
        launch.arg(&pointer).arg(&len);
        // SAFETY: the test-owned device input is not accessed on another stream,
        // each thread owns one element, and the caller restores it between runs
        unsafe { launch.launch(LaunchConfig::for_num_elems(input.len() as u32)) }?;
        Ok(())
    })
}

/// Reads the pre-normalization pooled boundary without changing the library path
pub(crate) fn pool(
    runtime: &CudaRuntime,
    input: &CudaSlice<f32>,
    in_len: usize,
    output: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    DEVICE.with(|cell| {
        let state = cell.borrow();
        let k = state.as_ref().expect("prepared kernels");
        let lengths = [input.len() as u64, output.len() as u64];
        let in_len = in_len as u64;
        let count = output.len() as u32;
        let mut launch = runtime.stream().launch_builder(&k.pool);
        launch
            .arg(input)
            .arg(&lengths[0])
            .arg(&in_len)
            .arg(output)
            .arg(&lengths[1]);
        // SAFETY: the driver checks complete NCW input and pooled output shapes
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }?;
        Ok(())
    })
}

/// Applies live device faults to the real layer output
pub(crate) fn post<Y: DevicePtrMut<f32>>(
    runtime: &CudaRuntime,
    output: &mut Y,
    batch: usize,
    mutant: Mutant,
) -> Result<(), CudaError> {
    DEVICE.with(|cell| {
        let mut state = cell.borrow_mut();
        let k = state.as_mut().expect("prepared kernels");
        let count = output.len() as u32;
        let len = output.len() as u64;
        let (pointer, _record) = output.device_ptr_mut(runtime.stream());
        if matches!(mutant, Mutant::Shape | Mutant::Tail) {
            let mode = if mutant == Mutant::Shape { 1u32 } else { 2u32 };
            let batch = batch as u32;
            let mut launch = runtime.stream().launch_builder(&k.fault);
            launch.arg(&mode).arg(&batch).arg(&pointer).arg(&len);
            // SAFETY: one guarded thread per output element, exact slice lengths
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }?;
        }
        if mutant == Mutant::Atomic {
            let bins_len = k.bins.len() as u64;
            let mut clear = runtime.stream().launch_builder(&k.clear);
            clear.arg(&mut k.bins).arg(&bins_len);
            // SAFETY: bins has exactly 256 initialized and uniquely owned elements
            unsafe { clear.launch(LaunchConfig::for_num_elems(256)) }?;
            let mut atomic = runtime.stream().launch_builder(&k.atomic);
            atomic.arg(&mut k.bins).arg(&bins_len);
            // SAFETY: all concurrent accesses use device-scope FP32 atomics
            unsafe {
                atomic.launch(LaunchConfig {
                    grid_dim: (1023, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            let mut add = runtime.stream().launch_builder(&k.add);
            add.arg(&k.bins).arg(&bins_len).arg(&pointer).arg(&len);
            // SAFETY: one guarded writer per output element, atomics finished first
            unsafe { add.launch(LaunchConfig::for_num_elems(256)) }?;
        }
        if mutant == Mutant::UninitShared {
            let kernel = k.uninit_shared.as_ref().expect("planted entry loaded");
            let mut launch = runtime.stream().launch_builder(kernel);
            launch.arg(&mut k.bins);
            // SAFETY: one thread writes the first word of the 256-element bins
            unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }?;
        }
        if mutant == Mutant::Unlisted {
            let mut launch = runtime.stream().launch_builder(&k.unlisted);
            // SAFETY: the kernel takes no parameters and touches no memory
            unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }?;
        }
        Ok(())
    })
}

thread_local! {
    static STASH: RefCell<Vec<(String, CudaSlice<f32>)>> = const { RefCell::new(Vec::new()) };
}

/// The `Lookup` mutant's table: the first buffer it saw under `key`, so later calls
/// answer from it instead of the data they were given
pub(crate) fn lookup<X: DevicePtr<f32>>(
    runtime: &CudaRuntime,
    key: &str,
    values: &X,
) -> Result<CudaSlice<f32>, CudaError> {
    let key = format!("{key}/{}", values.len());
    let found = STASH.with(|stash| {
        stash
            .borrow()
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, slice)| slice.clone())
    });
    if let Some(found) = found {
        return Ok(found);
    }

    let stream = runtime.stream();
    let mut copy = stream.alloc_zeros::<f32>(values.len())?;
    stream.memcpy_dtod(values, &mut copy)?;
    STASH.with(|stash| stash.borrow_mut().push((key, copy.clone())));
    Ok(copy)
}

/// A per-process random seed for the secret-input check: the standard library keys
/// each `RandomState` from the operating system, and nothing writes the value to a
/// path before the check runs
pub(crate) fn secret_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(u64::from(std::process::id()));
    hasher.finish()
}

/// The `Unscoped` mutant's extra launch, after its candidate scope has closed
pub(crate) fn unscoped(runtime: &CudaRuntime, mutant: Mutant) -> Result<(), CudaError> {
    if mutant != Mutant::Unscoped {
        return Ok(());
    }

    DEVICE.with(|cell| {
        let mut state = cell.borrow_mut();
        let k = state.as_mut().expect("prepared kernels");
        let bins_len = k.bins.len() as u64;
        let mut clear = runtime.stream().launch_builder(&k.clear);
        clear.arg(&mut k.bins).arg(&bins_len);
        // SAFETY: bins has exactly 256 initialized and uniquely owned elements
        unsafe { clear.launch(LaunchConfig::for_num_elems(256)) }?;
        Ok(())
    })
}

/// Positive controls for the sanitizer include filter, each in an isolated process
pub(crate) fn sanitizer_control(runtime: &CudaRuntime, control: &str) -> Result<(), CudaError> {
    let faults = load_recorded(
        runtime,
        "qualify",
        include_str!("../../../tests/cuda_qualify/device/qualify.sm75.ptx"),
    )?;
    let controls = load_recorded(
        runtime,
        "controls",
        include_str!("../../../tests/cuda_qualify/device/controls.sm75.ptx"),
    )?;
    let stream = runtime.stream();
    match control {
        "oob" => {
            let kernel = faults.load_function("qualify_oob")?;
            let mut output = stream.alloc_zeros::<f32>(257)?;
            let (pointer, _record) = output.device_ptr_mut(stream);
            let length = 257u64;
            let mut launch = stream.launch_builder(&kernel);
            launch.arg(&pointer).arg(&length);
            println!("sanitizer control: qualify_oob allocation=257 write-index=257");
            // SAFETY: deliberate one-past-allocation write, isolated from every case
            unsafe { launch.launch(LaunchConfig::for_num_elems(512)) }?;
        }
        "race" => {
            let kernel = controls.load_function("qualify_race")?;
            let mut output = stream.alloc_zeros::<u32>(64)?;
            let mut launch = stream.launch_builder(&kernel);
            launch.arg(&mut output);
            println!("sanitizer control: qualify_race two warps store one shared word");
            // SAFETY: deliberate shared-memory race; each thread writes its own output
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
        }
        "uninit" => {
            let kernel = faults.load_function("qualify_round")?;
            // SAFETY: deliberately uninitialized; the control reads it once
            let values = unsafe { stream.alloc::<f32>(256) }?;
            let (pointer, _record) = values.device_ptr(stream);
            let length = 256u64;
            let mut launch = stream.launch_builder(&kernel);
            launch.arg(&pointer).arg(&length);
            println!("sanitizer control: qualify_round reads 256 uninitialized floats");
            // SAFETY: in bounds; the read of unset memory is the planted fault
            unsafe { launch.launch(LaunchConfig::for_num_elems(256)) }?;
        }
        other => panic!("unknown sanitizer control {other}"),
    }

    runtime.synchronize()
}

mod tests {
    use super::{CALL_VIOLATIONS, Frame, Kind, STACK, check_call, entries};

    #[test]
    fn entry_names_come_from_the_loaded_bytes() {
        let ptx = ".visible .entry first(\n.param .u64 a)\n{}\n.entry  second_2()\n{}";
        assert_eq!(entries(ptx), ["first", "second_2"]);
    }

    #[test]
    fn projections_cannot_hide_candidate_library_calls() {
        let saved_stack = STACK.with(|stack| stack.take());
        let saved_violations = CALL_VIOLATIONS.with(|violations| violations.take());
        for kind in [Kind::Library, Kind::Candidate] {
            STACK.with(|stack| {
                *stack.borrow_mut() = vec![
                    Frame {
                        kind,
                        name: "lstm.stack".to_owned(),
                        capture: None,
                    },
                    Frame {
                        kind: Kind::Projection,
                        name: "L0.forward".to_owned(),
                        capture: None,
                    },
                ];
            });
            check_call("cublas.m589.n512.k60");
        }

        let actual = CALL_VIOLATIONS.with(|violations| violations.replace(saved_violations));
        STACK.with(|stack| stack.replace(saved_stack));
        assert_eq!(actual, ["cublas.m589.n512.k60 inside candidate lstm.stack"]);
    }
}
