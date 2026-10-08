//! Typed model boundaries, separate from the modules and configurations that implement
//! them
//!
//! A boundary is a model operation such as `resnet.layer1.0.conv1`. Its name is the
//! stable string that logs and tune files use. Which candidate module implements
//! it, if any, comes only from a production binding, never from the name

use std::fmt;

use crate::inference::cuda::KernelModule;

/// Batch classes production may authorize for one boundary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProductionBatches {
    /// The model batch classes 1 and 32; stress batches never grant production
    Model,
    /// Exact embedding trunk classes, independent of segmentation and dense heads
    Embedding,
    /// Every filterbank batch from 1 to 32
    Fbank,
}

impl ProductionBatches {
    /// The model batch classes
    pub(crate) const MODEL: [usize; 2] = [1, 32];

    /// Enumerate only batches supported by this model boundary
    pub(crate) fn iter(self) -> impl Iterator<Item = usize> {
        (1..=32).filter(move |batch| self.contains(*batch))
    }

    pub(crate) const fn contains(self, batch: usize) -> bool {
        match self {
            Self::Model => {
                let mut index = 0;
                while index < Self::MODEL.len() {
                    if Self::MODEL[index] == batch {
                        return true;
                    }
                    index += 1;
                }
                false
            }
            Self::Embedding => {
                let classes = super::super::embedding::EmbeddingBatchClass::ALL;
                let mut index = 0;
                while index < classes.len() {
                    if classes[index].chunks() == batch {
                        return true;
                    }
                    index += 1;
                }
                false
            }
            Self::Fbank => matches!(batch, 1..=32),
        }
    }
}

/// One row of the boundary table
#[derive(Debug)]
struct Boundary {
    name: &'static str,
    /// The area named in Library diagnostics; never a candidate route
    area: KernelModule,
    batches: ProductionBatches,
}

const fn model(name: &'static str, area: KernelModule) -> Boundary {
    Boundary {
        name,
        area,
        batches: ProductionBatches::Model,
    }
}

const fn resnet(name: &'static str) -> Boundary {
    Boundary {
        name,
        area: KernelModule::Resnet,
        batches: ProductionBatches::Embedding,
    }
}

/// Every model boundary that selection or a Library requirement can name
const BOUNDARIES: &[Boundary] = &[
    resnet("resnet.conv1"),
    resnet("resnet.layer1.0.conv1"),
    resnet("resnet.layer1.0.conv2"),
    resnet("resnet.layer1.1.conv1"),
    resnet("resnet.layer1.1.conv2"),
    resnet("resnet.layer1.2.conv1"),
    resnet("resnet.layer1.2.conv2"),
    resnet("resnet.layer2.0.conv1"),
    resnet("resnet.layer2.0.conv2"),
    resnet("resnet.layer2.0.shortcut.0"),
    resnet("resnet.layer2.1.conv1"),
    resnet("resnet.layer2.1.conv2"),
    resnet("resnet.layer2.2.conv1"),
    resnet("resnet.layer2.2.conv2"),
    resnet("resnet.layer2.3.conv1"),
    resnet("resnet.layer2.3.conv2"),
    resnet("resnet.layer3.0.conv1"),
    resnet("resnet.layer3.0.conv2"),
    resnet("resnet.layer3.0.shortcut.0"),
    resnet("resnet.layer3.1.conv1"),
    resnet("resnet.layer3.1.conv2"),
    resnet("resnet.layer3.2.conv1"),
    resnet("resnet.layer3.2.conv2"),
    resnet("resnet.layer3.3.conv1"),
    resnet("resnet.layer3.3.conv2"),
    resnet("resnet.layer3.4.conv1"),
    resnet("resnet.layer3.4.conv2"),
    resnet("resnet.layer3.5.conv1"),
    resnet("resnet.layer3.5.conv2"),
    resnet("resnet.layer4.0.conv1"),
    resnet("resnet.layer4.0.conv2"),
    resnet("resnet.layer4.0.shortcut.0"),
    resnet("resnet.layer4.1.conv1"),
    resnet("resnet.layer4.1.conv2"),
    resnet("resnet.layer4.2.conv1"),
    resnet("resnet.layer4.2.conv2"),
    model("resnet.seg_1", KernelModule::Embedding),
    model("sincnet.conv0.abs_pool", KernelModule::Sincnet),
    model("sincnet.conv1", KernelModule::Segmentation),
    model("sincnet.conv2", KernelModule::Segmentation),
    model("lstm.stack", KernelModule::Lstm),
    // the cuBLAS projections nested inside the pinned PR #36 stack
    model("lstm.stack.input_proj", KernelModule::Lstm),
    model("linear0", KernelModule::Segmentation),
    model("linear1", KernelModule::Segmentation),
    model("linear2", KernelModule::Segmentation),
    Boundary {
        name: "fbank.dft",
        area: KernelModule::Fbank,
        batches: ProductionBatches::Fbank,
    },
];

const _: () = {
    assert!(BOUNDARIES.len() <= u8::MAX as usize);
    let mut index = 0;
    while index < BOUNDARIES.len() {
        let mut other = index + 1;
        while other < BOUNDARIES.len() {
            assert!(
                !same(BOUNDARIES[index].name, BOUNDARIES[other].name),
                "boundary names must be unique"
            );
            other += 1;
        }
        index += 1;
    }
};

/// Byte equality of two names, usable in compile-time table validation
pub(super) const fn same(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// A model operation from the fixed boundary table
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct BoundaryId(u8);

impl BoundaryId {
    /// A fixed name; an unknown name fails compilation in constants
    pub(crate) const fn named(name: &str) -> Self {
        match Self::find(name) {
            Some(id) => id,
            None => panic!("unknown CUDA boundary name"),
        }
    }

    /// Resolve a name from outside the table, such as a harness argument
    pub(crate) fn parse(name: &str) -> Result<Self, UnknownBoundary> {
        Self::find(name).ok_or_else(|| UnknownBoundary(name.to_owned()))
    }

    const fn find(name: &str) -> Option<Self> {
        let mut index = 0;
        while index < BOUNDARIES.len() {
            if same(BOUNDARIES[index].name, name) {
                return Some(Self(index as u8));
            }
            index += 1;
        }
        None
    }

    /// Identity usable in compile-time table validation
    pub(crate) const fn same(self, other: Self) -> bool {
        self.0 == other.0
    }

    const fn row(self) -> &'static Boundary {
        &BOUNDARIES[self.0 as usize]
    }

    /// The stable record and harness name
    pub(crate) const fn name(self) -> &'static str {
        self.row().name
    }

    /// The name as a one-element static slice, for declarative coverage entries
    #[cfg(test)]
    pub(crate) fn name_slice(self) -> &'static [&'static str] {
        std::slice::from_ref(&BOUNDARIES[self.0 as usize].name)
    }

    /// The area Library diagnostics report; selection never routes by it
    pub(crate) const fn area(self) -> KernelModule {
        self.row().area
    }

    /// The batch classes production may authorize
    pub(crate) const fn batches(self) -> ProductionBatches {
        self.row().batches
    }

    /// Every boundary, in table order
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        (0..BOUNDARIES.len()).map(|index| Self(index as u8))
    }
}

impl fmt::Debug for BoundaryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for BoundaryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A boundary name outside the fixed table
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown CUDA model boundary `{0}`")]
pub(crate) struct UnknownBoundary(pub(crate) String);

#[cfg(test)]
mod tests {
    use super::{BoundaryId, ProductionBatches, UnknownBoundary};
    use crate::inference::cuda::KernelModule;

    #[test]
    fn names_round_trip_and_unknown_names_fail_typed() {
        for id in BoundaryId::all() {
            assert_eq!(BoundaryId::parse(id.name()), Ok(id));
            assert_eq!(id.name_slice(), [id.name()]);
        }
        for unknown in [
            "",
            "unknown",
            "resnet.layer5.0.conv1",
            "resnet.layer1.0.conv1 ",
        ] {
            assert_eq!(
                BoundaryId::parse(unknown),
                Err(UnknownBoundary(unknown.to_owned()))
            );
        }
    }

    #[test]
    fn batch_sets_are_per_boundary() {
        let lstm = BoundaryId::named("lstm.stack");
        let fbank = BoundaryId::named("fbank.dft");
        let trunk = BoundaryId::named("resnet.conv1");
        let head = BoundaryId::named("resnet.seg_1");
        assert_eq!(lstm.batches(), ProductionBatches::Model);
        assert_eq!(fbank.batches(), ProductionBatches::Fbank);
        for batch in 0..=64 {
            assert_eq!(lstm.batches().contains(batch), [1, 32].contains(&batch));
            assert_eq!(fbank.batches().contains(batch), (1..=32).contains(&batch));
            assert_eq!(
                trunk.batches().contains(batch),
                [1, 4, 8, 16, 32].contains(&batch)
            );
            assert_eq!(head.batches().contains(batch), [1, 32].contains(&batch));
        }
    }

    #[test]
    fn diagnostic_areas_do_not_follow_name_prefixes() {
        // the segmentation convolutions carry SincNet names but no SincNet route
        assert_eq!(
            BoundaryId::named("sincnet.conv1").area(),
            KernelModule::Segmentation
        );
        assert_eq!(
            BoundaryId::named("resnet.seg_1").area(),
            KernelModule::Embedding
        );
        assert_eq!(
            BoundaryId::named("linear0").area(),
            KernelModule::Segmentation
        );
    }
}
