//! Immutable facts about the device of one runtime context

use std::num::NonZeroU32;

use cudarc::driver::CudaContext;
use cudarc::driver::sys::CUdevice_attribute;

use super::{ComputeCapability, CudaError};

/// Device facts queried once per runtime
///
/// Selection and planning read these cached values, so a plan cannot observe
/// attributes that differ from the ones its selection used. The only constructors are
/// [`Self::query`] and the test-support builder
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceAttributes {
    capability: ComputeCapability,
    multiprocessors: NonZeroU32,
    l2_bytes: u32,
    shared_optin_bytes: u32,
    name: Box<str>,
}

impl DeviceAttributes {
    /// Query the context's device; the context must be the runtime's own
    pub(super) fn query(context: &CudaContext) -> Result<Self, CudaError> {
        let (major, minor) = context.compute_capability()?;
        let capability = ComputeCapability::new(
            u32::try_from(major).unwrap_or(0),
            u32::try_from(minor).unwrap_or(0),
        );
        // under per-context MPS partitioning the attribute follows the current context
        context.bind_to_thread()?;
        let attribute = |attribute| -> Result<u32, CudaError> {
            Ok(u32::try_from(context.attribute(attribute)?).unwrap_or(0))
        };
        let multiprocessors = NonZeroU32::new(attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )?)
        .ok_or_else(|| CudaError::Unsupported {
            context: "device attributes",
            reason: "the driver reports no streaming multiprocessors".to_owned(),
        })?;

        Ok(Self {
            capability,
            multiprocessors,
            l2_bytes: attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
            shared_optin_bytes: attribute(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?,
            name: context.name()?.into_boxed_str(),
        })
    }

    /// Exact compute capability
    pub(crate) const fn capability(&self) -> ComputeCapability {
        self.capability
    }

    /// Streaming multiprocessors this context may use: the device's, or the client's
    /// share under MPS active-thread limits
    pub(crate) const fn multiprocessors(&self) -> NonZeroU32 {
        self.multiprocessors
    }

    /// L2 cache capacity in bytes
    pub(crate) const fn l2_bytes(&self) -> u32 {
        self.l2_bytes
    }

    /// Largest dynamic shared memory one block may opt in to, in bytes
    pub(crate) const fn shared_optin_bytes(&self) -> u32 {
        self.shared_optin_bytes
    }

    /// Driver-reported device name
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::num::NonZeroU32;

    use super::DeviceAttributes;
    use crate::inference::cuda::ComputeCapability;

    /// Builds attributes for host-only tests; production code cannot invent a device
    #[derive(Debug, Clone)]
    pub(crate) struct Builder(DeviceAttributes);

    impl Builder {
        /// A device with representative defaults for every attribute except the ones set
        pub(crate) fn new(capability: ComputeCapability) -> Self {
            Self(DeviceAttributes {
                capability,
                multiprocessors: NonZeroU32::MIN,
                l2_bytes: 4 << 20,
                shared_optin_bytes: 99 << 10,
                name: "test device".into(),
            })
        }

        pub(crate) fn multiprocessors(mut self, count: u32) -> Self {
            self.0.multiprocessors = NonZeroU32::new(count).expect("a device has SMs");
            self
        }

        pub(crate) fn shared_optin_bytes(mut self, bytes: u32) -> Self {
            self.0.shared_optin_bytes = bytes;
            self
        }

        pub(crate) fn name(mut self, name: &str) -> Self {
            self.0.name = name.into();
            self
        }

        pub(crate) fn build(self) -> DeviceAttributes {
            self.0
        }
    }
}
