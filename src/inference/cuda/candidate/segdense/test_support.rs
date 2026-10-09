//! Forced segdense pins for development checks that time one kernel entry against
//! the library on the device in hand

use super::{Choice, Entry, Hardware, SegdensePin};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::{CudaMath, PtxTier};

impl SegdensePin {
    /// The batch-32 embedding pin forced to the entry named `name` (its `Debug` form),
    /// with the split count the device rule gives that entry on `device`
    pub(crate) fn forced_embed_b32(
        name: &str,
        math: CudaMath,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Option<Self> {
        let hardware = Hardware::of(device);
        let (entry, splits) = [
            (Entry::EmbedB32, hardware.splits(2, 2)),
            (Entry::EmbedB32Tf32, hardware.splits(2, 2)),
            (Entry::EmbedB32Tf32K2, hardware.splits(2, 1)),
            (Entry::EmbedB32Tf32E64, hardware.splits(4, 2)),
            (Entry::EmbedB32F16, hardware.splits(2, 1)),
        ]
        .into_iter()
        .find(|(entry, _)| format!("{entry:?}") == name)?;

        Some(Self {
            choice: Choice::split(entry, splits),
            math,
            tier,
        })
    }
}
