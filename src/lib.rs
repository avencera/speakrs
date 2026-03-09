#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod aggregate;
pub mod binarize;
pub mod clustering;
pub mod inference;
pub mod metrics;
#[cfg(feature = "online")]
pub mod models;
pub mod pipeline;
pub mod powerset;
pub mod reconstruct;
pub mod segment;
pub mod utils;
