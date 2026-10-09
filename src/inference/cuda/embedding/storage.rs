//! Monotonic activation capacity and captured-pointer validity

use std::sync::Arc;

use tracing::debug;

use super::batch_class::EmbeddingBatchClass;

/// One allocation whose generation changes only after successful growth
#[derive(Debug)]
pub(super) struct ActivationStorage<B> {
    capacity: usize,
    generation: Arc<()>,
    buffers: B,
}

impl<B> ActivationStorage<B> {
    /// Allocate one window, or reserve all classes for a multi-window session
    pub(super) fn allocate<E>(
        requested: usize,
        allocate: impl FnOnce(usize) -> Result<B, E>,
    ) -> Result<Self, E> {
        let capacity = Self::capacity_for(requested);
        let buffers = allocate(capacity)?;
        debug!(
            requested,
            capacity, "Allocated CUDA embedding activation storage"
        );
        Ok(Self {
            capacity,
            generation: Arc::new(()),
            buffers,
        })
    }

    fn capacity_for(requested: usize) -> usize {
        if requested <= 1 {
            return requested;
        }

        // library graph warm-up and recapture must not repeat for each larger class
        requested.max(EmbeddingBatchClass::ThirtyTwo.chunks())
    }

    /// Replace storage only for a larger request; failure preserves captured pointers
    ///
    /// The allocator must finish prior device work before it replaces old buffers
    pub(super) fn grow<E>(
        &mut self,
        requested: usize,
        allocate: impl FnOnce(usize) -> Result<B, E>,
    ) -> Result<(), E> {
        if requested <= self.capacity {
            return Ok(());
        }

        let capacity = Self::capacity_for(requested);
        let buffers = allocate(capacity)?;
        let generation = Arc::new(());
        let old_capacity = self.capacity;
        self.buffers = buffers;
        self.capacity = capacity;
        self.generation = generation;
        debug!(
            requested,
            old_capacity, capacity, "Grew CUDA embedding activation storage"
        );
        Ok(())
    }

    /// Current allocation for capacity validation
    pub(super) fn buffers(&self) -> &B {
        &self.buffers
    }

    /// Exclusive access to the current allocation for recording or eager execution
    pub(super) fn buffers_mut(&mut self) -> &mut B {
        &mut self.buffers
    }
}

/// A captured value that cannot be retrieved with a stale storage generation
pub(super) struct Captured<G> {
    generation: Arc<()>,
    value: G,
}

impl<G> Captured<G> {
    /// Bind a captured value to the allocation used to record it
    pub(super) fn new<B>(value: G, storage: &ActivationStorage<B>) -> Self {
        Self {
            generation: Arc::clone(&storage.generation),
            value,
        }
    }

    /// Return the value only while its recorded pointers are current
    pub(super) fn current<B>(&self, storage: &ActivationStorage<B>) -> Option<&G> {
        Arc::ptr_eq(&self.generation, &storage.generation).then_some(&self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::{ActivationStorage, Captured};
    use crate::inference::cuda::EmbeddingBatchClass;

    const BYTES_PER_CHUNK: usize = 35_768_320;

    fn storage(requested: usize) -> ActivationStorage<usize> {
        ActivationStorage::allocate(requested, |capacity| {
            Ok::<_, ()>(capacity * BYTES_PER_CHUNK)
        })
        .unwrap()
    }

    #[test]
    fn single_windows_stay_small_and_growth_invalidates_all_old_graphs() {
        let mut storage = storage(1);
        let one = Captured::new("one", &storage);
        for _ in 0..3 {
            storage
                .grow(1, |_| -> Result<usize, ()> {
                    panic!("single-window storage grew")
                })
                .unwrap();
            assert_eq!(*storage.buffers(), 35_768_320);
            assert_eq!(one.current(&storage), Some(&"one"));
        }

        let mut growths = 0;
        storage
            .grow(4, |chunks| {
                growths += 1;
                Ok::<_, ()>(chunks * BYTES_PER_CHUNK)
            })
            .unwrap();
        let four = Captured::new("four", &storage);
        assert_eq!(one.current(&storage), None);
        assert_eq!(four.current(&storage), Some(&"four"));
        assert_eq!(*storage.buffers(), 1_144_586_240);
        for chunks in [8, 16, 32, 1] {
            storage
                .grow(chunks, |_| -> Result<usize, ()> {
                    panic!("multi-window storage grew again")
                })
                .unwrap();
            assert_eq!(*storage.buffers(), 1_144_586_240);
            assert_eq!(one.current(&storage), None);
            assert_eq!(four.current(&storage), Some(&"four"));
        }
        assert_eq!(growths, 1);
        let recaptured_one = Captured::new("one", &storage);
        assert_eq!(recaptured_one.current(&storage), Some(&"one"));
    }

    #[test]
    fn every_initial_multi_window_class_reserves_the_maximum() {
        for class in EmbeddingBatchClass::ALL.into_iter().skip(1) {
            let mut storage = storage(class.chunks());
            let graph = Captured::new("multi-window", &storage);
            assert_eq!(storage.capacity, 32);
            assert_eq!(*storage.buffers(), 1_144_586_240);
            storage
                .grow(32, |_| -> Result<usize, ()> {
                    panic!("initial multi-window storage grew")
                })
                .unwrap();
            assert_eq!(graph.current(&storage), Some(&"multi-window"));
        }
    }

    #[test]
    fn thirty_two_then_thirty_one_retains_a_concrete_allocation_bound() {
        let mut storage = storage(32);
        let graph = Captured::new("thirty-two", &storage);
        let mut remaining = 31;
        while let Some(class) = EmbeddingBatchClass::fitting(remaining) {
            storage
                .grow(class.chunks(), |_| -> Result<usize, ()> {
                    panic!("smaller class allocated")
                })
                .unwrap();
            remaining -= class.chunks();
        }
        assert_eq!(remaining, 0);
        assert_eq!(storage.capacity, 32);
        assert_eq!(*storage.buffers(), 1_144_586_240);
        assert_eq!(graph.current(&storage), Some(&"thirty-two"));
    }

    #[test]
    fn a_graph_cannot_replay_on_a_different_storage_owner() {
        let first = storage(1);
        let other = storage(1);
        let graph = Captured::new("one", &first);
        assert_eq!(graph.current(&first), Some(&"one"));
        assert_eq!(graph.current(&other), None);
    }

    #[test]
    fn failed_growth_keeps_capacity_buffers_and_graphs_valid() {
        let mut storage = storage(1);
        let graph = Captured::new("one", &storage);
        assert_eq!(
            storage.grow(32, |_| Err("out of memory")),
            Err("out of memory")
        );
        assert_eq!(storage.capacity, 1);
        assert_eq!(*storage.buffers(), 35_768_320);
        assert_eq!(graph.current(&storage), Some(&"one"));
    }
}
