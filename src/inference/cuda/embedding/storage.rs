//! Monotonic activation capacity and captured-pointer validity

use std::sync::Arc;

/// One allocation whose generation changes only after successful growth
#[derive(Debug)]
pub(super) struct ActivationStorage<B> {
    capacity: usize,
    generation: Arc<()>,
    buffers: B,
}

impl<B> ActivationStorage<B> {
    /// Start with exactly the requested capacity
    pub(super) fn new(capacity: usize, buffers: B) -> Self {
        Self {
            capacity,
            generation: Arc::new(()),
            buffers,
        }
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

        let buffers = allocate(requested)?;
        let generation = Arc::new(());
        self.buffers = buffers;
        self.capacity = requested;
        self.generation = generation;
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

    #[test]
    fn single_windows_stay_small_and_growth_invalidates_all_old_graphs() {
        let mut storage = ActivationStorage::new(1, BYTES_PER_CHUNK);
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

        storage
            .grow(4, |chunks| Ok::<_, ()>(chunks * BYTES_PER_CHUNK))
            .unwrap();
        let four = Captured::new("four", &storage);
        assert_eq!(one.current(&storage), None);
        assert_eq!(four.current(&storage), Some(&"four"));
        assert_eq!(*storage.buffers(), 143_073_280);
        storage
            .grow(32, |chunks| Ok::<_, ()>(chunks * BYTES_PER_CHUNK))
            .unwrap();
        assert_eq!(*storage.buffers(), 1_144_586_240);
        assert_eq!(one.current(&storage), None);
        assert_eq!(four.current(&storage), None);
        let recaptured_one = Captured::new("one", &storage);
        assert_eq!(recaptured_one.current(&storage), Some(&"one"));
    }

    #[test]
    fn thirty_two_then_thirty_one_retains_a_concrete_allocation_bound() {
        let mut storage = ActivationStorage::new(32, 32 * BYTES_PER_CHUNK);
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
        let first = ActivationStorage::new(1, BYTES_PER_CHUNK);
        let other = ActivationStorage::new(1, BYTES_PER_CHUNK);
        let graph = Captured::new("one", &first);
        assert_eq!(graph.current(&first), Some(&"one"));
        assert_eq!(graph.current(&other), None);
    }

    #[test]
    fn failed_growth_keeps_capacity_buffers_and_graphs_valid() {
        let mut storage = ActivationStorage::new(1, BYTES_PER_CHUNK);
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
