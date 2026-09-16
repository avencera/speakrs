use std::cell::Cell;

/// Test-only chunk schedule used to force sequential or pipelined execution
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ForcedChunkSchedule {
    Sequential,
    Pipelined,
}

thread_local! {
    static FORCED_CHUNK_SCHEDULE: Cell<Option<ForcedChunkSchedule>> =
        const { Cell::new(None) };
}

pub(crate) fn force_chunk_schedule(schedule: Option<ForcedChunkSchedule>) {
    FORCED_CHUNK_SCHEDULE.with(|cell| cell.set(schedule));
}

pub(crate) fn forced_chunk_schedule() -> Option<ForcedChunkSchedule> {
    FORCED_CHUNK_SCHEDULE.with(Cell::get)
}

pub(crate) fn select_chunk_schedule(estimated_chunks: usize) -> bool {
    match forced_chunk_schedule() {
        Some(ForcedChunkSchedule::Sequential) => false,
        Some(ForcedChunkSchedule::Pipelined) => true,
        None => estimated_chunks >= 2,
    }
}

#[cfg(test)]
mod tests {
    use super::{ForcedChunkSchedule, force_chunk_schedule, select_chunk_schedule};

    #[test]
    fn forced_schedule_reaches_distinct_paths() {
        force_chunk_schedule(Some(ForcedChunkSchedule::Sequential));
        assert!(!select_chunk_schedule(8));
        force_chunk_schedule(Some(ForcedChunkSchedule::Pipelined));
        assert!(select_chunk_schedule(1));
        force_chunk_schedule(None);
        assert!(select_chunk_schedule(2));
        assert!(!select_chunk_schedule(1));
    }
}
