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
    use crate::pipeline::PipelineError;
    use crossbeam_channel::bounded;
    use std::time::{Duration, Instant};

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

    fn release_then_join<E, T>(
        endpoint: E,
        join: impl FnOnce() -> Result<T, PipelineError>,
    ) -> Result<T, PipelineError> {
        drop(endpoint);
        join()
    }

    #[test]
    fn filled_channel_joins_after_downstream_release() {
        let (tx, rx) = bounded::<u8>(1);
        tx.send(1).expect("fill the bounded channel");
        let start = Instant::now();
        std::thread::scope(|scope| {
            let handle = scope.spawn(move || {
                let _ = tx.send(2);
                let _ = tx.send(3);
                Ok::<(), PipelineError>(())
            });
            release_then_join(rx, || {
                handle.join().map_err(|_| PipelineError::WorkerPanic {
                    worker: "fault-injection producer".into(),
                })?
            })
            .expect("producer must join after downstream release");
        });
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "producer stayed blocked after downstream release"
        );
    }

    #[test]
    fn each_stage_release_unblocks_a_full_channel() {
        for bound in [1usize, 2, 4] {
            let (decoded_tx, decoded_rx) = bounded::<usize>(bound);
            let (prepared_tx, prepared_rx) = bounded::<usize>(bound);
            let (embedded_tx, embedded_rx) = bounded::<usize>(bound);
            let start = Instant::now();
            std::thread::scope(|scope| {
                let decoded_handle = scope.spawn(move || {
                    for item in 0..bound + 1 {
                        if decoded_tx.send(item).is_err() {
                            return Ok::<(), PipelineError>(());
                        }
                    }
                    Ok(())
                });
                let prep_handle = scope.spawn(move || {
                    while let Ok(item) = decoded_rx.recv() {
                        if prepared_tx.send(item).is_err() {
                            break;
                        }
                    }
                    Ok::<(), PipelineError>(())
                });
                let gpu_handle = scope.spawn(move || {
                    while let Ok(item) = prepared_rx.recv() {
                        if embedded_tx.send(item).is_err() {
                            break;
                        }
                    }
                    Ok::<(), PipelineError>(())
                });

                release_then_join(embedded_rx, || {
                    gpu_handle.join().map_err(|_| PipelineError::WorkerPanic {
                        worker: "gpu".into(),
                    })??;
                    prep_handle
                        .join()
                        .map_err(|_| PipelineError::WorkerPanic {
                            worker: "prep".into(),
                        })??;
                    decoded_handle
                        .join()
                        .map_err(|_| PipelineError::WorkerPanic {
                            worker: "decoded".into(),
                        })?
                })
                .expect("all stages must join after downstream failure");
            });
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "stage shutdown exceeded bound for capacity {bound}"
            );
        }
    }
}
