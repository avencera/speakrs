//! Batch streams whose segmentation runs one file ahead of embedding
//!
//! Started per file, segmentation of the next file waits until the current file's
//! embedding has finished, so each file starts with the GPU mostly idle while the
//! first segmentation batch runs. Here one worker segments the files in order, and
//! the next file's segmentation follows the current one's at once while the current
//! file's last embedding batches still run. The window queue bounds how far ahead it
//! gets, so file results and their order are unchanged

use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, bounded};
use ndarray::Array2;

use super::batch::BatchPostProcessor;
use super::{BatchStreamError, CONCURRENT_WINDOW_QUEUE, PipelineRunner};
use crate::inference::segmentation::SegmentationError;
use crate::pipeline::PipelineError;

/// One file for the segmentation worker
struct SegmentationJob<A> {
    audio: Arc<A>,
    windows: Sender<Array2<f32>>,
    done: Sender<Result<usize, SegmentationError>>,
}

/// A file whose segmentation is queued, before its embedding runs
struct QueuedFile<S, A> {
    file_id: S,
    audio: Arc<A>,
    windows: Receiver<Array2<f32>>,
    done: Receiver<Result<usize, SegmentationError>>,
}

const WORKER: &str = "batch segmentation";

fn worker_panic() -> PipelineError {
    PipelineError::WorkerPanic {
        worker: WORKER.to_owned(),
    }
}

impl PipelineRunner<'_> {
    /// Runs concurrent inference over `files` with segmentation one file ahead, and
    /// submits each file's artifacts to `post` in input order
    ///
    /// An input error surfaces after the file before it finishes, as when files run
    /// one at a time
    pub(super) fn run_prefetched_stream<E, A, S>(
        &mut self,
        files: impl IntoIterator<Item = Result<(S, A), E>>,
        post: &mut BatchPostProcessor<'_>,
    ) -> Result<(), BatchStreamError<E>>
    where
        A: AsRef<[f32]> + Send + Sync,
        S: AsRef<str>,
    {
        let plan = self.concurrent_plan();
        let Self {
            seg_model,
            emb_model,
            powerset,
            ..
        } = self;

        std::thread::scope(|scope| {
            let (jobs, queue) = bounded::<SegmentationJob<A>>(1);
            let worker = std::thread::Builder::new()
                .name(WORKER.to_owned())
                .spawn_scoped(scope, move || {
                    for job in queue {
                        let result =
                            seg_model.run_streaming(job.audio.as_ref().as_ref(), job.windows);
                        // the main thread drops a file's receivers when it stops early
                        let _ = job.done.send(result);
                    }
                })
                .map_err(|source| PipelineError::WorkerSpawn {
                    worker: WORKER,
                    source,
                })?;

            let run = {
                // dropped before the join below, which ends the worker's job loop
                let jobs = jobs;
                let enqueue = |(file_id, audio): (S, A)| {
                    let audio = Arc::new(audio);
                    let (windows_tx, windows) = bounded(CONCURRENT_WINDOW_QUEUE);
                    let (done_tx, done) = bounded(1);
                    jobs.send(SegmentationJob {
                        audio: Arc::clone(&audio),
                        windows: windows_tx,
                        done: done_tx,
                    })
                    .map_err(|_| worker_panic())?;
                    Ok::<_, PipelineError>(QueuedFile {
                        file_id,
                        audio,
                        windows,
                        done,
                    })
                };

                (|| {
                    let mut files = files.into_iter();
                    let Some(first) = files.next() else {
                        return Ok(());
                    };
                    let mut current = enqueue(first.map_err(BatchStreamError::Input)?)?;
                    loop {
                        post.collect_ready()?;
                        // queued now, the next file's segmentation starts as soon as
                        // this file's ends
                        let upcoming = match files.next() {
                            Some(Ok(file)) => Some(Ok(enqueue(file)?)),
                            Some(Err(error)) => Some(Err(error)),
                            None => None,
                        };

                        let QueuedFile {
                            file_id,
                            audio,
                            windows,
                            done,
                        } = current;
                        let file_id = file_id.as_ref().to_owned();
                        let start = Instant::now();
                        let span = tracing::debug_span!("batch_inference", %file_id);
                        let artifacts = span.in_scope(|| {
                            let audio = audio.as_ref().as_ref();
                            let embedded = plan.embed(powerset, emb_model, audio, windows);
                            // the worker reports once this file's window sender is gone
                            done.recv().map_err(|_| worker_panic())??;
                            plan.artifacts(embedded?, start.elapsed())
                        })?;
                        let inference_ms = start.elapsed().as_millis();
                        let audio_secs = audio.as_ref().as_ref().len() as f64 / 16_000.0;
                        drop(audio);
                        post.submit(artifacts, file_id, audio_secs, start, inference_ms)?;

                        match upcoming {
                            Some(Ok(next)) => current = next,
                            Some(Err(error)) => return Err(BatchStreamError::Input(error)),
                            None => return Ok(()),
                        }
                    }
                })()
            };

            let joined = worker.join().map_err(|_| worker_panic());
            run?;
            joined?;
            Ok(())
        })
    }
}
