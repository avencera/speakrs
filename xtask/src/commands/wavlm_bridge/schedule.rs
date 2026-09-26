//! Bounded worker pool that returns results in job order
//!
//! Each worker owns its resources, such as one embedding session, and runs one
//! job at a time. A dispatcher thread hands jobs over a rendezvous channel, so
//! a job starts only when a worker is free and at most one job per worker is
//! in flight. Results come back in any order and the caller receives them in
//! job order. Failures follow the serial contract: the error of the lowest
//! failing job index is returned, and no job with a higher index starts after
//! a failure is known

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::ScopedJoinHandle;

use color_eyre::eyre::{Report, Result, ensure, eyre};

/// One owner of per-worker resources that runs jobs in sequence
pub(crate) trait Worker: Send {
    /// Input for one job
    type Job: Send;
    /// Result of one successful job
    type Output: Send;

    /// Run one job to completion
    fn run(&mut self, job: Self::Job) -> Result<Self::Output>;
}

/// Order in which job indexes are handed to workers
///
/// The order is a permutation of the job indexes. It controls only when a job
/// starts; results are always returned in job index order
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DispatchOrder(Vec<usize>);

impl DispatchOrder {
    /// Dispatch jobs in index order
    pub(crate) fn sequential(len: usize) -> Self {
        Self((0..len).collect())
    }

    /// Dispatch jobs in descending cost order, with ties in index order
    ///
    /// Starting the longest jobs first keeps one long job from running alone
    /// at the end of the pool
    pub(crate) fn longest_first(costs: &[u64]) -> Self {
        let mut order = (0..costs.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| (std::cmp::Reverse(costs[index]), index));
        Self(order)
    }

    /// Dispatch order for a pool of the given size
    ///
    /// One worker keeps index order, the same order as a plain loop
    pub(crate) fn for_pool(costs: &[u64], workers: NonZeroUsize) -> Self {
        if workers.get() == 1 {
            return Self::sequential(costs.len());
        }
        Self::longest_first(costs)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Lowest job index that has failed, shared by the dispatcher and workers
struct LowestFailure(AtomicUsize);

impl LowestFailure {
    fn new() -> Self {
        Self(AtomicUsize::new(usize::MAX))
    }

    fn record(&self, index: usize) {
        self.0.fetch_min(index, Ordering::AcqRel);
    }

    // a job above the lowest failure cannot change the reported error, and
    // serial execution would never have started it
    fn skips(&self, index: usize) -> bool {
        index > self.0.load(Ordering::Acquire)
    }
}

/// Run every job on the workers and return the outputs in job index order
///
/// Worker panics are resumed on the calling thread after all threads join
pub(crate) fn run_ordered<W: Worker>(
    workers: Vec<W>,
    jobs: Vec<W::Job>,
    order: DispatchOrder,
) -> Result<Vec<W::Output>> {
    ensure!(!workers.is_empty(), "worker pool has no workers");
    ensure!(
        order.len() == jobs.len(),
        "dispatch order has {} indexes for {} jobs",
        order.len(),
        jobs.len()
    );
    let job_count = jobs.len();
    let worker_count = workers.len();
    let lowest_failure = LowestFailure::new();

    std::thread::scope(|scope| {
        let (job_tx, job_rx) = crossbeam_channel::bounded::<(usize, W::Job)>(0);
        let (result_tx, result_rx) =
            crossbeam_channel::bounded::<(usize, Result<W::Output>)>(worker_count);

        let worker_handles = workers
            .into_iter()
            .map(|worker| {
                let job_rx = job_rx.clone();
                let result_tx = result_tx.clone();
                let lowest_failure = &lowest_failure;
                scope.spawn(move || run_worker(worker, &job_rx, &result_tx, lowest_failure))
            })
            .collect::<Vec<_>>();
        drop(job_rx);
        drop(result_tx);

        let lowest_failure = &lowest_failure;
        let dispatcher = scope.spawn(move || dispatch(jobs, order, &job_tx, lowest_failure));

        let mut outputs = (0..job_count).map(|_| None).collect::<Vec<_>>();
        let mut failures = BTreeMap::new();
        for (index, result) in result_rx {
            match result {
                Ok(output) => outputs[index] = Some(output),
                Err(error) => {
                    failures.insert(index, error);
                }
            }
        }

        join_all(dispatcher, worker_handles);
        collect_outputs(outputs, failures)
    })
}

fn run_worker<W: Worker>(
    mut worker: W,
    job_rx: &crossbeam_channel::Receiver<(usize, W::Job)>,
    result_tx: &crossbeam_channel::Sender<(usize, Result<W::Output>)>,
    lowest_failure: &LowestFailure,
) {
    for (index, job) in job_rx {
        // the dispatcher may hand over a job just before a failure is recorded
        if lowest_failure.skips(index) {
            continue;
        }
        let result = worker.run(job);
        if result.is_err() {
            lowest_failure.record(index);
        }
        if result_tx.send((index, result)).is_err() {
            return;
        }
    }
}

fn dispatch<J>(
    jobs: Vec<J>,
    order: DispatchOrder,
    job_tx: &crossbeam_channel::Sender<(usize, J)>,
    lowest_failure: &LowestFailure,
) {
    let mut jobs = jobs.into_iter().map(Some).collect::<Vec<_>>();
    for index in order.0 {
        if lowest_failure.skips(index) {
            continue;
        }
        let Some(job) = jobs[index].take() else {
            continue;
        };
        // a send fails only when every worker has exited after a panic
        if job_tx.send((index, job)).is_err() {
            return;
        }
    }
}

fn join_all(dispatcher: ScopedJoinHandle<'_, ()>, workers: Vec<ScopedJoinHandle<'_, ()>>) {
    let mut panic = dispatcher.join().err();
    for worker in workers {
        if let Err(payload) = worker.join() {
            panic.get_or_insert(payload);
        }
    }
    if let Some(payload) = panic {
        std::panic::resume_unwind(payload);
    }
}

fn collect_outputs<O>(
    outputs: Vec<Option<O>>,
    mut failures: BTreeMap<usize, Report>,
) -> Result<Vec<O>> {
    if let Some((_, error)) = failures.pop_first() {
        return Err(error);
    }
    outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| output.ok_or_else(|| eyre!("job {index} produced no result")))
        .collect()
}

/// Number of workers to start for a given job count
pub(crate) fn worker_count(requested: NonZeroUsize, jobs: usize) -> NonZeroUsize {
    NonZeroUsize::new(requested.get().min(jobs)).unwrap_or(NonZeroUsize::MIN)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier, Mutex};
    use std::time::Duration;

    use super::*;

    #[derive(Clone, Default)]
    struct Journal {
        started: Arc<Mutex<Vec<usize>>>,
        running: Arc<AtomicUsize>,
        max_running: Arc<AtomicUsize>,
    }

    impl Journal {
        fn started(&self) -> Vec<usize> {
            self.started.lock().unwrap().clone()
        }
    }

    #[derive(Clone)]
    enum Step {
        Succeed { delay_ms: u64 },
        Fail,
        Panic,
        WaitThenSucceed(Arc<Barrier>),
        FailThenRelease(Arc<Barrier>),
    }

    struct TestJob {
        index: usize,
        step: Step,
    }

    struct TestWorker {
        id: usize,
        journal: Journal,
    }

    impl Worker for TestWorker {
        type Job = TestJob;
        type Output = (usize, usize);

        fn run(&mut self, job: TestJob) -> Result<(usize, usize)> {
            self.journal.started.lock().unwrap().push(job.index);
            let running = self.journal.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.journal
                .max_running
                .fetch_max(running, Ordering::SeqCst);
            let result = match job.step {
                Step::Succeed { delay_ms } => {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                    Ok((job.index, self.id))
                }
                Step::Fail => Err(eyre!("job {} failed", job.index)),
                Step::Panic => panic!("job {} panicked", job.index),
                Step::WaitThenSucceed(barrier) => {
                    barrier.wait();
                    // the failing job records its failure before this job returns
                    std::thread::sleep(Duration::from_millis(50));
                    Ok((job.index, self.id))
                }
                Step::FailThenRelease(barrier) => {
                    barrier.wait();
                    Err(eyre!("job {} failed", job.index))
                }
            };
            self.journal.running.fetch_sub(1, Ordering::SeqCst);
            result
        }
    }

    fn workers(count: usize, journal: &Journal) -> Vec<TestWorker> {
        (0..count)
            .map(|id| TestWorker {
                id,
                journal: journal.clone(),
            })
            .collect()
    }

    fn jobs(steps: Vec<Step>) -> Vec<TestJob> {
        steps
            .into_iter()
            .enumerate()
            .map(|(index, step)| TestJob { index, step })
            .collect()
    }

    #[test]
    fn returns_outputs_in_job_order_whatever_the_completion_order() {
        let journal = Journal::default();
        // early jobs are slow, so later jobs finish first
        let steps = (0..12)
            .map(|index| Step::Succeed {
                delay_ms: 5 * (12 - index as u64),
            })
            .collect();

        let outputs = run_ordered(
            workers(4, &journal),
            jobs(steps),
            DispatchOrder::sequential(12),
        )
        .unwrap();

        let indexes = outputs.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        assert_eq!(indexes, (0..12).collect::<Vec<_>>());
        let worker_ids = outputs
            .iter()
            .map(|(_, worker)| *worker)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(worker_ids.len() > 1, "jobs must spread over workers");
    }

    #[test]
    fn dispatch_order_changes_start_order_but_not_output_order() {
        let journal = Journal::default();
        let steps = (0..5).map(|_| Step::Succeed { delay_ms: 0 }).collect();
        let order = DispatchOrder(vec![4, 2, 0, 3, 1]);

        let outputs = run_ordered(workers(1, &journal), jobs(steps), order).unwrap();

        assert_eq!(journal.started(), [4, 2, 0, 3, 1]);
        let indexes = outputs.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        assert_eq!(indexes, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn runs_at_most_one_job_per_worker() {
        let journal = Journal::default();
        let steps = (0..24).map(|_| Step::Succeed { delay_ms: 5 }).collect();

        run_ordered(
            workers(3, &journal),
            jobs(steps),
            DispatchOrder::sequential(24),
        )
        .unwrap();

        assert_eq!(journal.started().len(), 24);
        assert!(journal.max_running.load(Ordering::SeqCst) <= 3);
    }

    #[test]
    fn stops_starting_later_jobs_after_a_failure() {
        let journal = Journal::default();
        let steps = vec![
            Step::Succeed { delay_ms: 0 },
            Step::Fail,
            Step::Succeed { delay_ms: 0 },
            Step::Succeed { delay_ms: 0 },
        ];

        let error = run_ordered(
            workers(1, &journal),
            jobs(steps),
            DispatchOrder::sequential(4),
        )
        .unwrap_err();

        assert_eq!(error.to_string(), "job 1 failed");
        assert_eq!(journal.started(), [0, 1]);
    }

    #[test]
    fn reports_the_lowest_failing_index_like_serial_execution() {
        let journal = Journal::default();
        // job 3 fails first; job 1 still runs because serial execution would
        // have reached it before job 3, and its failure wins
        let steps = vec![
            Step::Succeed { delay_ms: 0 },
            Step::Fail,
            Step::Succeed { delay_ms: 0 },
            Step::Fail,
            Step::Succeed { delay_ms: 0 },
        ];
        let order = DispatchOrder(vec![3, 4, 2, 1, 0]);

        let error = run_ordered(workers(1, &journal), jobs(steps), order).unwrap_err();

        assert_eq!(error.to_string(), "job 1 failed");
        assert_eq!(journal.started(), [3, 2, 1, 0]);
    }

    #[test]
    fn in_flight_jobs_finish_before_the_error_returns() {
        let journal = Journal::default();
        let barrier = Arc::new(Barrier::new(2));
        let steps = vec![
            Step::WaitThenSucceed(barrier.clone()),
            Step::FailThenRelease(barrier),
            Step::Succeed { delay_ms: 0 },
            Step::Succeed { delay_ms: 0 },
        ];

        let error = run_ordered(
            workers(2, &journal),
            jobs(steps),
            DispatchOrder::sequential(4),
        )
        .unwrap_err();

        assert_eq!(error.to_string(), "job 1 failed");
        let mut started = journal.started();
        started.sort_unstable();
        assert_eq!(started, [0, 1]);
        assert_eq!(journal.running.load(Ordering::SeqCst), 0);
    }

    #[test]
    #[should_panic(expected = "job 2 panicked")]
    fn resumes_a_worker_panic_on_the_caller() {
        let journal = Journal::default();
        let steps = vec![
            Step::Succeed { delay_ms: 0 },
            Step::Succeed { delay_ms: 0 },
            Step::Panic,
            Step::Succeed { delay_ms: 0 },
        ];

        let _ = run_ordered(
            workers(2, &journal),
            jobs(steps),
            DispatchOrder::sequential(4),
        );
    }

    #[test]
    fn rejects_mismatched_order_and_empty_pool() {
        let journal = Journal::default();
        let steps = vec![Step::Succeed { delay_ms: 0 }];
        assert!(
            run_ordered(
                workers(1, &journal),
                jobs(steps),
                DispatchOrder::sequential(2)
            )
            .is_err()
        );
        assert!(
            run_ordered(
                Vec::<TestWorker>::new(),
                Vec::new(),
                DispatchOrder::sequential(0)
            )
            .is_err()
        );
    }

    #[test]
    fn longest_first_orders_by_cost_then_index() {
        assert_eq!(
            DispatchOrder::longest_first(&[5, 9, 5, 1, 9]),
            DispatchOrder(vec![1, 4, 0, 2, 3])
        );
    }

    #[test]
    fn worker_count_never_exceeds_the_job_count() {
        let four = NonZeroUsize::new(4).unwrap();
        assert_eq!(worker_count(four, 2).get(), 2);
        assert_eq!(worker_count(four, 9).get(), 4);
        assert_eq!(worker_count(four, 0).get(), 1);
    }

    #[test]
    fn empty_job_list_returns_no_outputs() {
        let journal = Journal::default();
        let outputs = run_ordered(
            workers(2, &journal),
            Vec::new(),
            DispatchOrder::sequential(0),
        );
        assert!(outputs.unwrap().is_empty());
    }
}
