//! Bounded native CPU execution with shared models and private reusable scratch

use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::inference::InferenceError;

/// Owns at least one workspace and grows only for useful independent jobs
pub(crate) struct CpuWorkers<W> {
    workspaces: Vec<W>,
    budget: NonZeroUsize,
}

impl<W: Send> CpuWorkers<W> {
    /// Starts with one workspace and a host budget capped to bound private memory
    pub(crate) fn new(first: W) -> Self {
        let host = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        Self {
            workspaces: vec![first],
            budget: host.min(NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN)),
        }
    }

    /// Borrows the initial workspace without growing or starting threads
    pub(crate) fn first(&mut self) -> &mut W {
        &mut self.workspaces[0]
    }

    /// Executes independent jobs in input order and joins every worker before returning
    pub(crate) fn map<T: Sync, O: Send>(
        &mut self,
        inputs: &[T],
        init: impl Fn() -> W,
        execute: impl Fn(&mut W, &T) -> Result<O, InferenceError> + Sync,
    ) -> Result<Vec<O>, InferenceError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let count = inputs.len().min(self.budget.get());
        while self.workspaces.len() < count {
            self.workspaces.push(init());
        }

        let run = |workspace: &mut W, chunk: &[T]| {
            chunk
                .iter()
                .map(|input| execute(workspace, input))
                .collect::<Result<Vec<_>, _>>()
        };
        let results = if count == 1 {
            vec![catch_unwind(AssertUnwindSafe(|| run(self.first(), inputs)))]
        } else {
            std::thread::scope(|scope| {
                let mut remaining = inputs;
                let mut handles = Vec::with_capacity(count);
                for (index, workspace) in self.workspaces[..count].iter_mut().enumerate() {
                    let length = inputs.len() / count + usize::from(index < inputs.len() % count);
                    let (chunk, rest) = remaining.split_at(length);
                    remaining = rest;
                    let run = &run;
                    handles.push(scope.spawn(move || run(workspace, chunk)));
                }

                // join every worker before selecting errors or replacing damaged scratch
                handles
                    .into_iter()
                    .map(|handle| handle.join())
                    .collect::<Vec<_>>()
            })
        };
        let mut outcomes = Vec::with_capacity(count);
        for (index, result) in results.into_iter().enumerate() {
            let outcome = match result {
                Ok(outcome) => outcome,
                Err(_) => {
                    // a panic can interrupt scratch invariants; never reuse that workspace
                    self.workspaces[index] = init();
                    Err(worker_panic())
                }
            };
            outcomes.push(outcome);
        }

        let chunks = outcomes.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(chunks.into_iter().flatten().collect())
    }
}

fn worker_panic() -> InferenceError {
    InferenceError::WorkerPanic {
        worker: "native CPU",
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod test_support {
    use super::{CpuWorkers, NonZeroUsize};

    pub(crate) fn with_budget<W>(first: W, budget: usize) -> CpuWorkers<W> {
        CpuWorkers {
            workspaces: vec![first],
            budget: NonZeroUsize::new(budget).expect("nonzero test budget"),
        }
    }

    pub(crate) fn count<W>(workers: &CpuWorkers<W>) -> usize {
        workers.workspaces.len()
    }
}
