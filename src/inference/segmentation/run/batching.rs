use std::num::NonZeroUsize;

/// Model batch sizes supported by a loaded segmentation backend
#[derive(Clone, Copy)]
pub(super) enum Batching {
    #[cfg(any(feature = "migraphx", test))]
    Single,
    #[cfg(any(feature = "cpu", test))]
    Useful(NonZeroUsize),
    #[cfg(any(feature = "migraphx", feature = "coreml", feature = "_cuda", test))]
    Fixed(NonZeroUsize),
}

/// Output delivery whose existing fixed-tail behavior must be preserved
#[derive(Clone, Copy)]
pub(super) enum Delivery {
    Collected,
    Streaming,
}

/// A nonempty inference call, including only the outputs useful to the caller
#[derive(Clone, Copy)]
pub(super) struct BatchPlan {
    useful: NonZeroUsize,
    model: NonZeroUsize,
}

impl Batching {
    pub(super) fn plan(self, remaining: NonZeroUsize, _delivery: Delivery) -> BatchPlan {
        match self {
            #[cfg(any(feature = "migraphx", test))]
            Self::Single => BatchPlan {
                useful: NonZeroUsize::MIN,
                model: NonZeroUsize::MIN,
            },
            #[cfg(any(feature = "cpu", test))]
            Self::Useful(capacity) => {
                let useful = remaining.min(capacity);
                BatchPlan {
                    useful,
                    model: useful,
                }
            }
            #[cfg(any(feature = "migraphx", feature = "coreml", feature = "_cuda", test))]
            Self::Fixed(capacity) => {
                let one = NonZeroUsize::MIN;
                if remaining >= capacity {
                    return BatchPlan {
                        useful: capacity,
                        model: capacity,
                    };
                }

                // collected accelerator tails use the single-window model; streaming
                // tails use fixed padding, which can produce different scores
                if matches!(_delivery, Delivery::Streaming) && remaining > one {
                    return BatchPlan {
                        useful: remaining,
                        model: capacity,
                    };
                }

                BatchPlan {
                    useful: one,
                    model: one,
                }
            }
        }
    }
}

impl BatchPlan {
    pub(super) fn useful(self) -> usize {
        self.useful.get()
    }

    pub(super) fn model(self) -> usize {
        self.model.get()
    }

    pub(super) fn is_single(self) -> bool {
        self.model == NonZeroUsize::MIN
    }
}

#[cfg(test)]
mod tests {
    use super::{Batching, Delivery};
    use std::num::NonZeroUsize;

    fn calls(batching: Batching, delivery: Delivery, total: usize) -> Vec<(usize, usize, usize)> {
        let mut calls = Vec::new();
        let mut next = 0;
        while let Some(remaining) = NonZeroUsize::new(total - next) {
            let plan = batching.plan(remaining, delivery);
            calls.push((next, plan.useful(), plan.model()));
            next += plan.useful();
        }
        calls
    }

    #[test]
    fn collected_fixed_tails_keep_single_window_inference() {
        let fixed = Batching::Fixed(NonZeroUsize::new(32).unwrap());
        assert_eq!(calls(fixed, Delivery::Collected, 0), []);
        assert_eq!(calls(fixed, Delivery::Collected, 32), [(0, 32, 32)]);
        assert_eq!(
            calls(fixed, Delivery::Collected, 35),
            [(0, 32, 32), (32, 1, 1), (33, 1, 1), (34, 1, 1)]
        );
    }

    #[test]
    fn streaming_fixed_tails_pad_only_multi_window_calls() {
        let fixed = Batching::Fixed(NonZeroUsize::new(32).unwrap());
        assert_eq!(calls(fixed, Delivery::Streaming, 1), [(0, 1, 1)]);
        assert_eq!(calls(fixed, Delivery::Streaming, 31), [(0, 31, 32)]);
        assert_eq!(
            calls(fixed, Delivery::Streaming, 33),
            [(0, 32, 32), (32, 1, 1)]
        );
        assert_eq!(
            calls(fixed, Delivery::Streaming, 35),
            [(0, 32, 32), (32, 3, 32)]
        );
    }

    #[test]
    fn useful_batches_keep_partial_calls_unpadded_for_both_deliveries() {
        let useful = Batching::Useful(NonZeroUsize::new(8).unwrap());
        for delivery in [Delivery::Collected, Delivery::Streaming] {
            assert_eq!(calls(useful, delivery, 0), []);
            assert_eq!(calls(useful, delivery, 1), [(0, 1, 1)]);
            assert_eq!(calls(useful, delivery, 7), [(0, 7, 7)]);
            assert_eq!(calls(useful, delivery, 8), [(0, 8, 8)]);
            assert_eq!(calls(useful, delivery, 9), [(0, 8, 8), (8, 1, 1)]);
            assert_eq!(
                calls(useful, delivery, 19),
                [(0, 8, 8), (8, 8, 8), (16, 3, 3)]
            );
        }
    }

    #[test]
    fn single_window_backend_never_pads_or_groups_inputs() {
        for delivery in [Delivery::Collected, Delivery::Streaming] {
            assert_eq!(
                calls(Batching::Single, delivery, 3),
                [(0, 1, 1), (1, 1, 1), (2, 1, 1)]
            );
        }
    }
}
