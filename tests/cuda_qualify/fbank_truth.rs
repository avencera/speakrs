//! Complete independent f64 fbank stage truth, with private CPU-only row reuse

use super::{cpu, fbank_basis, fbank_frame, fbank_window};
use crate::inference::cuda::fbank::FbankConstants;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;

/// Version of the independent complete stage definition
pub(in super::super) const DEFINITION: &str = "fbank-stage-f64-direct-v1";
const SAMPLES: usize = 160_000;
const FRAMES: usize = 998;
const MELS: usize = 80;

/// SHA-256 of the unrounded little-endian f64 values, without an evidence prefix
pub(in super::super) fn sha(values: &[f64]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

struct Reference {
    window: Vec<f64>,
    basis: Vec<Vec<(f64, f64)>>,
    mel: Vec<f64>,
    constants: Value,
    // exact input bytes and one fixed constant owner prevent approximate cache matches
    rows: HashMap<Vec<u8>, Vec<f64>>,
}

impl Reference {
    fn new() -> Self {
        let built_in = FbankConstants::new();
        let window = fbank_window();
        let basis = fbank_basis();
        let mel: Vec<_> = built_in.mel().iter().copied().map(f64::from).collect();
        let mut all = window.clone();
        all.extend(
            basis
                .iter()
                .flatten()
                .flat_map(|&(cosine, sine)| [cosine, sine]),
        );
        all.extend(&mel);
        all.extend([32768.0, 0.97, f64::from(f32::EPSILON)]);
        let constants = json!({
            "sha256": sha(&all), "mel_sha256": sha(&mel),
            "window": "exact-f64-Hamming", "mel": "built-in-f32-converted-to-f64",
            "scale": 32768.0, "preemphasis": 0.97, "energy_floor": f64::from(f32::EPSILON),
            "window_f32_max_abs": window.iter().zip(built_in.window())
                .map(|(&exact, &rounded)| (exact - f64::from(rounded)).abs()).fold(0.0, f64::max),
        });
        Self {
            window,
            basis,
            mel,
            constants,
            rows: HashMap::new(),
        }
    }

    fn energies(&self, input: &[f32], frame: usize) -> Vec<f64> {
        let samples = fbank_frame(input, frame * 160, &self.window);
        // compute each direct sum once; mel outputs retain ascending bin term order
        let powers: Vec<_> = self
            .basis
            .iter()
            .map(|bin| {
                let mut real = 0.0;
                let mut imaginary = 0.0;
                for (sample, &(cosine, sine)) in samples.iter().zip(bin) {
                    real += sample * cosine;
                    imaginary += sample * sine;
                }
                real * real + imaginary * imaginary
            })
            .collect();
        (0..MELS)
            .map(|mel| {
                let mut energy = 0.0;
                for (bin, &power) in powers.iter().enumerate() {
                    let weight = self.mel[bin * MELS + mel];
                    if weight != 0.0 {
                        energy += power * weight;
                    }
                }
                energy
            })
            .collect()
    }

    fn stage(&mut self, input: &[f32], mode: cpu::Mode) -> Vec<f64> {
        assert!(!input.is_empty() && input.len().is_multiple_of(SAMPLES));
        let keys: Vec<Vec<u8>> = input
            .as_chunks::<SAMPLES>()
            .0
            .iter()
            .map(|row| row.iter().flat_map(|value| value.to_le_bytes()).collect())
            .collect();
        let mut missing = HashMap::new();
        let mut sources = Vec::new();
        for (row, key) in keys.iter().enumerate() {
            if !self.rows.contains_key(key) && !missing.contains_key(key) {
                missing.insert(key.clone(), sources.len());
                sources.push(row);
            }
        }
        // both proof policies see the same immutable cache and missing-row snapshot
        let compute = |policy| {
            let energies = cpu::ordered_map(policy, sources.len() * FRAMES, |index| {
                let row = sources[index / FRAMES];
                self.energies(&input[row * SAMPLES..(row + 1) * SAMPLES], index % FRAMES)
            })
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            let prepared = log_cmn(&energies, sources.len(), FRAMES, MELS);
            keys.iter()
                .flat_map(|key| {
                    self.rows
                        .get(key)
                        .map_or_else(
                            || {
                                let row = missing[key];
                                &prepared[row * FRAMES * MELS..(row + 1) * FRAMES * MELS]
                            },
                            Vec::as_slice,
                        )
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>()
        };
        let values = cpu::evaluate_mode(mode, compute);
        for row in sources {
            self.rows.insert(
                keys[row].clone(),
                values[row * FRAMES * MELS..(row + 1) * FRAMES * MELS].to_vec(),
            );
        }
        values
    }
}

/// Log and per-row, per-mel temporal CMN with ordered means and the f32 epsilon floor
fn log_cmn(energies: &[f64], rows: usize, frames: usize, mels: usize) -> Vec<f64> {
    assert!(frames > 0 && mels > 0);
    assert_eq!(energies.len(), rows * frames * mels);
    assert!(
        energies
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
    let mut values: Vec<_> = energies
        .iter()
        .map(|energy| energy.max(f64::from(f32::EPSILON)).ln())
        .collect();
    for row in values.chunks_exact_mut(frames * mels) {
        for mel in 0..mels {
            let mut sum = 0.0;
            for frame in 0..frames {
                sum += row[frame * mels + mel];
            }
            let mean = sum / frames as f64;
            for frame in 0..frames {
                row[frame * mels + mel] -= mean;
            }
        }
    }
    values
}

thread_local! {
    static REFERENCE: RefCell<Reference> = RefCell::new(Reference::new());
}

/// Complete deterministic truth; arrays and exact-byte row keys stay in locked CPU state
pub(in super::super) fn stage(input: &[f32]) -> Vec<f64> {
    REFERENCE.with(|reference| {
        reference
            .borrow_mut()
            .stage(input, cpu::Mode::from_environment())
    })
}

/// Identity of the independent definition and actual built-in constants
pub(in super::super) fn constants() -> Value {
    REFERENCE.with(|reference| reference.borrow().constants.clone())
}

#[cfg(test)]
mod tests {
    use super::{Reference, cpu, log_cmn};

    #[test]
    fn log_and_cmn_use_the_floor_and_each_rows_temporal_mean() {
        let floor = f64::from(f32::EPSILON);
        let energies = [
            1.0,
            0.0,
            4.0,
            floor * 4.0,
            16.0,
            floor * 16.0,
            2.0,
            8.0,
            2.0,
            8.0,
            2.0,
            8.0,
        ];
        let values = log_cmn(&energies, 2, 3, 2);
        for mel in 0..2 {
            for (frame, expected) in [-4.0f64.ln(), 0.0, 4.0f64.ln()].into_iter().enumerate() {
                assert!((values[frame * 2 + mel] - expected).abs() < 2e-15);
            }
        }
        assert_eq!(&values[6..], &[0.0; 6]);
        for row in values.as_chunks::<6>().0 {
            for mel in 0..2 {
                assert!(
                    ((0..3).map(|frame| row[frame * 2 + mel]).sum::<f64>() / 3.0).abs() < 2e-15
                );
            }
        }
    }

    #[test]
    fn full_direct_stage_matches_producer_constants_mapping_and_ordered_proof() {
        let input: Vec<_> = (0..160_000)
            .map(|n| (n as f32 * 0.037).sin() * 0.1)
            .collect();
        let mut reference = Reference::new();
        let energies = reference.energies(&input, 37);
        let constants = crate::inference::cuda::fbank::FbankConstants::new();
        let sample = super::super::fbank(&input, constants.mel(), vec![37 * 80, 37 * 80 + 79]);
        assert_eq!(sample.values, [energies[0], energies[79]]);
        assert!(energies.iter().all(|value| *value > 0.0));
        let _ = cpu::take_proofs();
        let full = reference.stage(&input, cpu::Mode::Verify);
        assert_eq!(full.len(), 998 * 80);
        assert_eq!(cpu::take_proofs().len(), 1);
        for mel in 0..80 {
            let mean = (0..998).map(|frame| full[frame * 80 + mel]).sum::<f64>() / 998.0;
            assert!(mean.abs() < 1e-12);
        }
        let repeated: Vec<_> = input.iter().copied().chain(input.iter().copied()).collect();
        let reused = reference.stage(&repeated, cpu::Mode::Verify);
        assert_eq!(&reused[..998 * 80], full);
        assert_eq!(&reused[998 * 80..], full);
        assert_eq!(reference.rows.len(), 1);
        assert_eq!(cpu::take_proofs().len(), 1);
        let mut changed = input.clone();
        changed[0] += 0.01;
        let mixed: Vec<_> = input.iter().copied().chain(changed).collect();
        let mixed_truth = reference.stage(&mixed, cpu::Mode::Verify);
        assert_eq!(&mixed_truth[..998 * 80], full);
        assert_ne!(&mixed_truth[998 * 80..], full);
        assert_eq!(reference.rows.len(), 2);
        assert_eq!(cpu::take_proofs().len(), 1);
        assert!(reference.constants["window_f32_max_abs"].as_f64().unwrap() > 0.0);
    }
}
