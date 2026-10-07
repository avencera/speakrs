"""The filterbank producer has its own boundary domain and evidence geometry."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
qualify = importlib.import_module("qualify")


class FbankDomain(unittest.TestCase):
    def test_domain_expansion_covers_every_valid_batch_only(self):
        coverage = qualify.parse_coverage(
            {"entries": [{"layers": "all", "batches": "all", "maths": "all"}]},
            "fbankdft",
        )
        self.assertEqual(coverage.batches, frozenset(range(1, 33)))
        self.assertEqual(len(coverage.triples), 64)
        self.assertTrue(coverage.declared("fbank.dft", 31, "tf32"))
        self.assertFalse(coverage.declared("fbank.dft", 33, "tf32"))
        for batch in (0, 33, 64):
            with self.assertRaisesRegex(qualify.Rejected, "cannot test"):
                qualify.parse_coverage(
                    {
                        "entries": [
                            {"layers": "all", "batches": [batch], "maths": "all"}
                        ]
                    },
                    "fbankdft",
                )
        self.assertEqual(qualify.sanitizer_batches(coverage), list(range(1, 33)))

    def test_case_evidence_uses_the_producer_and_consumer_boundaries(self):
        numeric = qualify.expected_ids("fbankdft", "numeric", "fp32")
        self.assertEqual(len(numeric), 140)
        self.assertIn("fp32/mixed/b31/fbank.dft/switched", numeric)
        self.assertIn("fp32/last/b1/stage", numeric)
        self.assertNotIn("fp32/mixed/b33/fbank.dft", numeric)
        self.assertEqual(len(qualify.expected_ids("fbankdft", "timing", "tf32")), 70)
        self.assertEqual(qualify.target_cases("sincnet"), qualify.CASES)
        self.assertEqual(qualify.target_batches("lstm"), qualify.BATCHES)

    def test_draw_bytes_are_bound_to_energy_layout_and_boundary_domain(self):
        self.assertEqual(
            qualify.draw_length("tf32/mixed/b31/stage/switched", "fbank.dft"),
            31 * 998 * 80,
        )
        for case, layer in (
            ("tf32/mixed/b33/stage", "fbank.dft"),
            ("tf32/mixed/b31/stage", "lstm.stack"),
            ("tf32/mixed/b0/stage", "fbank.dft"),
        ):
            with self.assertRaises(qualify.Rejected):
                qualify.draw_length(case, layer)
        coverage = qualify.Coverage.product(("fbank.dft",), (2, 31), ("tf32",))
        self.assertEqual(
            qualify.band_layers(coverage, "Oxide"), "2:fbank.dft;31:fbank.dft"
        )


if __name__ == "__main__":
    unittest.main()
