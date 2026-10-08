"""Collection inventories must not depend on candidate declarations."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
qualify = importlib.import_module("qualify")


class Collections(unittest.TestCase):
    def test_wideconv_exact_inventory(self):
        expected = (
            ("resnet.conv1",)
            + tuple(
                f"resnet.layer{stage}.{block}.conv{conv}"
                for stage, count in ((3, 6), (4, 3))
                for block in range(count)
                for conv in (1, 2)
            )
            + tuple(f"resnet.layer{stage}.0.shortcut.0" for stage in (2, 3, 4))
        )
        self.assertEqual(qualify.layers("wideconv"), expected)
        self.assertEqual(len(expected), 22)
        rows = {
            f"fp32/{case}/b{batch}/{boundary}"
            for case, batch in qualify.MODEL.cases
            for boundary in (*expected, "stage")
        }
        self.assertEqual(qualify.expected_ids("wideconv", "profile", "fp32"), rows)
        self.assertEqual(
            qualify.expected_ids("wideconv", "numeric", "fp32"),
            rows | {f"{row}/switched" for row in rows},
        )

    def test_segdense_resolution_and_rows(self):
        paths = {
            "conv1": "sincnet.conv1",
            "conv2": "sincnet.conv2",
            "linear0": "linear0",
            "linear1": "linear1",
            "classifier": "linear2",
            "embedding": "resnet.seg_1",
        }
        for name, layer in paths.items():
            target = qualify.resolve_target("segdense", name)
            self.assertEqual(target, f"segdense-{name}")
            self.assertEqual(qualify.layers(target), (layer,))
            self.assertEqual(qualify.target_batches(target), (1, 7, 32, 33, 64))
            expected = {
                f"tf32/{case}/b{batch}/{boundary}"
                for case, batch in qualify.MODEL.cases
                for boundary in (layer, "stage")
            }
            self.assertEqual(qualify.expected_ids(target, "timing", "tf32"), expected)
            self.assertEqual(
                qualify.expected_ids(target, "numeric", "tf32"),
                expected | {f"{row}/switched" for row in expected},
            )
        for target, name in (
            ("segdense", None),
            ("segdense", "unknown"),
            ("wideconv", "conv1"),
        ):
            with self.assertRaises(qualify.Rejected):
                qualify.resolve_target(target, name)

    def test_locked_draw_geometry(self):
        geometry = {
            "sincnet.conv1": 60 * 5321,
            "sincnet.conv2": 60 * 1769,
            "linear0": 589 * 128,
            "linear1": 589 * 128,
            "linear2": 589 * 7,
            "resnet.seg_1": 3 * 256,
            "resnet.conv1": 32 * 80 * 998,
            "resnet.layer2.0.shortcut.0": 64 * 40 * 499,
            "resnet.layer3.0.shortcut.0": 128 * 20 * 250,
            "resnet.layer4.0.shortcut.0": 256 * 10 * 125,
        }
        for stage, count in ((3, 6), (4, 3)):
            for block in range(count):
                for conv in (1, 2):
                    size = 128 * 20 * 250 if stage == 3 else 256 * 10 * 125
                    geometry[f"resnet.layer{stage}.{block}.conv{conv}"] = size
        for layer, elements in geometry.items():
            for batch in (1, 7, 32, 33, 64):
                self.assertEqual(
                    qualify.draw_length(f"tf32/mixed/b{batch}/stage", layer),
                    batch * elements,
                )

    def test_fault_applicability_and_preflight(self):
        registry = qualify.COLLECTION_REGISTRY
        for target in registry["LAYERS"]:
            for fault in ("StageTail", "StageTailControl"):
                self.assertEqual(
                    registry["MUTANT_APPLICABILITY"][target][fault],
                    "requires_accepted_production_plan",
                )
                with self.assertRaisesRegex(
                    qualify.Rejected, "accepted production plan"
                ):
                    qualify.preflight(target, fault)
            for fault in qualify.MUTANTS:
                if fault != "StageTail":
                    self.assertEqual(
                        registry["MUTANT_APPLICABILITY"][target][fault], "applicable"
                    )
                    qualify.preflight(target, fault)
            self.assertEqual(
                registry["MUTANT_APPLICABILITY"][target]["WrongLayout"],
                "out_of_scope_no_padded_layout",
            )


if __name__ == "__main__":
    unittest.main()
