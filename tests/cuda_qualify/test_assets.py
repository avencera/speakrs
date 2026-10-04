"""External evidence stays hash-bound and fails closed without cached bytes."""

import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/cuda/qualify"))
assets = importlib.import_module("assets")
lock = importlib.import_module("lock")


class Assets(unittest.TestCase):
    def test_missing_changed_and_valid_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tree"
            cache = Path(directory) / "cache"
            harness = root / "scripts/cuda/qualify"
            harness.mkdir(parents=True)
            name = "tests/cuda_qualify/baselines/lstm-source-test.json"
            contents = b"owner evidence"
            digest = hashlib.sha256(contents).hexdigest()
            definition = {
                "schema": 1,
                "directories": [],
                "files": [],
                "external_assets": {name: digest},
            }
            (harness / "SCOPE.json").write_text(json.dumps(definition))
            (harness / "LOCK").write_bytes(lock.canonical({name: digest}))
            with patch.dict(os.environ, SPEAKRS_QUALIFY_CACHE=str(cache)):
                self.assertEqual(
                    lock.verify(root),
                    hashlib.sha256((harness / "LOCK").read_bytes()).hexdigest(),
                )
                with self.assertRaisesRegex(
                    lock.LockError, "missing qualification asset.*import with"
                ):
                    assets.resolve(name, root)
                path = assets.asset_path(digest, root)
                path.parent.mkdir(parents=True)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(lock.LockError, "hash mismatch"):
                    assets.resolve(name, root)
                path.write_bytes(contents)
                self.assertEqual(assets.resolve(name, root), path)
                tree_asset = root / name
                tree_asset.parent.mkdir(parents=True)
                tree_asset.write_bytes(contents)
                with self.assertRaisesRegex(lock.LockError, "outside the tree"):
                    lock.verify(root)

    def test_inside_tree_cache_and_invalid_digest_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, SPEAKRS_QUALIFY_CACHE=str(root / "cache")):
                with self.assertRaisesRegex(lock.LockError, "outside the repository"):
                    assets.cache_directory(root)
            with self.assertRaisesRegex(
                lock.LockError, "invalid external asset digest"
            ):
                assets.asset_path("../../other-task", root)
