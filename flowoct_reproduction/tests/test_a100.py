"""Deployment checks: data selection and baseline preservation, without training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import a100


class A100DeploymentTests(unittest.TestCase):
    def setUp(self):
        workspace = a100.ROOT.parent.resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="a100_test_", dir=workspace)
        self.root = Path(self.temporary.name).resolve()
        # Verify the cleanup target before TemporaryDirectory removes it.
        self.assertTrue(self.root.is_relative_to(workspace))
        self.assertNotEqual(self.root, workspace)

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, parent, name="OCTA500_v1"):
        data = parent / name
        (data / "z256/3M").mkdir(parents=True)
        (data / "split_3M_z256.csv").write_text("case_id,path,split\n", encoding="utf-8")
        return data

    def test_extracting_inside_renamed_data_directory_needs_no_native(self):
        data = self.dataset(self.root)
        project = data / "flowoct_reproduction"
        project.mkdir()
        with patch.object(a100, "ROOT", project), patch.object(Path, "cwd", return_value=project), \
                patch.object(Path, "home", return_value=self.root):
            self.assertEqual(a100.find_data("3m"), data)
        self.assertFalse((data / "native").exists())

    def test_ambiguous_datasets_require_an_explicit_choice(self):
        first = self.dataset(self.root)
        self.dataset(self.root, "OCTA500_common_v1")
        project = self.root / "flowoct_reproduction"
        project.mkdir()
        with patch.object(a100, "ROOT", project), patch.object(Path, "cwd", return_value=project), \
                patch.object(Path, "home", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                a100.find_data("3m")
            self.assertEqual(a100.find_data("3m", first), first)

    def test_config_changes_only_manifest_and_device(self):
        data = self.dataset(self.root)
        original = json.loads((a100.ROOT / "configs/3m.json").read_text(encoding="utf-8"))
        adapted = a100.make_config("3m", data, "cuda:0")
        changed = {key for key in original if original[key] != adapted[key]}
        self.assertEqual(changed, {"manifest", "device"})
        self.assertEqual((adapted["network_chunk"], adapted["epochs"], adapted["train_slices"]), (0, 200, 0))

    def test_empty_upload_directory_is_rejected(self):
        empty = self.root / "OCTA500_v1"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No matching CSV"):
            a100.find_data("3m", empty)

    def test_resume_uses_saved_config_without_discovering_other_data(self):
        project = self.root / "flowoct_reproduction"
        run = project / "runs/flowoct_3m_200e"
        run.mkdir(parents=True)
        saved = run / "config.json"
        saved.write_text(json.dumps({"fov": "3M", "network_chunk": 0,
                                     "diagnostic_limits": None}), encoding="utf-8")
        before = saved.read_bytes()
        with patch.object(a100, "ROOT", project), patch("sys.argv", ["a100.py", "resume"]), \
                patch.object(a100, "find_data", side_effect=AssertionError("Must use saved data")), \
                patch.object(a100, "invoke") as invoke:
            a100.main()
        arguments = invoke.call_args.args
        self.assertIn(saved, arguments)
        self.assertIn(run / "latest.pt", arguments)
        self.assertEqual(saved.read_bytes(), before)

    def test_existing_run_is_not_overwritten(self):
        project = self.root / "flowoct_reproduction"
        (project / "runs/flowoct_3m_200e").mkdir(parents=True)
        with patch.object(a100, "ROOT", project), patch("sys.argv", ["a100.py", "train"]):
            with self.assertRaisesRegex(FileExistsError, "Run already exists"):
                a100.main()

    def test_diagnostic_or_chunked_run_cannot_be_resumed_as_formal(self):
        project = self.root / "flowoct_reproduction"
        run = project / "runs/flowoct_3m_200e"
        run.mkdir(parents=True)
        for chunk, limits in ((4, None), (0, {"train": 1, "validation": 1})):
            with self.subTest(chunk=chunk, limits=limits):
                (run / "config.json").write_text(json.dumps({"fov": "3M", "network_chunk": chunk,
                                                            "diagnostic_limits": limits}), encoding="utf-8")
                with patch.object(a100, "ROOT", project), patch("sys.argv", ["a100.py", "resume"]):
                    with self.assertRaisesRegex(ValueError, "formal unchunked"):
                        a100.main()


if __name__ == "__main__":
    unittest.main()
