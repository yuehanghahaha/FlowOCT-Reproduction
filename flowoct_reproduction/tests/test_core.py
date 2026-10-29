import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from flowoct.data import PairedVolumeDataset, read_manifest
from flowoct.metrics import volume_metrics
from flowoct.runner import (ROOT, load_config, make_model, read_checkpoint,
                            restore_rng, save_checkpoint, setup)
from flowoct.wavelets import DWT_2D, IDWT_2D


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        z, y, x = np.indices((32, 32, 32), dtype=np.float32)
        self.array = (4 * z + 2 * y + x) / 217 * 255
        self.rows = []
        for index, split in enumerate(("train", "validation", "test")):
            cid = str(index + 1)
            path = self.root / (cid + ".npz")
            np.savez(path, oct=self.array, octa=255 - self.array, axis_order="ZYX")
            self.rows.append({"case_id": cid, "patient_id": cid, "path": path.name,
                              "resolved_path": str(path), "split": split, "fov": "3M",
                              "leakage_group": cid})
        self.cfg = load_config(ROOT / "configs/3m.json")
        self.cfg.update(load_size=32, crop_size=32)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        path = self.root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def test_manifest_rejects_patient_and_content_leakage(self):
        read_manifest(self.manifest(), "3M", check_counts=False)
        self.rows[2]["patient_id"] = "1"
        with self.assertRaisesRegex(ValueError, "Patient crosses"):
            read_manifest(self.manifest(), "3M", check_counts=False)
        self.rows[2]["patient_id"] = "3"
        self.rows[2]["leakage_group"] = "1"
        with self.assertRaisesRegex(ValueError, "image group"):
            read_manifest(self.manifest(), "3M", check_counts=False)

    def test_orientation_and_paired_minmax(self):
        sample = PairedVolumeDataset(self.rows, "validation", self.cfg)[0]
        expected = self.array.transpose(1, 2, 0) / 255
        np.testing.assert_allclose(sample["A"][0], expected, atol=2e-7)
        np.testing.assert_allclose(sample["A"] + sample["B"], 1, atol=3e-7)
        self.assertEqual(sample["geometry"]["model_axis_order"], "YXZ")
        self.assertFalse(sample["geometry"]["flipped_z"])

    def test_constant_volume_fails_before_normalization(self):
        np.savez(self.rows[1]["resolved_path"], oct=np.ones((32,) * 3), octa=np.ones((32,) * 3))
        with self.assertRaisesRegex(ValueError, "Constant crop"):
            PairedVolumeDataset(self.rows, "validation", self.cfg)[0]


class NumericalTests(unittest.TestCase):
    def test_haar_reconstruction_gradients_and_subband_signs(self):
        # These signs/order match the author's matrix implementation.
        x = torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=torch.float64, requires_grad=True)
        bands = DWT_2D("haar")(x)
        np.testing.assert_allclose([band.item() for band in bands], [5., -1., -2., 0.], atol=1e-14)
        restored = IDWT_2D("haar")(*bands)
        torch.testing.assert_close(restored, x)
        restored.square().sum().backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_metrics_are_finite_and_projection_uses_z(self):
        rng = np.random.default_rng(8)
        target = rng.random((8, 9, 10), dtype=np.float32)
        scores = volume_metrics(target, target)
        self.assertEqual(scores["volume_mae"], 0)
        self.assertAlmostEqual(scores["volume_ssim"], 1)
        self.assertAlmostEqual(scores["axial_mip_ssim"], 1)
        self.assertTrue(all(np.isfinite(list(scores.values()))))

    def config(self):
        cfg = load_config(ROOT / "configs/3m.json")
        cfg.update(device="cpu", threads=2, crop_size=32, load_size=32, network_chunk=0)
        setup(cfg)
        return cfg

    @staticmethod
    def update(model, batch):
        model.train()
        model.set_input(batch)
        model.forward()
        return {"total": model.loss.item(), **model.get_current_loss()}

    def test_chunk_recomputation_keeps_objective_and_gradients(self):
        cfg = self.config()
        batch = {"A": torch.rand(1, 1, 8, 32, 32), "B": torch.rand(1, 1, 8, 32, 32)}
        regular = make_model(cfg, True)
        # A warmup avoids testing only the initially zero output layer.
        self.update(regular, batch)
        chunked = make_model({**cfg, "network_chunk": 1}, True)
        chunked.load_state_dict(regular.state_dict())
        # Deep-copy optimizer state so the two updates cannot share moments.
        import copy
        chunked.optimizer_G.load_state_dict(copy.deepcopy(regular.optimizer_G.state_dict()))
        torch.manual_seed(59)
        loss_a = self.update(regular, batch)
        torch.manual_seed(59)
        loss_b = self.update(chunked, batch)
        for key in loss_a:
            self.assertAlmostEqual(loss_a[key], loss_b[key], delta=3e-5)
        # Batch-size-dependent FP32 reductions can leave ~1e-9 residual bias
        # gradients around GroupNorm. Adam can amplify those via its epsilon;
        # the invariant is the full objective/gradient, not bitwise weights.
        for key, parameter in regular.named_parameters():
            other = dict(chunked.named_parameters())[key]
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=5e-4)
            self.assertTrue(torch.isfinite(other).all())

    def test_complete_checkpoint_replays_next_update(self):
        cfg = self.config()
        model = make_model(cfg, True)
        batch = {"A": torch.rand(1, 1, 8, 32, 32), "B": torch.rand(1, 1, 8, 32, 32)}
        self.update(model, batch)
        generators = {"train": torch.Generator().manual_seed(5),
                      "validation": torch.Generator().manual_seed(6)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, model, cfg, 1, 1, 0.0, "test-manifest", generators)
            expected_random = (random.random(), np.random.rand(), torch.randperm(8, generator=generators["train"]))
            expected_loss = self.update(model, batch)
            payload = read_checkpoint(path)
            restored = make_model(cfg, True)
            restored.load_state_dict(payload["model"])
            restored.optimizer_G.load_state_dict(payload["optimizer"])
            restore_rng(payload["rng"])
            generators["train"].set_state(payload["loader_rng"]["train"])
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            self.assertTrue(torch.equal(torch.randperm(8, generator=generators["train"]), expected_random[2]))
            actual_loss = self.update(restored, batch)
            self.assertEqual(expected_loss, actual_loss)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_projection_roi_with_nonzero_y_is_finite(self):
        cfg = self.config()
        model = make_model(cfg, True)
        target = torch.zeros(1, 1, 8, 32, 32)
        # Content at row 16 has a strictly positive padded ROI origin. The
        # upstream [y,x] indexing slices the C=1 axis empty on this example.
        target[:, :, :, 16:20, 10:20] = 0.75
        losses = self.update(model, {"A": torch.zeros_like(target), "B": target})
        self.assertTrue(all(np.isfinite(list(losses.values()))))
        self.assertGreater(losses["proj_loss"], 0)


if __name__ == "__main__":
    unittest.main()
