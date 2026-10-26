"""CPU regressions; run: python -m unittest discover -s tests -v."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AT-ReID-fast"))


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load_source("original_moae", ROOT / "AT-ReID/model/moae.py")
fast = load_source("fast_moae", ROOT / "AT-ReID-fast/model/moae.py")
checkpoint_utils = load_source("checkpoint_utils", ROOT / "AT-ReID-fast/others/utils.py")


class OriginalRoutingTests(unittest.TestCase):
    def make_model(self, k=1):
        model = original.Mlp_moe(4, 8, 4, ncls=6, moae=True, k=k).eval()
        # Equal, constant expert outputs isolate learning through the gate.
        with torch.no_grad():
            for expert in model.ffn:
                expert[-1].weight.zero_()
                expert[-1].bias.fill_(1.0)
            for gate in model.gate:
                gate.weight[0].fill_(0.1)
                gate.weight[1].fill_(-0.1)
        return model

    def test_top1_gate_receives_task_gradient_and_learns(self):
        model = self.make_model()
        optimizer = torch.optim.SGD(model.gate.parameters(), lr=0.1)
        inputs = torch.ones(2, 6, 4)
        loss = model(inputs).square().mean()
        loss.backward()
        for gate in model.gate:
            self.assertTrue(torch.isfinite(gate.weight.grad).all())
            self.assertGreater(gate.weight.grad.norm().item(), 1e-3)
        optimizer.step()
        self.assertLess(model(inputs).square().mean().item(), loss.item())

    def test_tied_scores_keep_one_weighted_expert(self):
        model = self.make_model()
        with torch.no_grad():
            for gate in model.gate:
                gate.weight.zero_()
        # One selected expert with p=0.5, not zero experts or a unit weight.
        torch.testing.assert_close(model(torch.ones(2, 6, 4)), torch.full((2, 6, 4), 0.5))

    def test_dense_and_non_moae_outputs_are_preserved(self):
        inputs = torch.ones(2, 8, 4)
        torch.testing.assert_close(self.make_model(k=2)(inputs), inputs)
        baseline = original.Mlp_moe(4, 8, 4, ncls=6, moae=False).eval()
        torch.testing.assert_close(baseline(inputs), baseline.ffn[0](inputs))


class FastCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def make_model(self, moae=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return fast.ViT(
                img_size=(32, 16), patch_size=16, stride_size=16,
                embed_dim=16, depth=2, num_heads=4, mlp_ratio=2,
                ncls=6, moae=moae, attention_backend="eager",
            ).eval()

    def set_bias(self, model):
        for index, block in enumerate(model.blocks):
            block.mlp.balance_bias.copy_(
                torch.tensor([0.3, -0.3]).expand(6, -1) * (index + 1) / 2
            )

    def make_legacy_state(self, model):
        state = model.state_dict()
        for key in list(state):
            if key.endswith(".balance_bias"):
                del state[key]
        for key, metadata in state._metadata.items():
            if key.endswith(".mlp"):
                metadata["version"] = 1
        return state

    def capture_forward(self, model, images):
        routes = []

        def capture(module, inputs):
            probs, choices = module._compute_route_probabilities(inputs[0][:, :6])
            routes.append((probs.clone(), choices.clone()))

        handles = [block.mlp.register_forward_pre_hook(capture) for block in model.blocks]
        try:
            with torch.no_grad():
                outputs = model(images)
        finally:
            for handle in handles:
                handle.remove()
        return outputs, routes

    def test_checkpoint_helpers_preserve_bias_routes_and_outputs(self):
        model = self.make_model()
        self.set_bias(model)
        images = torch.randn(3, 3, 32, 16)
        expected, expected_routes = self.capture_forward(model, images)
        restored = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_utils.save_checkpoint(model, directory, epoch=120)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                checkpoint = checkpoint_utils.load_checkpoint(
                    restored, Path(directory) / "epoch_best.t"
                )
            self.assertFalse(any("zero routing bias" in str(w.message) for w in caught))
            self.assertEqual(checkpoint["epoch"], 120)
            for index in range(2):
                self.assertIn(f"blocks.{index}.mlp.balance_bias", checkpoint["model"])
        actual, actual_routes = self.capture_forward(restored, images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for old_route, new_route in zip(expected_routes, actual_routes):
            for before, after in zip(old_route, new_route):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
        for before, after in zip(model.blocks, restored.blocks):
            torch.testing.assert_close(after.mlp.balance_bias, before.mlp.balance_bias, rtol=0, atol=0)

    def test_legacy_checkpoint_warns_and_resets_existing_bias(self):
        model = self.make_model()
        self.set_bias(model)
        legacy_state = self.make_legacy_state(model)
        restored = self.make_model()
        self.set_bias(restored)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.t"
            torch.save({"model": legacy_state, "epoch": 120}, path)
            with self.assertWarnsRegex(UserWarning, "original training-time bias cannot be recovered"):
                checkpoint_utils.load_checkpoint(restored, path)
        for block in restored.blocks:
            self.assertEqual(torch.count_nonzero(block.mlp.balance_bias).item(), 0)
        for block in model.blocks:
            block.mlp.balance_bias.zero_()
        images = torch.randn(3, 3, 32, 16)
        with torch.no_grad():
            torch.testing.assert_close(restored(images), model(images), rtol=0, atol=0)

    def test_legacy_plain_state_dict_is_supported(self):
        model = self.make_model()
        legacy_state = dict(self.make_legacy_state(model))
        with self.assertWarnsRegex(UserWarning, "zero routing bias"):
            model.load_state_dict(legacy_state)

    def test_new_checkpoint_missing_bias_is_rejected(self):
        model = self.make_model()
        state = model.state_dict()
        del state["blocks.0.mlp.balance_bias"]
        with self.assertRaisesRegex(RuntimeError, "blocks.0.mlp.balance_bias"):
            model.load_state_dict(state)

    def test_missing_parameter_and_invalid_bias_shape_are_rejected(self):
        model = self.make_model()
        legacy_state = self.make_legacy_state(model)
        del legacy_state["blocks.0.mlp.gate_pair"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaisesRegex(RuntimeError, "blocks.0.mlp.gate_pair"):
                model.load_state_dict(legacy_state)
        state = model.state_dict()
        state["blocks.0.mlp.balance_bias"] = torch.zeros(1, 2)
        with self.assertRaisesRegex(RuntimeError, "size mismatch.*balance_bias"):
            model.load_state_dict(state)

    def test_non_moae_checkpoint_loads_without_legacy_warning(self):
        model = self.make_model(moae=False)
        restored = self.make_model(moae=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.load_state_dict(model.state_dict())
        self.assertEqual(len(caught), 0)
        self.assertFalse(any(key.endswith("balance_bias") for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
