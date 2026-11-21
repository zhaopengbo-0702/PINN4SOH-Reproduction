"""Compare two-epoch author PINN updates with explicit formulas for all lanes."""

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import run_extended as runner


REPO = Path(__file__).resolve().parents[1] / "PINN4SOH"


def manual_training(module, arguments, loaders, variant):
    import numpy as np
    import torch
    model = module.PINN(arguments)
    handlers = list(model.logger.handlers)
    minimum, best, labels, prediction, best_epoch = 10, None, None, None, None
    first_penalty = None
    for epoch in range(1, 3):
        model.train()
        for x1, x2, y1, y2 in loaders["train"]:
            u1, f1 = model.forward(x1)
            u2, f2 = model.forward(x2)
            data = .5 * model.loss_func(u1, y1) + .5 * model.loss_func(u2, y2)
            residual = .5 * model.loss_func(f1, torch.zeros_like(f1)) + .5 * model.loss_func(f2, torch.zeros_like(f2))
            if variant == "published_label_monotonicity":
                physical = torch.relu(y2 - y1).sum()
            elif variant == "paper_prediction_monotonicity":
                physical = torch.relu(u2 - u1).sum()
            else:
                physical = torch.relu((u2 - u1) * (y1 - y2)).sum()
            if first_penalty is None:
                first_penalty = float(physical.detach())
            loss = data + arguments.alpha * residual + arguments.beta * physical
            model.optimizer1.zero_grad()
            model.optimizer2.zero_grad()
            loss.backward()
            model.optimizer1.step()
            model.optimizer2.step()
        model.scheduler.step()
        valid = model.Valid(loaders["valid"])
        if valid < minimum:
            minimum = valid
            labels, prediction = model.Test(loaders["test"])
            best = copy.deepcopy({"solution_u": model.solution_u.state_dict(),
                                  "dynamical_F": model.dynamical_F.state_dict()})
            best_epoch = epoch
    model.clear_logger()
    for handler in handlers:
        handler.close()
    return best, np.asarray(labels), prediction, best_epoch, first_penalty


class LossVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO))
        import torch
        torch.set_num_threads(1)
        cls.module = importlib.import_module("main_XJTU")

    @classmethod
    def tearDownClass(cls):
        if sys.path and sys.path[0] == str(REPO):
            sys.path.pop(0)

    def test_three_two_epoch_updates_match_explicit_formulas(self):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(91)
        x1 = torch.randn(8, 17)
        x2 = x1 + .02
        y1 = torch.linspace(.95, .88, 8).reshape(-1, 1)
        y2 = y1 + .01
        dataset = TensorDataset(x1, x2, y1, y2)
        loaders = {role: DataLoader(dataset, batch_size=4, shuffle=False) for role in ["train", "valid", "test"]}
        with tempfile.TemporaryDirectory(prefix="pinn_loss_formulas_") as temporary:
            root = Path(temporary)
            for variant in runner.LOSS_VARIANTS:
                with self.subTest(variant=variant):
                    base_args = runner.get_author_args(self.module, "PINN", root / variant)
                    base_args.epochs, base_args.warmup_epochs = 2, 1
                    base_args.alpha, base_args.beta = .7, 20
                    manual_args = argparse.Namespace(**vars(base_args))
                    manual_args.save_folder = str(root / (variant + "_manual"))
                    torch.manual_seed(442)
                    expected, expected_y, expected_prediction, expected_epoch, expected_penalty = manual_training(
                        self.module, manual_args, loaders, variant)
                    torch.manual_seed(442)
                    actual_y, actual_prediction, actual_epoch = runner.train_task(
                        {"model": "PINN", "loss_variant": variant}, base_args, loaders, self.module, None, root / variant)
                    state = torch.load(root / variant / "model.pth", map_location="cpu", weights_only=True)
                    for component in expected:
                        for name, tensor in expected[component].items():
                            self.assertTrue(torch.equal(tensor, state[component][name]), (variant, component, name))
                    self.assertTrue(np.array_equal(actual_y, expected_y))
                    self.assertTrue(np.array_equal(actual_prediction, expected_prediction))
                    self.assertEqual(actual_epoch, expected_epoch)
                    diagnostic = json.loads((root / variant / "physical_loss_diagnostic.json").read_text())
                    self.assertAlmostEqual(diagnostic["first_batch_physical_penalty_sum"], expected_penalty, places=7)
                    self.assertEqual(diagnostic["first_batch_output_requires_grad"], variant != "published_label_monotonicity")
                    self.assertEqual(diagnostic["parameter_gradient_expected_identically_zero"], variant == "published_label_monotonicity")

    def test_formula_gradients_and_nonzero_label_penalty(self):
        import torch
        first = torch.tensor([[.8]], requires_grad=True)
        second = torch.tensor([[.9]], requires_grad=True)
        high, low = torch.tensor([[1.0]]), torch.tensor([[.9]])
        paper = torch.relu(second - first).sum()
        current = torch.relu((second - first) * (high - low)).sum()
        self.assertEqual([round(float(value), 5) for value in torch.autograd.grad(paper, (first, second), retain_graph=True)], [-1, 1])
        self.assertEqual([round(float(value), 5) for value in torch.autograd.grad(current, (first, second))], [-.1, .1])
        published = torch.relu(high - low).sum()
        self.assertGreater(float(published), 0)
        self.assertFalse(published.requires_grad)

    def test_default_variant_and_baseline_guards(self):
        task = dict(task_id="test", family="full", dataset="XJTU", batch="2C", model="PINN",
                    repeat=1, seed=1, protocol_id="paper_note3")
        original = copy.deepcopy(task)
        normalized = runner.normalize_task(task)
        self.assertEqual(task, original)
        self.assertEqual(normalized["loss_variant"], "author_current")
        for model in ["CNN", "MLP"]:
            with self.assertRaisesRegex(ValueError, "only apply to PINN"):
                runner.normalize_task(dict(task, model=model, loss_variant="published_label_monotonicity"))
        normalized = runner.normalize_task(dict(task, overrides={"early_stop": 10}, protocol_evidence=["2024 MIT entry"]))
        self.assertEqual(normalized["overrides"]["early_stop"], 10)


if __name__ == "__main__":
    unittest.main()
