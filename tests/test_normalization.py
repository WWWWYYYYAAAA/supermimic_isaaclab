"""Native BatchNorm statistics and consistent normalization throughout RL."""

import unittest

import torch

from supermimic.normalization import FeatureBatchNorm, HistoryBatchNorm


class PolicyNormalizationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_statistics_match_native_batchnorm_and_forward_does_not_update(self):
        norm = HistoryBatchNorm(5, 10)
        reference = torch.nn.BatchNorm1d(5, eps=1e-4, momentum=0.1, affine=False)
        for count in (8, 3):
            data = torch.randn(count, 5) * 3 + 2
            norm.update(data)
            reference(data)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(norm.state_dict()[key], value)
        saved = {key: value.clone() for key, value in norm.state_dict().items()}
        history = torch.randn(3, 10, 5)
        reference.eval()
        expected = reference(history.transpose(1, 2)).transpose(1, 2).flatten(-2)
        for training in (True, False):
            norm.train(training)
            torch.testing.assert_close(norm(history.flatten(-2)), expected)
            torch.testing.assert_close(norm(history[:1].flatten(-2)), expected[:1])
            for key, value in saved.items():
                torch.testing.assert_close(norm.state_dict()[key], value)

    def test_input_gradients_and_single_sample_collection(self):
        norm = HistoryBatchNorm(5, 10)
        norm.update(torch.randn(8, 5))
        norm.eval()
        norm.update(torch.randn(1, 5))
        self.assertFalse(norm.training)
        self.assertEqual(norm.num_batches_tracked.item(), 1)
        x = torch.randn(1, 10, 5, requires_grad=True)
        norm(x.flatten(-2)).sum().backward()
        expected = torch.rsqrt(norm.running_var + norm.eps).expand_as(x)
        torch.testing.assert_close(x.grad, expected)

    def test_history_output_and_gradients_match_previous_layout_after_loading(self):
        for history in (1, 2, 10):
            with self.subTest(history=history):
                original = FeatureBatchNorm(5)
                original.update(torch.randn(9, 5) * 3 + 2)
                norm = HistoryBatchNorm(5, history)
                norm.load_state_dict(original.state_dict(), strict=True)
                self.assertEqual(set(norm.state_dict()), set(original.state_dict()))
                for batch in (1, 3):
                    x = torch.randn(batch, 5, history).transpose(1, 2).requires_grad_()
                    old = original(x.flatten(0, -2)).reshape_as(x).flatten(-2)
                    new = norm(x.flatten(-2))
                    weight = torch.randn_like(new)
                    old_grad = torch.autograd.grad((old * weight).sum(), x, retain_graph=True)[0]
                    new_grad = torch.autograd.grad((new * weight).sum(), x)[0]
                    torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(new_grad, old_grad, rtol=1e-6, atol=1e-6)
                graph = str(torch.jit.script(original).inlined_graph)
                self.assertNotIn("aten::flatten", graph)
                self.assertNotIn("aten::reshape", graph)


if __name__ == "__main__":
    unittest.main()
