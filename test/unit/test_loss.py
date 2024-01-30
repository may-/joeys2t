import unittest

import torch

from joeynmt.loss import XentLoss


class TestXentLoss(unittest.TestCase):

    def setUp(self):
        seed = 42
        torch.manual_seed(seed)

    def test_label_smoothing(self):
        pad_index = 0
        smoothing = 0.4
        criterion = XentLoss(pad_index=pad_index, smoothing=smoothing)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor([
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        ])  # yapf: disable

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1], [2, 0], [1, 0]])

        # test the smoothing function
        # pylint: disable=protected-access
        smoothed_targets = criterion._smooth_targets(
            targets=targets.view(-1), vocab_size=predict.size(-1)
        )
        # pylint: enable=protected-access
        torch.testing.assert_close(
            smoothed_targets,
            torch.Tensor([
                [0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                [0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ]),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertEqual(torch.max(smoothed_targets), 1 - smoothing)

        # test the loss computation

        v, = criterion(predict.log(), **{"trg": targets})
        self.assertAlmostEqual(v.item(), 2.1326, places=4)

    def test_no_label_smoothing(self):
        pad_index = 0
        smoothing = 0.0
        criterion = XentLoss(pad_index=pad_index, smoothing=smoothing)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor([
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        ])  # yapf: disable

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1], [2, 0], [1, 0]])

        # test the smoothing function: should still be one-hot
        # pylint: disable=protected-access
        smoothed_targets = criterion._smooth_targets(
            targets=targets.view(-1), vocab_size=predict.size(-1)
        )
        # pylint: enable=protected-access

        self.assertEqual(torch.max(smoothed_targets), 1)
        self.assertEqual(torch.min(smoothed_targets), 0)

        torch.testing.assert_close(
            smoothed_targets,
            torch.Tensor([
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            rtol=1e-4,
            atol=1e-4,
        )

        v, = criterion(predict.log(), **{"trg": targets})
        self.assertAlmostEqual(v.item(), 5.6268, places=4)
