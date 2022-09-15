import unittest

import torch

from joeynmt.encoders import (
    Conv1dSubsampler,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class TestTransformerEncoder(unittest.TestCase):

    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
        self.alpha = 1.0
        self.layer_norm = "pre"
        self.seed = 42
        torch.manual_seed(self.seed)

    def test_transformer_encoder_freeze(self):
        encoder = TransformerEncoder(freeze=True)
        for _, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_encoder_forward(self):
        batch_size = 2
        time_dim = 4
        torch.manual_seed(self.seed)

        encoder = TransformerEncoder(
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            alpha=self.alpha,
            layer_norm=self.layer_norm,
        )

        for p in encoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        x = torch.rand(size=(batch_size, time_dim, self.emb_size))

        # no padding, no mask
        x_length = torch.Tensor([time_dim] * batch_size).int()
        mask = torch.ones([batch_size, 1, time_dim]) == 1

        output, hidden, _ = encoder(x, x_length, mask)

        self.assertEqual(output.shape,
                         torch.Size([batch_size, time_dim, self.hidden_size]))
        self.assertEqual(hidden, None)

        # yapf: disable
        output_target = torch.Tensor([
            [[1.9728e-01, -1.2042e-01, 8.0998e-02, 1.3411e-03, -3.5960e-01,
              -5.2988e-01, -5.6056e-01, -3.5297e-01, 2.6680e-01, 2.8343e-01,
              -3.7342e-01, -5.9112e-03],
             [8.9687e-02, -1.2491e-01, 7.7809e-02, -1.3500e-03, -2.7002e-01,
              -4.7312e-01, -5.7981e-01, -4.1998e-01, 1.0457e-01, 2.9726e-01,
              -3.9461e-01, 8.1598e-02],
             [3.4988e-02, -1.3020e-01, 6.0043e-02, 2.7782e-02, -3.1483e-01,
              -3.8940e-01, -5.5557e-01, -5.9540e-01, -2.9808e-02, 3.1468e-01,
              -4.5809e-01, 4.3313e-03],
             [1.2234e-01, -1.3285e-01, 6.3068e-02, -2.3343e-02, -2.3519e-01,
              -4.0794e-01, -5.6063e-01, -5.5484e-01, -1.1272e-01, 3.0103e-01,
              -4.0983e-01, 3.3038e-02]],
            [[9.8597e-02, -1.2121e-01, 1.0718e-01, -2.2644e-02, -4.0282e-01,
              - 4.2646e-01, -5.9981e-01, -3.7200e-01, 1.9538e-01, 2.7036e-01,
              -3.4072e-01, -1.7965e-03],
             [8.8470e-02, -1.2618e-01, 5.3351e-02, -1.8531e-02, -3.3834e-01,
              -4.9047e-01, -5.7063e-01, -4.9790e-01, 2.2070e-01, 3.3964e-01,
              -4.1604e-01, 2.3519e-02],
             [5.8373e-02, -1.2706e-01, 1.0598e-01, 9.3255e-05, -3.0493e-01,
              -4.4406e-01, -5.4723e-01, -5.2214e-01, 8.0374e-02, 2.6307e-01,
              -4.4571e-01, 8.7052e-02],
             [7.9567e-02, -1.2977e-01, 1.1731e-01, 2.6198e-02, -2.4024e-01,
              -4.2161e-01, -5.7604e-01, -7.3298e-01, 1.6698e-01, 3.1454e-01,
              -4.9189e-01, 2.4027e-02]],
        ])
        torch.testing.assert_close(output, output_target, rtol=1e-4, atol=1e-4)

        for layer in encoder.layers:
            self.assertTrue(isinstance(layer, TransformerEncoderLayer))
            self.assertTrue(hasattr(layer, "src_src_att"))
            self.assertTrue(hasattr(layer, "feed_forward"))
            self.assertEqual(layer.alpha, self.alpha)
            self.assertEqual(layer.size, self.hidden_size)
            self.assertEqual(layer.feed_forward.pwff_layer[0].in_features,
                             self.hidden_size)
            self.assertEqual(layer.feed_forward.pwff_layer[0].out_features,
                             self.ff_size)
            # pylint: disable=protected-access
            self.assertEqual(layer._layer_norm_position, self.layer_norm)


class TestSubsampler(unittest.TestCase):

    def setUp(self):
        self.hidden_size = 12
        self.in_channels = 10
        self.conv_channels = 24
        self.conv_kernel_sizes = [3, 3]
        self.seed = 42
        torch.manual_seed(self.seed)

    def test_subsampler_forward(self):
        batch_size = 2
        time_dim = 9

        subsampler = Conv1dSubsampler(
            in_channels=self.in_channels,
            mid_channels=self.conv_channels,
            out_channels=self.hidden_size,
            kernel_sizes=self.conv_kernel_sizes,
        )

        for p in subsampler.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        x = torch.rand(size=(batch_size, time_dim, self.in_channels))
        x_length = torch.Tensor([time_dim] * batch_size).int()
        x, x_length = subsampler(x, x_length)

        # x shape [batch_size, seq_len, emb_dim]: [2, 9, 10] -> [2, 3, 12]
        self.assertEqual(x.size(), torch.Size([batch_size, 3, self.hidden_size]))

        # yapf: disable
        x_target = torch.tensor([
            [[-0.4831, -0.0188, -0.0643, 0.2323, 0.1843, -0.0599, 0.0333,
              -0.0295, 0.0926, 0.0629, 0.4416, -0.3737],
             [-0.0230, 0.0513, -0.2007, -0.2211, 0.7072, 0.0523, -0.0546,
              0.0382, -0.0606, -0.8240, -0.3379, -0.7052],
             [0.0229, 0.1770, -0.2644, -0.5954, 0.8251, -0.0118, -0.0228,
              -0.2697, 0.1242, 0.1570, -0.2263, -0.9022]],
            [[-0.4647, 0.0986, -0.1160, 0.0453, 0.2717, -0.0112, 0.0018,
              0.0935, 0.2077, -0.2647, 0.3621, -0.4435],
             [0.0116, -0.1874, -0.0305, -0.5209, 0.7063, -0.0522, 0.0577,
              0.4307, 0.1027, -0.1947, 0.0964, -0.8076],
             [-0.2909, -0.0827, -0.1345, -0.4011, 0.4482, 0.4247, 0.2187,
              -0.2467, 0.0096, -0.2841, 0.0799, -1.2243]],
        ])
        torch.testing.assert_close(x, x_target, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(x_length, torch.tensor([3, 3]))
