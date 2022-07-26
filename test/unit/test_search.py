from test.unit.test_helpers import TensorTestCase

import numpy as np
import torch

from joeynmt.decoders import RecurrentDecoder, TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import RecurrentEncoder
from joeynmt.model import Model
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary


class TestSearch(TensorTestCase):
    # pylint: disable=too-many-instance-attributes
    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
        self.encoder_hidden_size = 3
        self.vocab = Vocabulary(tokens=["word"])
        self.vocab_size = len(self.vocab)  # = 5
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        # self.bos_index = 2
        self.pad_index = 1
        # self.eos_index = 3

        self.expected_transformer_ids = [[5, 5, 5], [5, 5, 5]]
        self.expected_transformer_scores = np.array([[-1.257812, -1.382812, -1.421875],
                                                     [-1.234375, -1.382812, -1.414062]])

        self.expected_recurrent_ids = [[4, 0, 4], [4, 4, 4]]
        self.expected_recurrent_scores = np.array([
            [-1.1953125, -1.21875, -1.2421875],
            [-1.171875, -1.2109375, -1.2109375],
        ])

        # tolerance: cpu -> bfloat16, gpu -> float32
        self.tol = 1e-5 if torch.cuda.is_available() else 0.01


class TestSearchTransformer(TestSearch):

    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 7

        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=vocab_size,
            padding_idx=self.pad_index,
        )

        decoder = TransformerDecoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            vocab_size=vocab_size,
            layer_norm="pre",
        )

        encoder_output = torch.rand(size=(batch_size, src_time_dim, self.hidden_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1

        encoder_hidden = None  # unused

        model = Model(
            encoder=None,
            decoder=decoder,
            src_embed=emb,
            trg_embed=emb,
            src_vocab=self.vocab,
            trg_vocab=self.vocab,
        )
        return src_mask, model, encoder_output, encoder_hidden

    def test_transformer_greedy(self):
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        with torch.autocast(device_type="cpu", enabled=False):
            output, scores, attention_scores = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )
        # Transformer greedy doesn't return attention scores
        # `return_attention = False` by default
        self.assertIsNone(attention_scores)

        # outputs
        self.assertEqual(output.shape, (batch_size, max_output_length))  # batch x time
        np.testing.assert_equal(output, self.expected_transformer_ids)

        # scores
        self.assertEqual(scores.shape, (batch_size, max_output_length))  # batch x time
        np.testing.assert_allclose(scores,
                                   self.expected_transformer_scores,
                                   rtol=self.tol)

    def test_transformer_beam1(self):
        batch_size = 2
        beam_size = 1
        alpha = 0.0
        n_best = 1
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        with torch.autocast(device_type="cpu", enabled=False):
            beam_output, beam_scores, attention_scores = beam_search(
                beam_size=beam_size,
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                alpha=alpha,
                n_best=n_best,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len
        self.assertEqual(beam_output.shape, (batch_size * n_best, max_output_length))
        np.testing.assert_equal(beam_output, self.expected_transformer_ids)
        np.testing.assert_allclose(beam_scores, [[-4.1875], [-4.125]], rtol=self.tol)

        # now compare to greedy, they should be the same for beam=1
        with torch.autocast(device_type="cpu", enabled=False):
            greedy_output, greedy_scores, _ = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )
        np.testing.assert_equal(beam_output, greedy_output)
        np.testing.assert_allclose(
            greedy_scores,
            self.expected_transformer_scores,
            rtol=self.tol,
        )

    def test_transformer_beam7(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        with torch.autocast(device_type="cpu", enabled=False):
            output, scores, attention_scores = beam_search(
                beam_size=beam_size,
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                alpha=alpha,
                n_best=n_best,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len (= time steps)
        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))
        expected_output = [[5, 5, 5], [0, 5, 5], [0, 0, 5], [5, 5, 0], [5, 0, 5],
                           [5, 5, 5], [0, 5, 5], [5, 5, 0], [5, 0, 5], [0, 0, 5]]
        np.testing.assert_equal(output, expected_output)
        expected_scores = [[-3.140625], [-3.28125], [-3.4375], [-3.4375], [-3.46875],
                           [-3.09375], [-3.296875], [-3.42187], [-3.42187], [-3.46875]]
        np.testing.assert_allclose(scores, expected_scores, rtol=self.tol)

    def test_repetition_penalty_and_generate_unk(self):
        batch_size = 3
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        # no repetition penalty
        with torch.autocast(device_type="cpu", enabled=False):
            output, _, _ = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                generate_unk=False,
            )
        expected_output = [[5, 6, 4], [5, 6, 4], [5, 6, 6]]
        np.testing.assert_equal(output, expected_output)
        np.testing.assert_equal(np.count_nonzero(output), 9)  # no unk token generated

        # trg repetition penalty
        with torch.autocast(device_type="cpu", enabled=False):
            output_trg_penalty, _, _ = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                encoder_input=None,
                repetition_penalty=1.5,
                generate_unk=False,
            )
        expected_output_trg_penalty = [[5, 6, 4], [5, 6, 4], [5, 6, 4]]
        np.testing.assert_equal(output_trg_penalty, expected_output_trg_penalty)

        # src + trg repetition penalty
        # src_len = 4 (see self._build())
        src_tokens = torch.tensor([[4, 3, 1, 1], [5, 4, 3, 1], [5, 5, 6, 3]])
        src_mask = (src_tokens != 1).unsqueeze(1)
        with torch.autocast(device_type="cpu", enabled=False):
            output_src_penalty, _, attention = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                encoder_input=src_tokens,
                repetition_penalty=1.5,
                generate_unk=False,
                return_attention=True,
            )
        expected_output_src_penalty = [[5, 6, 4], [6, 4, 6], [4, 5, 6]]
        np.testing.assert_equal(output_src_penalty, expected_output_src_penalty)

        # Transformer Greedy can return attention probs
        expected_attention = np.array(
            [[[0.50390625, 0.49414062, 0.0, 0.0], [0.49804688, 0.5, 0.0, 0.0],
              [0.49804688, 0.50390625, 0.0, 0.0]],
             [[0.3359375, 0.33398438, 0.328125, 0.0],
              [0.33203125, 0.33984375, 0.32617188, 0.0],
              [0.33203125, 0.33984375, 0.328125, 0.0]],
             [[0.25390625, 0.24511719, 0.265625, 0.23632812],
              [0.25390625, 0.24414062, 0.26367188, 0.23828125],
              [0.24511719, 0.24023438, 0.26953125, 0.24609375]]])
        # attention (batch_size, trg_len, src_len) = (3, 3, 4)
        np.testing.assert_allclose(attention, expected_attention, rtol=self.tol)

    def test_repetition_penalty_in_beam_search(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        src_tokens = torch.tensor([[5, 5, 4], [5, 6, 6]])
        expected_output_with_penalty = [
            [6, 0, 5],
            [6, 0, 1],
            [6, 1, 0],
            [6, 0, 0],
            [0, 5, 6],
            [5, 1, 0],
            [1, 0, 5],
            [0, 5, 1],
            [4, 0, 5],
            [0, 5, 3],
        ]
        expected_scores_with_penalty = [
            [-3.953125],
            [-4.03125],
            [-4.125],
            [-4.125],
            [-4.15625],
            [-4.21875],
            [-4.25],
            [-4.3125],
            [-4.3125],
            [-4.34375],
        ]
        with torch.autocast(device_type="cpu", enabled=False):
            output_with_penalty, scores_with_penalty, _ = beam_search(
                beam_size=beam_size,
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                alpha=alpha,
                n_best=n_best,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                encoder_input=src_tokens,
                repetition_penalty=1.5,
                return_prob="hyp",
            )

        np.testing.assert_equal(output_with_penalty, expected_output_with_penalty)
        np.testing.assert_allclose(scores_with_penalty,
                                   expected_scores_with_penalty,
                                   rtol=self.tol)

    def test_ngram_blocker(self):
        batch_size = 2
        max_output_length = 10
        no_repeat_ngram_size = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        expected_output = [[5, 5, 5, 0, 5, 5, 0, 0, 5, 0],
                           [5, 5, 5, 0, 5, 5, 0, 0, 5, 0]]
        # yapf: disable
        expected_scores = [[-1.257812, -1.382812, -1.421875, -1.78125, -1.359375,
                            -1.40625, -1.835938, -1.46875, -1.382812, -1.828125],
                           [-1.234375, -1.382812, -1.414062, -1.796875, -1.34375,
                            -1.398438, -1.84375, -1.507812, -1.359375, -1.835938]]
        with torch.autocast(device_type="cpu", enabled=False):
            output, scores, _ = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                encoder_input=None,
                return_prob="hyp",
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        np.testing.assert_equal(output, expected_output)
        np.testing.assert_allclose(scores, expected_scores, rtol=self.tol)

    def test_ngram_blocker_in_beam_search(self):
        batch_size = 2
        beam_size = 3
        n_best = 3
        alpha = 1.0
        max_output_length = 10
        no_repeat_ngram_size = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        expected_output = [
            [5, 5, 5, 0, 0, 0, 0, 0, 5, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 5, 5],
            [5, 5, 5, 0, 0, 0, 5, 5, 0, 5],
            [5, 5, 5, 0, 0, 0, 5, 5, 0, 0],
        ]
        expected_scores = [[-5.90625], [-5.90625], [-5.90625], [-5.90625], [-5.90625],
                           [-5.9375]]
        with torch.autocast(device_type="cpu", enabled=False):
            output, scores, _ = beam_search(
                beam_size=beam_size,
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                alpha=alpha,
                n_best=n_best,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                encoder_input=None,
                return_prob="hyp",
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        np.testing.assert_equal(output, expected_output)
        np.testing.assert_allclose(scores, expected_scores, rtol=self.tol)


class TestSearchRecurrent(TestSearch):

    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 7

        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=vocab_size,
            padding_idx=self.pad_index,
        )

        encoder = RecurrentEncoder(
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            hidden_size=self.encoder_hidden_size,
            bidirectional=True,
        )

        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=encoder,
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="bridge",
            input_feeding=True,
        )

        encoder_output = torch.rand(size=(batch_size, src_time_dim,
                                          encoder.output_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1

        encoder_hidden = torch.rand(size=(batch_size, encoder.output_size))

        model = Model(
            encoder=encoder,
            decoder=decoder,
            src_embed=emb,
            trg_embed=emb,
            src_vocab=self.vocab,
            trg_vocab=self.vocab,
        )

        return src_mask, model, encoder_output, encoder_hidden

    def test_recurrent_greedy(self):
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        with torch.autocast(device_type="cpu", enabled=False):
            output, scores, attention_scores = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )
        self.assertEqual(output.shape, (batch_size, max_output_length))
        np.testing.assert_equal(output, self.expected_recurrent_ids)
        np.testing.assert_allclose(scores,
                                   self.expected_recurrent_scores,
                                   rtol=self.tol)

        expected_attention_scores = np.array(
            [[[0.22949219, 0.24707031, 0.21386719, 0.31445312],
              [0.22949219, 0.24707031, 0.21386719, 0.31445312],
              [0.22851562, 0.24707031, 0.2109375, 0.31445312]],
             [[0.25390625, 0.2890625, 0.25585938, 0.19921875],
              [0.25195312, 0.2890625, 0.2578125, 0.20117188],
              [0.25390625, 0.2890625, 0.2578125, 0.20019531]]])
        np.testing.assert_array_almost_equal(attention_scores,
                                             expected_attention_scores)
        self.assertEqual(attention_scores.shape, (batch_size, max_output_length, 4))

    def test_recurrent_beam1(self):
        # beam=1 and greedy should return the same result
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        with torch.autocast(device_type="cpu", enabled=False):
            greedy_output, greedy_scores, _ = greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )
        self.assertEqual(greedy_output.shape, (batch_size, max_output_length))
        np.testing.assert_equal(greedy_output, self.expected_recurrent_ids)
        np.testing.assert_allclose(
            greedy_scores,
            self.expected_recurrent_scores,
            rtol=self.tol,
        )

        beam_size = 1
        alpha = 0.0
        n_best = 1
        with torch.autocast(device_type="cpu", enabled=False):
            beam_output, beam_scores, _ = beam_search(
                beam_size=beam_size,
                src_mask=src_mask,
                n_best=n_best,
                max_output_length=max_output_length,
                model=model,
                alpha=alpha,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )
        np.testing.assert_array_equal(greedy_output, beam_output)
        np.testing.assert_allclose(beam_scores,
                                   [[-3.65625], [-3.609375]],
                                   rtol=self.tol)

    def test_recurrent_beam7(self):
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        beam_size = 7
        n_best = 5
        alpha = 1.0
        with torch.autocast(device_type="cpu", enabled=False):
            output, scores, _ = beam_search(
                beam_size=beam_size,
                src_mask=src_mask,
                max_output_length=max_output_length,
                model=model,
                alpha=alpha,
                n_best=n_best,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                return_prob="hyp",
            )

        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        # output indices
        expected_output = [[4, 4, 4], [4, 4, 0], [4, 0, 4], [4, 0, 0], [0, 4, 4],
                           [4, 4, 4], [4, 4, 0], [4, 0, 4], [4, 0, 0], [0, 4, 4]]
        np.testing.assert_array_equal(output, expected_output)

        # log probabilities
        expected_scores = [
            [-2.71875],
            [-2.734375],
            [-2.734375],
            [-2.765625],
            [-2.859375],
            [-2.703125],
            [-2.71875],
            [-2.765625],
            [-2.8125],
            [-2.875],
        ]
        np.testing.assert_allclose(scores, expected_scores, rtol=self.tol)
