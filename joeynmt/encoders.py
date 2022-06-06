# coding: utf-8
"""
Various encoders
"""
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from joeynmt.constants import PAD_ID
from joeynmt.helpers import freeze_params, lengths_to_padding_mask, pad
from joeynmt.transformer_layers import PositionalEncoding, TransformerEncoderLayer


class Encoder(nn.Module):
    """
    Base encoder class
    """

    # pylint: disable=abstract-method
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs,
    ) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type: RNN type: `gru` or `lstm`.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward(self, embed_src: Tensor, src_length: Tensor,
                                    mask: Tensor) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param embed_src: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        # pylint: disable=unused-argument
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
        # assert mask.shape == embed_src.shape
        assert len(src_length.shape) == 1

    def forward(self, embed_src: Tensor, src_length: Tensor, mask: Tensor,
                **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :param kwargs:
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(embed_src=embed_src,
                                         src_length=src_length,
                                         mask=mask)
        total_length = embed_src.size(1)

        # apply dropout to the rnn input
        embed_src = self.emb_dropout(embed_src)

        packed = pack_padded_sequence(embed_src, src_length.cpu(), batch_first=True)
        output, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden  # pylint: disable=unused-variable

        output, _ = pad_packed_sequence(output,
                                        batch_first=True,
                                        total_length=total_length)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        # pylint: disable=no-member
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden

        assert hidden_concat.size(0) == output.size(0), (
            hidden_concat.size(),
            output.size(),
        )
        return output, hidden_concat, None

    def __repr__(self):
        return f"{self.__class__.__name__}(rnn={self.rnn})"


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self._output_size = hidden_size

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                alpha=kwargs.get("alpha", 1.0),
                layer_norm=kwargs.get("layer_norm", "post"),
            ) for _ in range(num_layers)
        ])

        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6) \
            if kwargs.get("layer_norm", "post") == "pre" else None

        if freeze:
            freeze_params(self)

        # conv1d subsampling for audio inputs
        self.subsample = kwargs.get("subsample", False)
        if self.subsample:
            self.subsampler = Conv1dSubsampler(
                kwargs["in_channels"], kwargs["conv_channels"], hidden_size,
                kwargs.get("conv_kernel_sizes", [3, 3]))
            self.pad_index = kwargs.get("pad_index", PAD_ID)
            assert self.pad_index is not None

    def forward(
        self,
        embed_src: Tensor,
        src_length: Tensor,
        mask: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, 1, src_len)
        :return:
            - output: hidden states with shape (batch_size, max_length, hidden)
            - None
            - mask
        """
        if self.subsample:
            embed_src, src_length = self.subsampler(embed_src, src_length)

        if mask is None:
            mask = lengths_to_padding_mask(src_length).unsqueeze(1)

        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if kwargs.get('pad', False) and "src_max_len" in kwargs and self.subsample:
            x = pad(x, kwargs["src_max_len"], pad_index=self.pad_index, dim=1)
            mask = pad(mask, kwargs["src_max_len"], pad_index=self.pad_index, dim=-1)
        assert src_length.size() == (x.size(0),), (src_length.size(), x.size())
        assert mask is not None
        return x, None, mask

    def __repr__(self):
        return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
                f"num_heads={self.layers[0].src_src_att.num_heads}, "
                f"alpha={self.layers[0].alpha}, "
                f'layer_norm="{self.layers[0]._layer_norm_position}", '
                f'subsample={self.subsample})')


class Conv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension)
    followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    cf.) https://github.com/pytorch/fairseq/blob/main/fairseq/models/speech_to_text/s2t_transformer.py

    :param in_channels: the number of input channels (embed_size = num_freq)
    :param mid_channels: the number of intermediate channels
    :param out_channels: the number of output channels (hidden_size)
    :param kernel_sizes: the kernel size for each convolutional layer
    :return:
        - output tensor
        - sequence length after subsampling
    """

    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int = None,
                 kernel_sizes: List[int] = (3, 3)):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for k in self.kernel_sizes:
            out = ((out.float()+2*(k//2)-(k-1)-1)/2+1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        # reshape after DataParallel batch split
        max_len = torch.max(src_lengths).item()
        if src_tokens.size(1) != max_len:
            src_tokens = src_tokens[:, :max_len, :]
        assert src_tokens.size(1) == max_len, (src_tokens.size(), max_len, src_lengths)

        _, in_seq_len, _ = src_tokens.size()  # -> B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous() # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).contiguous()  # -> B x T x (C x D)
        out_seq_lens = self.get_out_seq_lens_tensor(src_lengths)

        assert x.size(1) == torch.max(out_seq_lens).item(), \
            (x.size(), in_seq_len, out_seq_len, out_seq_lens)
        return x, out_seq_lens
