# &nbsp; ![Joey-NMT](joey2-small.png) Joey S2T
[![build](https://github.com/may-/joeys2t/actions/workflows/main.yml/badge.svg)](https://github.com/may-/joeys2t/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


JoeyS2T is an extention of [JoeyNMT](https://github.com/joeynmt/joeynmt) for Speech-to-Text tasks.


## What's new
- Upgraded to JoeyNMT v2.2.
- Our paper has been accepted at [EMNLP 2022](https://2022.emnlp.org/) System Demo Track!


## Features
Joey S2T implements the following features:
- Transformer Encoder-Decoder
- 1d-Conv Subsampling
- Cross-entropy and CTC joint obvective
- Mel filterbank spectrogram extraction
- CMVN, SpecAugment
- WER evaluation

Furthermore, all the functionalities in JoeyNMT v2 are also available from JoeyS2T:
- BLEU and ChrF evaluation
- BPE tokenization (with BPE dropout option)
- Beam search and greedy decoding (with repetition penalty, ngram blocker)
- Customizable initialization
- Attention visualization
- Learning curve plotting
- Scoring hypotheses and references



## Installation

JoeyS2T is built on [PyTorch](https://pytorch.org/). Please make sure you have a compatible environment.
We tested JoeyS2T with
- python 3.10
- torch 1.13.1
- torchaudio 0.13.1
- cuda 11.6

Clone this repository and install via pip:

```bash
$ git clone https://github.com/may-/joeys2t.git
$ cd joeys2t
$ pip install -e .
```

## Documentation & Tutorials

Please check the JoeyNMT's [documentation](https://joeynmt.readthedocs.io) first, if you are not familiar with JoeyNMT yet.

For details, follow the tutorials in [notebooks](notebooks) dir.

- [quick-start-with-joeynmt2](notebooks/quick-start-with-joeynmt2.ipynb)
- [speech-to-text-with-joeynmt2](notebooks/joeyS2T_ASR_tutorial.ipynb)



## Benchmarks & pretrained models

We provide [benchmarks](benchmarks_s2t.md) and pretraind models for Speech recognition (ASR) and speech-to-text translation (ST) with JoeyS2T.

- [ASR on LibriSpeech](benchmarks_s2t.md#librispeech)
- [ST on MuST-C en-de](benchmarks_s2t.md#must-c-v2-en-de)


Models are also available via Torch Hub!
```python
import torch

model = torch.hub.load('may-/joeys2t', 'mustc_v2_ende_st')
translations = model.generate(['test.wav'])
print(translations[0])
# 'Hallo, world!'
```
> :warning: **Attention**
> The 1d-conv layer may raise an error for too short audio inputs.
> (We cannot convolve the frames shorter than the kernel size!)


## Reference
If you use JoeyS2T in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/2210.02545):

```
@inproceedings{ohta-etal-2022-joeys2t,
    title = "{JoeyS2T}: Minimalistic Speech-to-Text Modeling with {JoeyNMT}",
    author = "Ohta, Mayumi and Kreutzer, Julia and Riezler, Stefan",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP): System Demonstrations",
    month = "December",
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2210.02545",
}
```

## Contact
Please leave an issue if you have found a bug in the code.

For general questions, email me at `ohta <at> cl.uni-heidelberg.de`.

