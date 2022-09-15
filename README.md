# &nbsp; ![Joey-NMT](joey2-small.png) Joey S2T
[![build](https://github.com/may-/joeys2t/actions/workflows/main.yml/badge.svg)](https://github.com/may-/joeys2t/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


JoeyS2T is an extention of [JoeyNMT](https://github.com/joeynmt/joeynmt) for Speech-to-Text tasks.

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
- torch 1.12.1
- cuda 11.6

Clone this repository and install via pip:
```bash
$ git clone https://github.com/may-/joeys2t.git
$ cd joeynmt
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


## Contact
Please leave an issue if you have found a bug in the code.

For general questions, email me at `ohta <at> cl.uni-heidelberg.de`.

