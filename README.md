# &nbsp; ![Joey-S2T](joey2-small.png) Joey S2T
[![build](https://github.com/may-/joeys2t/actions/workflows/main.yml/badge.svg)](https://github.com/may-/joeys2t/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2210.02545-b31b1b.svg)](https://arxiv.org/abs/2210.02545)


JoeyS2T is a [JoeyNMT](https://github.com/joeynmt/joeynmt) extension for Speech-to-Text tasks such as Automatic Speech Recognition (ASR) and end-to-end Speech Translation (ST). It inherits the core philosophy of JoeyNMT, a minimalist novice-friendly toolkit built on PyTorch, seeking **simplicity** and **accessibility**.


## What's new
- Upgraded to JoeyNMT v2.3.
- Our paper has been accepted at [EMNLP 2022](https://2022.emnlp.org/) System Demo Track!


## Features
JoeyS2T implements the following features:
- Transformer Encoder-Decoder
- 1d-Conv Subsampling
- Cross-entropy and CTC joint objective
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
- Multilingual translation with language tags


## Installation

JoeyS2T is built on [PyTorch](https://pytorch.org/). Please make sure you have a compatible environment.
We tested JoeyS2T v2.3 with
- python 3.11
- torch 2.1.2
- torchaudio 2.1.2
- cuda 12.1

Clone this repository and install via pip:

```bash
$ git clone https://github.com/may-/joeys2t.git
$ cd joeys2t
$ python -m pip install -e .
$ python -m unittest
```

> :memo: **Note**
> You may need to install extra dependencies (torchaudio backends): [ffmpeg](https://ffmpeg.org/), [sox](https://sox.sourceforge.net/), [soundfile](https://pysoundfile.readthedocs.io/), etc.
> See [torchaudio installation instructions](https://pytorch.org/audio/stable/installation.html).


## Documentation & Tutorials

Please check the JoeyNMT's [documentation](https://joeys2t.readthedocs.io) first, if you are not familiar with JoeyNMT yet.

For details, follow the tutorials in [notebooks](notebooks) dir.

- [quick-start-with-joeynmt2](notebooks/quick-start-with-joeynmt2.ipynb)
- [speech-to-text-with-joeys2t](notebooks/joeyS2T_ASR_tutorial.ipynb)


## Benchmarks & Pretrained models

We provide [benchmarks](https://joeys2t.readthedocs.io/en/latest/benchmarks.html) and pretraind models for Speech Recognition (ASR) and Speech Translation (ST) with JoeyS2T.

The models are also available via Torch Hub!
```python
import torch

model = torch.hub.load('may-/joeys2t', 'mustc_v2_ende_st')
translations = model.generate(['test.wav'])
print(translations[0])
# 'Hallo, Welt!'
```

> :warning: **Warning**
> The 1d-conv layer may raise an error for too short audio inputs.
> (We cannot convolve the frames shorter than the kernel size!)


## Reference
If you use JoeyS2T in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/2210.02545):

```
@inproceedings{ohta-etal-2022-joeys2t,
    title = "{JoeyS2T}: Minimalistic Speech-to-Text Modeling with {JoeyNMT}",
    author = "Ohta, Mayumi  and
      Kreutzer, Julia  and
      Riezler, Stefan",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-demos.6",
    pages = "50--59",
}
```

## Contact
Please leave an issue if you have found a bug in the code.

For general questions, email me at `ohta <at> cl.uni-heidelberg.de`.

