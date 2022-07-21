# &nbsp; ![Joey-NMT](joey2-small.png) Joey S2T
[![build](https://github.com/may-/joeynmt/actions/workflows/main.yml/badge.svg)](https://github.com/may-/joeynmt/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


JoeyS2T is an extention of [JoeyNMT](https://github.com/joeynmt/joeynmt) for Speech-to-Text tasks.

## Features
Joey S2T implements the following features:
- Transformer Encoder-Decoder
- 1d-Conv Subsampling
- BPE tokenization (with BPE dropout option)
- Mel filterbank spectrogram extraction
- CMVN, SpecAugment
- WER evaluation

Furthermore, all the functionalities in JoeyNMT v2.0 are also available from JoeyS2T:
- BLEU and ChrF evaluation
- Beam search and greedy decoding (with repetition penalty, ngram blocker)
- Customizable initialization
- Attention visualization
- Learning curve plotting
- Scoring hypotheses and references



## Installation
JoeyS2T is built on [PyTorch](https://pytorch.org/). Please make sure you have a compatible environment.
We tested Joey NMT with
- python 3.10
- torch 1.11.0
- cuda 11.5

Clone this repository and install via pip:
```bash
$ git clone https://github.com/may-/joeys2t.git
$ cd joeynmt
$ pip install . -e
```
Run the unit tests:
```bash
$ python -m unittest
```

**[Optional]** For fp16 training, install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Documentation & Tutorials

Please check the JoeyNMT's [documentation](https://joeynmt.readthedocs.io) first, if you are not familiar with JoeyNMT yet.

For details, follow the tutorials in [notebooks](notebooks) dir.
- [quick-start-with-joeynmt2](notebooks/quick-start-with-joeynmt2.ipynb)
- [speech-to-text-with-joeynmt2](notebooks/joeyS2T_ASR_tutorial.ipynb) 



## Benchmarks & pretrained models


### LibriSpeech

JoeyS2T requires tsv format input file to feed the data. You can get the tsv input file using the following script:
```
$ python scripts/prepare_librispeech.py --data_root data/LibriSpeech
```
Then you can speify the path to the tsv file generated above in your config.yaml and start the training:
```
$ python -m joeynmt train configs/librispeech_100h.yaml
$ python -m joeynmt train configs/librispeech_960h.yaml
```

#### LibriSpeech 100h

System | Architecture | dev-clean | dev-other | test-clean | test-other | #params | download
------ | :----------: | :-------- | --------: | ---------: | ---------: | ------: | :-------
[Kahn etal](https://arxiv.org/abs/1909.09116) | BiLSTM | 14.00 | 37.02 | 14.85 | 39.95 | - | -
[Laptev etal](https://arxiv.org/abs/2005.07157) | Transformer | 10.3 | 24.0 | 11.2 | 24.9 | - | -
[ESPnet](https://github.com/espnet/espnet/tree/master/egs2/librispeech_100/asr1#asr_transformer_win400_hop160_ctc03_lr2e-3_warmup15k_timemask5_amp_no-deterministic) | Transformer | 8.1 | 20.2 | 8.4 | 20.5 | - | -
[ESPnet](https://github.com/espnet/espnet/tree/master/egs2/librispeech_100/asr1#asr_conformer_win400_hop160_ctc03_lr2e-3_warmup15k_timemask5_amp_no-deterministic) | Conformer | 6.3 | 17.0 | 6.5 | 17.3 | - | -
JoeyS2T | Transformer | 10.18 | 23.39 | 11.58 | 24.31 | 93M | -

#### LibriSpeech 960h

System | Architecture | dev-clean | dev-other | test-clean | test-other | #params | download
------ | :----------: | :-------- | --------: | ---------: | ---------: | ------: | :-------
[Gulati etal](https://arxiv.org/abs/2005.08100) | BiLSTM | 1.9 | 4.4 | 2.1 | 4.9 | - | -
[ESPnet](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1#without-lm) | Conformer | 2.3 | 6.1 | 2.6 | 6.0 | - | -
[SpeechBrain](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | Conformer |  | 5.51 | 2.31 | 5.61 | 165M | -
[facebook S2T](https://huggingface.co/facebook/s2t-small-librispeech-asr) | Transformer | 3.23 | 8.01 | 3.52 | 7.83 | 71M | -
[facebook wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h) | Conformer | 3.17 | 8.87 | 3.39 | 8.57 | 94M | -
JoeyS2T | Transformer | 3.50 | 8.44 | 3.78 | 8.32 | 102M | -

*We compute the WER on lowercased transcriptions without punctuations using sacrebleu's 13a tokenizer.




## Contact
Please leave an issue if you have questions or issues with the code.

For general questions, email me at `ohta <at> cl.uni-heidelberg.de`. :love_letter:



