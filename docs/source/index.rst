.. _index:

====================================
Welcome to Joey S2T's documentation!
====================================

**JoeyS2T** is a `JoeyNMT <https://github.com/joeynmt/joeynmt>`_ extension for Speech-to-Text tasks such as Automatic Speech Recognition (ASR) and end-to-end Speech Translation (ST). It inherits the core philosophy of **JoeyNMT**, a minimalist NMT toolkit built on PyTorch, seeking **simplicity** and **accessibility**.


**JoeyS2T** implements the following features:

- Transformer Encoder-Decoder
- 1d-Conv Subsampling
- Cross-entropy and CTC joint objective
- Mel filterbank spectrogram extraction
- CMVN, SpecAugment
- WER evaluation

Furthermore, all the functionalities in **JoeyNMT** v2 are also available from **JoeyS2T**:

- BLEU and ChrF evaluation
- BPE tokenization (with BPE dropout option)
- Beam search and greedy decoding (with repetition penalty, ngram blocker)
- Customizable initialization
- Attention visualization
- Learning curve plotting
- Scoring hypotheses and references
- Multilingual translation with language tags



If you use **JoeyS2T** in a publication or thesis, please cite the following `paper <https://arxiv.org/abs/2210.02545>`_:

::

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


.. toctree::
    :hidden:
    :caption: Getting Started
    :maxdepth: 3

    install
    cli
    tutorial
    benchmarks


.. toctree::
    :hidden:
    :caption: Development
    :maxdepth: 3

    overview
    api
    faq
    resources
    changelog
