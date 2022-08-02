# Benchmarks

Here we provide benchmarks for reference. You can find all the [scripts](scripts) to replicate these results.
See [benchmarks](notebooks/benchmarks.ipynb) for details.


*Note that the interactive `translate` mode is currently not supported for Speech-to-Text tasks. Please use `test` mode.

## LibriSpeech

**Data Preparation:**
JoeyS2T requires tsv format input file to feed the data. You can get the tsv input file using the following script:
```
$ python scripts/prepare_librispeech.py --data_root data/LibriSpeech
```
Then specify the path to the tsv files generated above in the configuration file.

**Training:**
```
$ python -m joeynmt train configs/librispeech_{100h|960h}.yaml
```

**Inference:**
```
$ python -m joeynmt test configs/librispeech_{100h|960h}.yaml --output_path models/librispeech_{100h|960h}/hyps
```



### LibriSpeech 100h

System | Architecture | dev-clean | dev-other | test-clean | test-other | #params | download
------ | :----------: | :-------- | --------: | ---------: | ---------: | ------: | :-------
[Kahn etal](https://arxiv.org/abs/1909.09116) | BiLSTM | 14.00 | 37.02 | 14.85 | 39.95 | - | -
[Laptev etal](https://arxiv.org/abs/2005.07157) | Transformer | 10.3 | 24.0 | 11.2 | 24.9 | - | -
[ESPnet](https://github.com/espnet/espnet/tree/master/egs2/librispeech_100/asr1#asr_transformer_win400_hop160_ctc03_lr2e-3_warmup15k_timemask5_amp_no-deterministic) | Transformer | 8.1 | 20.2 | 8.4 | 20.5 | - | -
[ESPnet](https://github.com/espnet/espnet/tree/master/egs2/librispeech_100/asr1#asr_conformer_win400_hop160_ctc03_lr2e-3_warmup15k_timemask5_amp_no-deterministic) | Conformer | 6.3 | 17.4 | 6.5 | 17.3 | - | -
JoeyS2T | Transformer | 10.18 | 23.39 | 11.58 | 24.31 | 93M | [librispeech100h.tar.gz](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt2/librispeech100h.tar.gz) (948M)

### LibriSpeech 960h

System | Architecture | dev-clean | dev-other | test-clean | test-other | #params | download
------ | :----------: | :-------- | --------: | ---------: | ---------: | ------: | :-------
[Gulati etal](https://arxiv.org/abs/2005.08100) | BiLSTM | 1.9 | 4.4 | 2.1 | 4.9 | - | -
[ESPnet](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1#without-lm) | Conformer | 2.3 | 6.1 | 2.6 | 6.0 | - | -
[SpeechBrain](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | Conformer | 2.13 | 5.51 | 2.31 | 5.61 | 165M | -
[facebook S2T](https://huggingface.co/facebook/s2t-small-librispeech-asr) | Transformer | 3.23 | 8.01 | 3.52 | 7.83 | 71M | -
[facebook wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h) | Conformer | 3.17 | 8.87 | 3.39 | 8.57 | 94M | -
JoeyS2T | Transformer | 3.50 | 8.44 | 3.78 | 8.32 | 102M | [librispeech960h.tar.gz](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt2/librispeech960h.tar.gz) (1.1G)

*We compute the WER on lowercased transcriptions without punctuations using sacrebleu's 13a tokenizer.


## MuST-C v2 en-de

**Data Preparation:**
First, download the dataset builder [here](https://github.com/may-/datasets/blob/main/datasets/mustc/mustc.py), and follow the instruction there to download the [data](https://ict.fbk.eu/must-c/).
Second, run the following preparation script and generate the input tsv files.
```
$ python scripts/prepare_mustc.py --data_root data/MuSTC_v2.0 --trg_lang de
```
Then specify the path to the tsv files generated above in the configuration file.

**Training:**
```
$ python -m joeynmt train configs/mustc_{asr|mt|st}.yaml
```

**Inference:**
```
$ python -m joeynmt test configs/mustc_{asr|mt|st}.yaml --output_path models/mustc_{asr|mt|st}/hyps
```


### MuST-C ASR pretraining (WER)

System | train | eval | dev | tst-COMMON | tst-HE | #params | download
------ | :---: | :--: | --: | ---------: | -----: | ------: | :-------
[Gangi etal](https://cris.fbk.eu/retrieve/handle/11582/319654/29817/3045.pdf) | v1 | v1 | - | 27.0 | - | - | -
[ESPnet](https://github.com/espnet/espnet/blob/master/egs/must_c/asr1/RESULTS.md) | v1 | v1 | - | 12.70 | - | - | -
[fairseq S2T](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md) | v1 | v1 | 13.07 | 12.72 | 10.93 | 29.5M | -
JoeyS2T | v2 | v1 | 18.09 | 18.66 | 14.97 | 96M | -
[fairseq S2T](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md) | v1 | v2 | 9.11 | 11.88 | 10.43 | 29.5M | -
JoeyS2T | v2 | v2 | 9.77 | 12.51 | 10.73 | 96M | [mustc_asr.tar.gz](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt2/mustc_asr.tar.gz) (940M)



### MuST-C MT pretraining (BLEU)

System | train | eval | dev | tst-COMMON | tst-HE | #params | download
------ | :---: | :--: | --: | ---------: | -----: | ------: | :-------
[Gangi etal](https://cris.fbk.eu/retrieve/handle/11582/319654/29817/3045.pdf) | v1 | v1 | - | 25.3 | - | - | -
[Zhang etal](https://aclanthology.org/2020.findings-emnlp.230/) | v1 | v1 | - | 29.69 | - | - | -
[ESPnet](https://github.com/espnet/espnet/blob/master/egs/must_c/mt1/RESULTS.md) | v1 | v1 | - | 27.63 | - | - | -
JoeyS2T | v2 | v1 | 21.85 | 23.15 | 20.37 | 66.5M | -
JoeyS2T | v2 | v2 | 26.99 | 27.61 | 25.26 | 66.5M | [mustc_mt.tar.gz](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt2/mustc_mt.tar.gz) (729M)


### MuST-C end-to-end ST (BLEU)

System | train | eval | dev | tst-COMMON | tst-HE | #params | download
------ | :---: | :--: | --: | ---------: | -----: | ------: | :-------
[Gangi etal](https://cris.fbk.eu/retrieve/handle/11582/319654/29817/3045.pdf) | v1 | v1 | - | 17.3 | - | - | -
[Zhang etal](https://aclanthology.org/2020.findings-emnlp.230/) | v1 | v1 | - | 20.67 | - | - | -
[ESPnet](https://github.com/espnet/espnet/blob/master/egs/must_c/st1/RESULTS.md) | v1 | v1 | - | 22.91 | - | - | -
[fairseq S2T](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md) | v1 | v1 | 22.05 | 22.70 | 21.70 | 31M | -
JoeyS2T | v2 | v1 | 21.06 | 20.92 | 21.78 | 96M | -
[fairseq S2T](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md) | v1 | v2 | 23.38 | 23.20 | 22.23 | 31M | -
JoeyS2T | v2 | v2 | 24.26 | 23.86 | 23.86 | 96M | [mustc_st.tar.gz](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt2/mustc_st.tar.gz) (952M)


*sacrebleu signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.1.0`
