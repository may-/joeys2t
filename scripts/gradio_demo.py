# coding: utf-8
"""
JoeyS2T Demo
"""
import logging

import gradio as gr
import numpy as np
import torch

from joeynmt.helpers_for_audio import extract_fbank_features

logger = logging.getLogger(__name__)

title = "JoeyS2T Demo @ EMNLP 2022"
description =\
    '**JoeyS2T** is a [JoeyNMT](https://github.com/joeynmt/joeynmt) extension ' \
    + 'for Speech-to-Text tasks such as ASR and End-to-End ST. ' \
    + '![JoeyS2T](https://raw.githubusercontent.com/may-/joeys2t/main/joey2-small.png)'

article = "The paper is available on [arxiv](https://arxiv.org/abs/2210.02545)!"

# load models
model_names = [
    "librispeech_960h_en_asr",
    "mustc_v2_en_asr",
    "mustc_v2_ende_mt",
    "mustc_v2_ende_st",
]
models = {}
for m in model_names:
    models[m] = torch.hub.load('may-/joeys2t', m)


def reformat_freq(sr, y):
    if sr not in (
        48000,
        16000,
    ):  # we convert 48k -> 16k
        raise ValueError("Unsupported rate", sr)
    if sr == 48000:
        y = (((y / max(np.max(y), 1)) * 32767).reshape((-1, 3)).mean(axis=1
                                                                     ).astype("int16"))
        sr = 16000
    return y, sr


def recognize(model_choice, speech):
    # resample
    waveform, sample_rate = reformat_freq(*speech)
    logger.info(waveform.shape, sample_rate)
    features = extract_fbank_features(
        torch.tensor(waveform[np.newaxis, :]).float(), sample_rate
    )
    np.save('demo.npy', features)

    # predict
    kwargs = {"beam_size": 5, "return_prob": "none"}
    if model_choice == "mustc_v2_ende_mt":
        transcriptions = models["mustc_v2_en_asr"].generate(['demo.npy'], **kwargs)
        translations = models[model_choice].generate(transcriptions)
        predictions = [transcriptions[0] + "\n" + translations[0]]
    else:
        predictions = models[model_choice].generate(['demo.npy'], **kwargs)
    logger.info(model_choice, predictions[0])
    return predictions[0]


# pylint: disable=unexpected-keyword-arg
select = gr.Dropdown(model_names)
mic = gr.Audio(source="microphone", type="numpy", label="Speak here...")
demo = gr.Interface(
    fn=recognize,
    inputs=[select, mic],
    outputs="text",
    allow_flagging="auto",
    title=title,
    description=description,
    article=article,
    server_name="0.0.0.0",
    ssl_certfile="/path/to/cert.pem",
    ssl_keyfile="/path/to/key.pem"
)
demo.launch()
