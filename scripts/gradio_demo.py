import numpy as np
import gradio as gr
import torchaudio
import torch


title = "JoeyS2T Demo"
description = "JoeyS2T is an extention of [JoeyNMT](https://github.com/joeynmt/joeynmt) for Speech-to-Text tasks. ![JoeyS2T](https://raw.githubusercontent.com/may-/joeys2t/main/joey2-small.png)"
article = "The paper is available on [arxiv](https://arxiv.org/abs/2210.02545)!"

# load models
model_names = ["librispeech_960h_en_asr", "mustc_v2_en_asr", "mustc_v2_en_mt", "mustc_v2_en_st"]
models = {}
for m in model_names:
    models[m] = torch.hub.load('may-/joeys2t', m)
new_sample_rate = 16000
resample = torchaudio.transforms.Resample(
    sample_rate, downsample_rate, resampling_method='sinc_interpolation')


def recognize(model_choice, tmp_file):
    # resample
    audio_file = tmp_file.name
    waveform, old_sample_rate = torchaudio.load(audio_file)
    resampled = resample(waveform)
    torchaudio.save(audio_file, torch.clamp(resampled, -1, 1), new_sample_rate)

    # predict
    if model_choice == "mustc_v2_en_mt":
        transcription = model["mustc_v2_en_asr"].generate([audio_file])
        translation = model[model_choice].generate([transcription])
        prediction = transcription + "\n" + translation
    else:
        prediction = model[model_choice].generate([audio_file])
    return prediction


select = gr.inputs.Dropdown([model_names])
mic = gr.Audio(source="microphone", type="file", label="Speak here...")
demo = gr.Interface(recognize, [select, mic], "text",
                    title=title, description=description, article=article)
demo.launch()