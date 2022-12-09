"""PyTorch Hub models

Usage:
    import torch
    model = torch.hub.load('repo', 'model')
"""

from pathlib import Path
import importlib
import logging
import tarfile

import torch.hub

dependencies = ['torch', 'torchaudio', 'yaml', 'numpy', 'matplotlib', 'pandas',
                'sentencepiece', 'subword_nmt', 'sacremoses', 'editdistance']

# Check for required dependencies and raise a RuntimeError if any are missing.
missing_deps = []
for dep in dependencies:
    try:
        importlib.import_module(dep)
    except ImportError:
        missing_deps.append(dep)
if len(missing_deps) > 0:
    raise RuntimeError("Missing dependencies: {}".format(", ".join(missing_deps)))


# only do joeynmt imports after checking dependencies
from joeynmt.hub_interface import _from_pretrained, TranslatorHubInterface

ROOT_URL = "https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2"

import logging
logger = logging.getLogger(__name__)


def _download_and_extract(
    model_name_or_path: str,
    ext: str = ".tar.gz",
    **kwargs
) -> Path:
    hub_dir = Path(torch.hub.get_dir())
    download_dir = hub_dir / model_name_or_path
    force_reload = kwargs.get('force_reload', False)
    if (not force_reload) and download_dir.is_dir():
        logger.info('Model cache found in %s. Skip redownload.', download_dir)
        return download_dir

    torch.hub.download_url_to_file(f"{ROOT_URL}/{model_name_or_path}{ext}",
                                   download_dir.with_suffix(ext))
    # extract
    if ext.startswith(".tar"):
        with tarfile.open(download_dir.with_suffix(ext)) as f:
            f.extractall(hub_dir)
    # delete .tar.gz
    download_dir.with_suffix(ext).unlink()
    assert download_dir.is_dir(), download_dir
    return download_dir


def _load_from_remote(
    model_name_or_path: str,
    ckpt_file: str = "best.ckpt",
    cfg_file: str = "config.yaml",
    **kwargs
) -> TranslatorHubInterface:
    download_dir = _download_and_extract(model_name_or_path, **kwargs)
    config, test_data, model = _from_pretrained(
        model_name_or_path=download_dir,
        ckpt_file=ckpt_file,
        cfg_file=cfg_file,
        **kwargs,
    )
    return TranslatorHubInterface(config, test_data, model)


def transformer_iwslt14_deen_bpe(*args, **kwargs) -> TranslatorHubInterface:
    """
    IWSLT14 deen transformer
    See
    """
    return _load_from_remote(
        model_name_or_path="transformer_iwslt14_deen_bpe",
        ckpt_file="best.ckpt",
        cfg_file="config_v2.yaml",
        **kwargs
    )


def rnn_iwslt14_deen_bpe(*args, **kwargs) -> TranslatorHubInterface:
    """
    IWSLT14 deen RNN
    See
    """
    return _load_from_remote(
        model_name_or_path="rnn_iwslt14_deen_bpe",
        ckpt_file="best.ckpt",
        cfg_file="config_v2.yaml",
        **kwargs
    )


def wmt14_deen(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 deen
    See: https://github.com/may-/datasets/blob/main/datasets/wmt14/wmt14.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_deen",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_deen_it_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 deen + fine-tuned on IT data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_deen_it_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_deen_koran_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 deen + fine-tuned on Koran data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_deen_koran_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_deen_law_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 deen + fine-tuned on Law data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_deen_law_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_deen_medical_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 deen + fine-tuned on Medical data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_deen_medical_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_deen_subtitles_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 deen + fine-tuned on Subtitles data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_deen_subtitles_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_ende(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 ende
    See: https://github.com/may-/datasets/blob/main/datasets/wmt14/wmt14.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_ende",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_ende_it_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 ende + fine-tuned on IT data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_ende_it_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_ende_koran_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 ende + fine-tuned on Koran data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_ende_koran_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_ende_law_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 ende + fine-tuned on Law data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_ende_law_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_ende_medical_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 ende + fine-tuned on Medical data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_ende_medical_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_ende_subtitles_finetune(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 ende + fine-tuned on Subtitles data
    See: https://github.com/may-/datasets/blob/main/datasets/damt/damt.py
    """
    return _load_from_remote(
        model_name_or_path="wmt14_ende_subtitles_finetune",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def librispeech_960h_en_asr(*args, **kwargs) -> TranslatorHubInterface:
    """
    English ASR model trained on Librispeech 960h data
    See: https://github.com/may-/datasets/blob/main/datasets/librispeech_asr/librispeech_asr.py
    """
    return _load_from_remote(
        model_name_or_path="librispeech960h",
        ckpt_file="avg10.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def librispeech_100h_en_asr(*args, **kwargs) -> TranslatorHubInterface:
    """
    English ASR model trained on Librispeech 100h data
    See: https://github.com/may-/datasets/blob/main/datasets/librispeech_asr/librispeech_asr.py
    """
    return _load_from_remote(
        model_name_or_path="librispeech100h",
        ckpt_file="avg10.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def mustc_v2_en_asr(*args, **kwargs) -> TranslatorHubInterface:
    """
    English ASR model trained on MuST-C v2 data
    See: https://github.com/may-/datasets/blob/main/datasets/mustc/mustc.py
    """
    return _load_from_remote(
        model_name_or_path="mustc_asr",
        ckpt_file="avg10.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def mustc_v2_ende_mt(*args, **kwargs) -> TranslatorHubInterface:
    """
    English-German MT model trained on MuST-C v2 data
    See: https://github.com/may-/datasets/blob/main/datasets/mustc/mustc.py
    """
    return _load_from_remote(
        model_name_or_path="mustc_mt",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def mustc_v2_ende_st(*args, **kwargs) -> TranslatorHubInterface:
    """
    English-German ST model trained on MuST-C v2 data
    See: https://github.com/may-/datasets/blob/main/datasets/mustc/mustc.py
    """
    return _load_from_remote(
        model_name_or_path="mustc_st",
        ckpt_file="avg10.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def local(model_name_or_path, ckpt_file, cfg_file, **kwargs) -> TranslatorHubInterface:
    """
    joeynmt model saved in local
    """
    config, test_data, model = _from_pretrained(
        model_name_or_path=model_name_or_path,
        ckpt_file=ckpt_file,
        cfg_file=cfg_file,
        **kwargs,
    )
    return TranslatorHubInterface(config, test_data, model)


def jparacrawl_enja(*args, **kwargs) -> TranslatorHubInterface:
    """
    JParaCrawl enja
    See: https://github.com/may-/datasets/blob/main/datasets/jparacrawl/jparacrawl.py
    """
    return _load_from_remote(
        model_name_or_path="jparacrawl_enja",
        ckpt_file="avg5.ckpt",
        cfg_file="config_v2.yaml",
        **kwargs
    )


def jparacrawl_jaen(*args, **kwargs) -> TranslatorHubInterface:
    """
    JParaCrawl jaen
    See: https://github.com/may-/datasets/blob/main/datasets/jparacrawl/jparacrawl.py
    """
    return _load_from_remote(
        model_name_or_path="jparacrawl_enja",
        ckpt_file="avg5.ckpt",
        cfg_file="config_v2.yaml",
        **kwargs
    )

if __name__ == '__main__':
    translator = transformer_iwslt14_deen_bpe()
    print(translator.translate(['Hello World!']))
