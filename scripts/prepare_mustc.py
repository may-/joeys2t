#!/usr/bin/env python3
"""
Prepare MUST-C

Adapted from
http://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/prep_mustc_data.py


Depends on Huggingface's datasets library.
As of 16. July 2022, MUST-C is not in the official Huggingface's datasets repo.
Pull the script:
https://github.com/may-/datasets/blob/main/datasets/mustc/mustc.py

Expected dir structure:
    MUSTC_ROOT    # <- point here in --data_root in arg
        └── en-de
            ├── fbank80
            │   ├── ted_767_1.npy
            │   ├── [...]
            │   └── ted_837_1.npy
            ├── fbank80.zip
            ├── joey_train_{asr|st}.tsv
            ├── joey_dev_{asr|st}.tsv
            ├── joey_tst-COMMON_{asr|st}.tsv
            ├── joey_tst-HE_{asr|st}.tsv
            ├── {train|dev|tst-COMMON|tst-HE}.en
            └── {train|dev|tst-COMMON|tst-HE}.de
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from audiodata_utils import (
    Normalizer,
    build_sp_model,
    create_zip,
    get_zip_manifest,
    save_tsv,
)
from datasets import DatasetDict, load_dataset

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features

COLUMNS = ["id", "src", "n_frames", "trg", "speaker"]
SPLITS = ["train", "validation", "tst.COMMON", "tst.HE"]

N_MEL_FILTERS = 80
N_WORKERS = 16  # cpu_count()
SP_MODEL_TYPE = "bpe"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 5000  # joint vocab
LOWERCASE = {'en': True, 'de': False, 'ja': False}
CHARACTER_COVERAGE = {'en': 1.0, 'de': 1.0, 'ja': 0.9995}
FEATURE_ROOT = f"fbank{N_MEL_FILTERS}"


def process(data_root, languages):
    root = Path(data_root).absolute()
    for lang in languages:
        cur_root = root / f"en-{lang}"

        # dir for filterbank (shared across splits)
        feature_root = cur_root / FEATURE_ROOT
        feature_root.mkdir(exist_ok=True)

        # normalizer
        mapping_path = Path(__file__).resolve().parent / "mapping_en.txt"
        normalizer = {
            'en':
            Normalizer(lang='en',
                       lowercase=LOWERCASE['en'],
                       remove_punc=True,
                       normalize_num=True,
                       mapping_path=mapping_path,
                       escape=True),
            lang:
            Normalizer(lang=lang,
                       lowercase=LOWERCASE[lang],
                       remove_punc=False,
                       normalize_num=False,
                       escape=True)
        }

        # Extract features
        dataset_dict = DatasetDict()

        # pylint: disable=cell-var-from-loop
        for split in SPLITS:
            print(f"Load MuST-C en-{lang} {split} dataset.")
            dataset_dict[split] = load_dataset("mustc",
                                               split=split,
                                               name=f"en-{lang}",
                                               data_dir=root.as_posix())

            def _extract(row, i):
                n_frames = 0
                try:
                    wav = torch.tensor(row['audio']['array']).unsqueeze(0)
                    npy = extract_fbank_features(
                        waveform=wav,
                        sample_rate=row['audio']['sampling_rate'],
                        output_path=feature_root / f"{row['id']}.npy",
                        n_mel_bins=N_MEL_FILTERS,
                        overwrite=False)
                    n_frames = npy.shape[0]
                except Exception as e:  # pylint: disable=broad-except
                    print(i, row['id'], e)
                return n_frames

            dataset_dict[split] = dataset_dict[split].map(
                lambda row, i: {"n_frames": _extract(row, i)},
                with_indices=True,
                num_proc=N_WORKERS,
                desc=f"Extracting log mel filter bank features ({split})...")

        # Pack features into ZIP
        print("ZIPing features...")
        create_zip(feature_root, feature_root.with_suffix(".zip"))
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"))

        # Generate TSV manifest
        dfs = []
        for split in SPLITS:
            dataset_dict[split] = dataset_dict[split].map(
                lambda row: {
                    "src": zip_manifest[row["id"]],
                    "en_utt": normalizer["en"](row["sentence"]) \
                        if split == "train" else row["sentence"],  # noqa: E131
                    f"{lang}_utt": normalizer[lang](row["translation"]) \
                        if split == "train" else row["translation"],  # noqa: E131
                    "split": split,
                },
                num_proc=N_WORKERS, desc="Generating manifest...",
                remove_columns=["file", "audio", "client_id"],
            )
            dfs.append(dataset_dict[split].to_pandas())
        all_df = pd.concat(dfs)
        save_tsv(all_df, (cur_root / 'joey_all_data.tsv'))
        del dfs
        # pylint: enable=cell-var-from-loop

        # Generate joint vocab
        print("Building joint vocab...")
        raw_textfile = cur_root / f"train.clean.en{lang}"
        train_df = all_df[all_df.split == "train"]
        train = pd.concat([train_df["en_utt"], train_df[f"{lang}_utt"]])
        write_list_to_file(raw_textfile, train.to_list())

        spm_filename = cur_root / f"spm_{SP_MODEL_TYPE}{VOCAB_SIZE}"
        # pylint: disable=consider-using-set-comprehension
        symbols = set([x[1] for x in normalizer["en"].escape + normalizer[lang].escape])
        # pylint: enable=consider-using-set-comprehension
        kwargs = {
            'model_type': SP_MODEL_TYPE,
            'vocab_size': VOCAB_SIZE,
            'character_coverage': CHARACTER_COVERAGE[lang],
            'num_workers': N_WORKERS,
            'user_defined_symbols': ','.join(symbols)
        }
        build_sp_model(raw_textfile, spm_filename, **kwargs)

        print("Saving in tsv ...")
        for split in SPLITS:
            split_df = all_df[all_df.split == split]
            for _task, _lang in [("asr", "en"), ("st", lang)]:
                # save tsv file
                save_tsv(
                    split_df.rename(columns={f'{_lang}_utt': 'trg'})[COLUMNS],
                    cur_root / f"joey_{split}_{_task}.tsv")
                # save text file (for mt pretraining)
                write_list_to_file(cur_root / f"{split}.{_lang}",
                                   split_df[f"{_lang}_utt"].to_list())
            print(f'\t{split} tsv saved.')
        print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", "-d", required=True, type=str)
    parser.add_argument("--trg_lang", default="de", required=True, type=str, nargs='+')
    args = parser.parse_args()

    process(args.data_root, args.trg_lang)


if __name__ == "__main__":
    main()
