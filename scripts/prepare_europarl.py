#!/usr/bin/env python3
"""
Prepare Europarl_ST

Depends on Huggingface's datasets library.
As of 16. July 2022, Europarl ST is not in the official Huggingface's datasets repo.
Pull the script:
https://github.com/may-/datasets/blob/main/datasets/europarl_st/europarl_st.py

Expected dir structure:
    Europarl_ST    # <- point here in --data_root in arg
        └── de
            └── en
                ├── fbank80
                │   ├── en.20081009.3.4-013_1.npy
                │   ├── [...]
                │   └── en.20100707.26.3-328_1.npy
                ├── fbank80.zip
                ├── joey_train_{asr|st}.tsv
                ├── joey_validation_{asr|st}.tsv
                ├── joey_test_{asr|st}.tsv
                ├── {train|validation|test}.en
                └── {train|validation|test}.de
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from audiodata_utils import (
    SPECIAL_SYMBOLS,
    build_sp_model,
    create_zip,
    get_zip_manifest,
    save_tsv,
)
from datasets import DatasetDict, load_dataset

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features

COLUMNS = ["id", "src", "n_frames", "trg"]
SPLITS = ["train", "train.noisy", "validation", "test"]

N_MEL_FILTERS = 80
N_WORKERS = 16  # cpu_count()
SP_MODEL_TYPE = "bpe"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 5000  # joint vocab
LOWERCASE = {'en': False, 'de': False}
FEATURE_ROOT = f"fbank{N_MEL_FILTERS}"


def process(data_root, src_lang, trg_lang):
    root = Path(data_root).absolute()
    cur_root = root / src_lang / trg_lang

    # dir for filterbank (shared across splits)
    feature_root = cur_root / FEATURE_ROOT
    feature_root.mkdir(exist_ok=True)

    # Extract features
    dataset_dict = DatasetDict()
    # pylint: disable=cell-var-from-loop
    for split in SPLITS:
        print(f"Load Europarl ST v1.1 {src_lang}-{trg_lang} {split} dataset.")
        dataset_dict[split] = load_dataset(
            "europarl_st",
            split=split,
            name=f"{src_lang}-{trg_lang}",
            data_dir=root.as_posix()
        )

        def _extract(row, i):
            n_frames = 0
            try:
                wav = torch.tensor(row['audio']['array']).unsqueeze(0)
                npy = extract_fbank_features(
                    waveform=wav,
                    sample_rate=row['audio']['sampling_rate'],
                    output_path=feature_root / f"{row['id']}.npy",
                    n_mel_bins=N_MEL_FILTERS,
                    overwrite=True
                )
                n_frames = npy.shape[0]
            except Exception as e:  # pylint: disable=broad-except
                print(i, row['id'], e)
            return n_frames

        dataset_dict[split] = dataset_dict[split].map(
            lambda row, i: {"n_frames": _extract(row, i)},
            with_indices=True,
            desc=f"Extracting log mel filter bank features ({split})..."
        )

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
                f"{src_lang}_utt": row["sentence"].lower()
                if LOWERCASE[src_lang] else row["sentence"],
                f"{trg_lang}_utt": row["translation"].lower()
                if LOWERCASE[src_lang] else row["translation"],
                "split": split,
            },
            desc="Generating manifest...",
            remove_columns=["file", "audio"],
        )
        dfs.append(dataset_dict[split].to_pandas())
    all_df = pd.concat(dfs)
    save_tsv(all_df, (cur_root / 'joey_all_data.tsv'))
    del dfs

    # Generate joint vocab
    print("Building joint vocab...")
    raw_textfile = cur_root / f"train.{src_lang}{trg_lang}"
    train_df = all_df[all_df.split.str.startswith("train")]
    train = pd.concat([train_df[f"{src_lang}_utt"], train_df[f"{trg_lang}_utt"]])
    write_list_to_file(raw_textfile, train.to_list())

    spm_filename = cur_root / f"spm_{SP_MODEL_TYPE}{VOCAB_SIZE}"
    kwargs = {
        'model_type': SP_MODEL_TYPE,
        'vocab_size': VOCAB_SIZE,
        'character_coverage': 1.0,
        'num_workers': N_WORKERS,
    }
    cfg = SimpleNamespace(**SPECIAL_SYMBOLS)
    build_sp_model(raw_textfile, spm_filename, cfg, **kwargs)

    print("Saving in tsv ...")
    for split in [s for s in SPLITS if "." not in s]:
        split_df = all_df[all_df.split.str.startswith(split)]
        for _task, _lang in [("asr", src_lang), ("st", trg_lang)]:
            # save tsv file
            save_tsv(
                split_df.rename(columns={f'{_lang}_utt': 'trg'})[COLUMNS],
                cur_root / f"joey_{split}_{_task}.tsv"
            )
            # save text file (for mt pretraining)
            write_list_to_file(
                cur_root / f"{split}.{_lang}", split_df[f"{_lang}_utt"].to_list()
            )
        print(f'\t{split} tsv saved.')
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--src-lang", default="de", required=True, type=str)
    parser.add_argument("--trg-lang", default="en", required=True, type=str)
    args = parser.parse_args()

    process(args.data_root, args.src_lang, args.trg_lang)


if __name__ == "__main__":
    main()
