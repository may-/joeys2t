#!/usr/bin/env python3
"""
prepare LibriSpeech dataset

Adapted from
https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/prep_librispeech_data.py

    expected dir structure:
        LibriSpeech/    # <- point this dir in --data_root in arg
        ├── fbank80/
        ├── fbank80.zip
        ├── joey_validation.clean.tsv
        ├── joey_validation.other.tsv
        ├── joey_test.clean.tsv
        ├── joey_test.other.tsv
        ├── joey_train.clean.100.tsv
        ├── joey_train.960.tsv
        ├── spm_train.clean.100_unigram5000.model
        ├── spm_train.clean.100_unigram5000.vocab
        ├── spm_train.960_unigram10000.model
        └── spm_train.960_unigram10000.vocab
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from audiodata_utils import build_sp_model, create_zip, get_zip_manifest, save_tsv
from datasets import load_dataset

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features

# pre-defined parameters
N_MEL_FILTERS = 80
N_WORKERS = 16
SP_MODEL_TYPE = "unigram"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = {"train.clean.100": 5000, "train.960": 10000}
LOWERCASE = True

SPLITS = [
    "train.clean.100",
    "train.clean.360",
    "train.other.500",
    "validation.clean",
    "validation.other",
    "test.clean",
    "test.other",
]


def process(output_root):
    out_root = Path(output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # feature_root across splits
    feature_root = out_root / f"fbank{N_MEL_FILTERS}"
    feature_root.mkdir(exist_ok=True)

    # Extract features
    print("Fetching librispeech dataset...")
    dataset_dict = load_dataset("librispeech_asr", name="all")
    # pylint: disable=cell-var-from-loop
    for split in SPLITS:

        def _extract(row, i):
            n_frames = 0
            try:
                wav = torch.tensor(row['audio']['array']).unsqueeze(0)
                npy = extract_fbank_features(waveform=wav,
                                             sample_rate=row['audio']['sampling_rate'],
                                             output_path=feature_root /
                                             f"{row['id']}.npy",
                                             n_mel_bins=N_MEL_FILTERS,
                                             overwrite=False)
                n_frames = npy.shape[0]
            except Exception as e:  # pylint: disable=broad-except
                print(i, row['id'], e)
            return n_frames

        dataset_dict[split] = dataset_dict[split].map(
            lambda row, i: {"n_frames": _extract(row, i)},
            with_indices=True,
            desc="Extracting log mel filter bank features...")

    # Pack features into ZIP
    print("ZIPing features...")
    create_zip(feature_root, feature_root.with_suffix(".zip"))
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"),
                                    npy_root=feature_root)

    # Generate TSV manifest
    dfs = []
    for split in SPLITS:
        dataset_dict[split] = dataset_dict[split].map(
            lambda row: {
                "src": zip_manifest[row["id"]],
                "trg": row["text"].lower() if LOWERCASE else row["text"],
                "split": split,
            },
            desc="Generating manifest...",
            remove_columns=["file", "audio", "speaker_id", "chapter_id", "text"],
        )
        dfs.append(dataset_dict[split].to_pandas())

    all_df = pd.concat(dfs)
    save_tsv(all_df, out_root / 'joey_all_data.tsv')
    del dfs

    print("Saving in tsv...")
    trainsets = {
        "train.clean.100": ["train.clean.100"],
        "train.960": ["train.clean.100", "train.clean.360", "train.other.500"]
    }

    for trainset, train_splits in trainsets.items():
        raw_textfile = out_root / f"{trainset}.en"
        train_df = all_df[all_df.split.isin(train_splits)]
        write_list_to_file(raw_textfile, train_df.trg.to_list())

        print(f"\tBuilding vocab of {trainset}...")
        vocab_size = VOCAB_SIZE[trainset]
        spm_filename = out_root / f"spm_{trainset}_{SP_MODEL_TYPE}{vocab_size}"
        kwargs = {
            'model_type': SP_MODEL_TYPE,
            'vocab_size': vocab_size,
            'character_coverage': 1.0,
            'num_workers': N_WORKERS
        }
        build_sp_model(raw_textfile, spm_filename, **kwargs)

        print(f"\tSaving {trainset} in tsv ...")
        for split in [
                train_splits, "validation.clean", "validation.other", "test.clean",
                "test.other"
        ]:
            if isinstance(split, list):
                df = all_df[all_df.split.isin(split)]
                out_filename = trainset
            elif isinstance(split, str):
                df = all_df[all_df.split == split]
                out_filename = split
            # df["trg"] = df["trg"].apply(
            #    lambda x: ' '.join(spm.encode(x.lower(), out_type=str)))
            tsv_file = out_root / f"joey_{out_filename}.tsv"
            save_tsv(df[["id", "src", "n_frames", "trg"]], tsv_file)
            print(f'\t{tsv_file} saved.')
    print('done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", "-d", required=True, type=str)
    args = parser.parse_args()

    process(args.data_root)


if __name__ == "__main__":
    main()
