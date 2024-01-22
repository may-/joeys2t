#!/usr/bin/env python
# coding: utf-8
"""
Prepare OpenSLR

expected dir structure:
    OpenSLR/  # <- point here in --data_root in argument
    └── SLR70/
            ├── fbank80/
            ├── fbank80.zip
            ├── joey_train_asr.tsv
            ├── joey_dev_asr.tsv
            └── joey_test_asr.tsv
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from audiodata_utils import (
    SPECIAL_SYMBOLS,
    build_sp_model,
    create_zip,
    get_zip_manifest,
    save_tsv,
)
from datasets import load_dataset
from tqdm import tqdm

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features

COLUMNS = ["id", "src", "n_frames", "trg"]

SEED = 123
N_MEL_FILTERS = 80
N_WORKERS = 4  # cpu_count()
SP_MODEL_TYPE = "bpe"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 1000  # joint vocab


def process(data_root, name):
    root = Path(data_root).absolute()
    cur_root = root / name

    # dir for filterbank (shared across splits)
    feature_root = cur_root / f"fbank{N_MEL_FILTERS}"
    feature_root.mkdir(parents=True, exist_ok=True)

    # Extract features
    print(f"Create OpenSLR {name} dataset.")

    print("Fetching train split ...")
    dataset = load_dataset("openslr", name=name, split="train")

    print("Extracting log mel filter bank features ...")
    for instance in tqdm(dataset):
        utt_id = Path(instance['path']).stem
        try:
            extract_fbank_features(
                waveform=torch.from_numpy(instance['audio']['array']).unsqueeze(0),
                sample_rate=instance['audio']['sampling_rate'],
                output_path=(feature_root / f'{utt_id}.npy'),
                n_mel_bins=N_MEL_FILTERS,
                overwrite=False
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f'Skip instance {utt_id}.', e)
            continue

    # Pack features into ZIP
    print("ZIPing features...")
    create_zip(feature_root, feature_root.with_suffix(".zip"))

    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"))

    # Generate TSV manifest
    print("Generating manifest...")
    all_data = []

    for instance in tqdm(dataset):
        utt_id = Path(instance['path']).stem
        n_frames = np.load(feature_root / f'{utt_id}.npy').shape[0]
        try:
            all_data.append({
                "id": utt_id,
                "src": zip_manifest[str(utt_id)],
                "n_frames": n_frames,
                "trg": instance['sentence'],
            })
        except Exception as e:  # pylint: disable=broad-except
            print(f'Skip instance {utt_id}.', e)
            continue

    all_df = pd.DataFrame.from_records(all_data)
    save_tsv(all_df, cur_root / "joey_all_data.tsv")

    # Split the data into train and test set and save the splits in tsv
    np.random.seed(SEED)
    probs = np.random.rand(len(all_df))
    mask = {}
    mask['train'] = probs < 0.995
    mask['dev'] = (probs >= 0.995) & (probs < 0.998)
    mask['test'] = probs >= 0.998

    for split in ['train', 'dev', 'test']:
        split_df = all_df[mask[split]]
        # save tsv
        save_tsv(split_df, cur_root / f"{split}.tsv")
        # save plain txt
        write_list_to_file(cur_root / f"{split}.txt", split_df['trg'].to_list())
        print(split, len(split_df))

    # Generate joint vocab
    print("Building joint vocab...")
    kwargs = {
        'model_type': SP_MODEL_TYPE, 'vocab_size': VOCAB_SIZE,
        'character_coverage': 1.0, 'num_workers': N_WORKERS
    }
    cfg = SimpleNamespace(**SPECIAL_SYMBOLS)
    build_sp_model(
        cur_root / "train.txt", cur_root / f"spm_bpe{VOCAB_SIZE}", cfg, **kwargs
    )
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--dataset-name", default="SLT70", required=True, type=str)
    args = parser.parse_args()

    process(args.data_root, args.dataset_name)


if __name__ == "__main__":
    main()
