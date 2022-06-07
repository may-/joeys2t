#!/usr/bin/env python
# coding: utf-8
"""
Prepare OpenSLR

expected dir structure:
    OpenSLR/  # <- point here in --data_root in argument
    └── SLR32/
            ├── fbank80/
            ├── fbank80.zip
            ├── joey_train_asr.tsv
            ├── joey_dev_asr.tsv
            └── joey_test_asr.tsv
"""

import argparse
from itertools import groupby
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple

from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from audiodata_utils import (
    Normalizer,
    build_sp_model,
    create_zip,
    get_zip_manifest,
    load_tsv,
    save_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features, get_n_frames

COLUMNS = ["id", "src", "n_frames", "trg", "speaker"]

SEED = 123
N_MEL_FILTERS = 80
N_WORKERS = 4  # cpu_count()
SP_MODEL_TYPE = "bpe"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 1000  # joint vocab


def process(data_root, name, language):
    root = Path(data_root).absolute()
    cur_root = root / name

    # dir for filterbank (shared across splits)
    feature_root = cur_root / f"fbank{N_MEL_FILTERS}"
    feature_root.mkdir(parents=True, exist_ok=True)

    # Extract features
    print(f"Create OpenSLR {name} {language} dataset.")

    print(f"Fetching train split ...")
    dataset = load_dataset("openslr", name=name, split="train")

    print(f"Extracting log mel filter bank features ...")
    for instance in tqdm(dataset):
        utt_id = Path(instance['path']).stem
        try:
            extract_fbank_features(
                waveform=torch.from_numpy(instance['audio']['array']).unsqueeze(0),
                sample_rate=instance['audio']['sampling_rate'],
                output_path=(feature_root / f'{utt_id}.npy'),
                n_mel_bins=N_MEL_FILTERS,
                overwrite=False)
        except Exception as e:
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
        lang = Path(instance['path']).parents[2].name
        n_frames = np.load(feature_root / f'{utt_id}.npy').shape[0]
        try:
            all_data.append({
                "id": utt_id,
                "src": zip_manifest[str(utt_id)],
                "n_frames": n_frames,
                "trg": instance['sentence'],
                "lang": lang,
            })
        except Exception as e:
            print(f'Skip instance {utt_id}.', e)
            continue

    all_df = pd.DataFrame.from_records(all_data)
    save_tsv(all_df, cur_root / f"all_data.tsv")
    all_df = all_df[all_df.lang == language]

    # Split the data into train and test set and save the splits in tsv
    np.random.seed(SEED)
    probs = np.random.rand(len(all_df))
    mask = {}
    mask['train'] = probs < 0.995
    mask['dev'] = (0.995 <= probs) & (probs < 0.998)
    mask['test'] = 0.998 <= probs

    for split in ['train', 'dev', 'test']:
        split_df = all_df[mask[split]]
        # save tsv
        save_tsv(split_df, cur_root / f"{split}.tsv")
        # save plain txt
        write_list_to_file(cur_root / f"{split}.{language}", split_df['trg'].to_list())
        print(split, len(split_df))

    # Generate joint vocab
    print("Building joint vocab...")
    kwargs = {'model_type': SP_MODEL_TYPE,
              'vocab_size': VOCAB_SIZE,
              'character_coverage': 1.0,
              'num_workers': N_WORKERS}
    build_sp_model(cur_root / f"train.{language}", cur_root / "spm_bpe1000", **kwargs)
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", "-d", required=True, type=str)
    parser.add_argument("--dataset_name", "-n", default="SLT32", required=True, type=str)
    parser.add_argument("--language", "-l", default="af_za", required=True, type=str)
    args = parser.parse_args()

    process(args.data_root, args.dataset_name, args.language)


if __name__ == "__main__":
    main()
