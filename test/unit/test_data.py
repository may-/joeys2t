import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from joeynmt.data import load_data
from joeynmt.datasets import (
    RandomSubsetSampler,
    SentenceBatchSampler,
    TokenBatchSampler,
)


class TestDataSampler(unittest.TestCase):

    def setUp(self):

        # minimal data config
        data_cfg = {
            "train": "test/data/toy/train",
            "dev": "test/data/toy/dev",
            "test": "test/data/toy/test",
            "src": {
                "lang": "de",
                "level": "word",
                "lowercase": False,
                "max_length": 10,
            },
            "trg": {
                "lang": "en",
                "level": "word",
                "lowercase": False,
                "max_length": 10,
            },
            "sample_train_subset": -1,
            "sample_dev_subset": 10,
            "dataset_type": "plain",
            "special_symbols": SimpleNamespace(
                **{
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "sep_token": "<sep>",
                    "unk_id": 0,
                    "pad_id": 1,
                    "bos_id": 2,
                    "eos_id": 3,
                    "sep_id": 4,
                    "lang_tags": ["<de>", "<en>"],
                }
            ),
        }

        # load toy data
        _, self.trg_vocab, self.train_data, self.dev_data, self.test_data = load_data(
            data_cfg, datasets=["train", "dev", "test"], task="MT"
        )

        # random seed
        self.seed = 42

    def testSentenceBatchSampler(self):
        batch_size = 10  # 10 sentences

        # load all sents here, filtering happens in iterator
        self.assertEqual(len(self.train_data), 1000)

        # make batches by number of sentences
        train_loader = self.train_data.make_iter(
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=True,
            seed=self.seed,
            eos_index=self.trg_vocab.eos_index,
            pad_index=self.trg_vocab.pad_index,
            device=torch.device("cpu"),
            num_workers=0,
        )
        self.assertTrue(isinstance(train_loader, DataLoader))
        self.assertTrue(isinstance(train_loader.batch_sampler, SentenceBatchSampler))
        self.assertTrue(
            isinstance(train_loader.batch_sampler.sampler, RandomSubsetSampler)
        )  # shuffle=True
        initial_seed = train_loader.batch_sampler.sampler.generator.initial_seed()
        self.assertEqual(initial_seed, self.seed)
        self.assertEqual(len(train_loader), 100)  # num_samples // batch_size

        # reset seed
        train_loader.batch_sampler.set_seed(self.seed + 0)
        batch_idx = next(train_loader.batch_sampler.sampler.__iter__())
        self.assertEqual(batch_idx, 542)

        _ = next(train_loader.batch_sampler.sampler.__iter__())
        _ = next(train_loader.batch_sampler.sampler.__iter__())
        train_loader.batch_sampler.set_seed(self.seed + 0)
        batch_idx = next(train_loader.batch_sampler.sampler.__iter__())
        self.assertEqual(batch_idx, 542)

        # subsampling
        train_loader.batch_sampler.sampler.data_source.random_subset = 500
        train_loader.batch_sampler.set_seed(self.seed + 20)
        self.assertEqual(len(train_loader), 50)  # num_samples // batch_size

        with self.assertRaises(AssertionError) as e:
            self.test_data.reset_indices(random_subset=2000)
            self.assertEqual(
                "Can only subsample from train or dev set larger than 2000.",
                str(e.exception)
            )

        # make batches by number of sentences
        dev_loader = self.dev_data.make_iter(
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=False,
            seed=self.seed,
            eos_index=self.trg_vocab.eos_index,
            pad_index=self.trg_vocab.pad_index,
            device=torch.device("cpu"),
            num_workers=0,
        )
        self.assertTrue(isinstance(dev_loader, DataLoader))
        self.assertTrue(isinstance(dev_loader.batch_sampler, SentenceBatchSampler))
        self.assertTrue(
            isinstance(dev_loader.batch_sampler.sampler, RandomSubsetSampler)
        )  # shuffle=False

        # reset seed
        self.assertEqual(
            dev_loader.batch_sampler.sampler.data_source.indices,
            [0, 1, 2, 4, 6, 10, 11, 14, 15, 18]
        )

        dev_loader.batch_sampler.set_seed(self.seed + 10)
        self.assertEqual(
            dev_loader.batch_sampler.sampler.data_source.indices,
            [1, 2, 6, 8, 9, 11, 12, 13, 16, 17]
        )

        dev_loader.batch_sampler.set_seed(self.seed + 20)
        self.assertEqual(
            dev_loader.batch_sampler.sampler.data_source.indices,
            [1, 2, 3, 5, 7, 9, 13, 14, 18, 19]
        )

    def testTokenBatchSampler(self):
        batch_size = 50  # 50 tokens

        self.assertEqual(len(self.test_data), 20)

        # make batches by number of tokens
        test_loader = self.test_data.make_iter(
            batch_size=batch_size,
            batch_type="token",
            shuffle=False,
            seed=self.seed,
            eos_index=self.trg_vocab.eos_index,
            pad_index=self.trg_vocab.pad_index,
            device=torch.device("cpu"),
            num_workers=0,
        )
        self.assertTrue(isinstance(test_loader, DataLoader))
        self.assertEqual(test_loader.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(test_loader.batch_sampler, TokenBatchSampler))
        self.assertTrue(
            isinstance(test_loader.batch_sampler.sampler, RandomSubsetSampler)
        )  # shuffle=False

        with self.assertRaises(NotImplementedError) as e:
            len(test_loader)
            self.assertEqual("NotImplementedError", str(e.exception))

        # subsampling
        with self.assertRaises(AssertionError) as e:
            self.test_data.reset_indices(random_subset=10)
            self.assertEqual(
                "Can only subsample from train or dev set larger than 10.",
                str(e.exception)
            )


class TestSpeechData(unittest.TestCase):

    def setUp(self):
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "train": "test/data/speech/test",
            "test": "test/data/speech/test",
            "src": {
                "lang": "en",
                "level": "frame",
                "num_freq": 80,
                "max_length": 500,
                "tokenizer_type": "speech",
            },
            "trg": {
                "lang": "en", "level": "char", "lowercase": True, "max_length": 50,
                "voc_file": "test/data/speech/char.txt"
            },
            "dataset_type": "speech",
            "special_symbols": SimpleNamespace(
                **{
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "sep_token": None,
                    "unk_id": 0,
                    "pad_id": 1,
                    "bos_id": 2,
                    "eos_id": 3,
                    "sep_id": None,
                    "lang_tags": [],
                }
            ),
        }

    def testIteratorBatchShape(self):

        current_cfg = self.data_cfg.copy()

        # load speech data
        _, trg_vocab, train_data, _, test_data = load_data(
            current_cfg, datasets=["train", "test"], task="S2T"
        )

        # no filtering in data loading
        self.assertEqual(len(train_data), 10)
        self.assertEqual(len(test_data), 10)

        # make train batches (filtered by length)
        train_iter = iter(
            train_data.make_iter(
                batch_size=2,
                batch_type="sentence",
                shuffle=True,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            )
        )
        _ = next(train_iter)  # skip a batch
        _ = next(train_iter)  # skip another batch
        train_batch = next(train_iter)

        self.assertEqual(train_batch.src.shape, (2, 310, 80))
        self.assertEqual(train_batch.trg.shape, (2, 42))

        # make test batches (not filtered by length)
        test_iter = iter(
            test_data.make_iter(
                batch_size=2,
                batch_type="sentence",
                shuffle=False,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            )
        )
        _ = next(test_iter)  # skip a batch
        _ = next(test_iter)  # skip another batch
        test_batch = next(test_iter)

        self.assertEqual(test_batch.src.shape, (2, 500, 80))
        self.assertEqual(test_batch.trg.shape, (2, 169))
