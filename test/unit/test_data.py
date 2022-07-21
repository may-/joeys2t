import unittest

import torch

from joeynmt.data import load_data, make_data_iter


class TestData(unittest.TestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.test_path = "test/data/toy/test"
        self.max_length = 10
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "task": "MT",
            "train": self.train_path,
            "dev": self.dev_path,
            "src": {
                "lang": "de",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
            },
            "trg": {
                "lang": "en",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
            },
            "dataset_type": "plain",
        }

    def testIteratorBatchType(self):

        current_cfg = self.data_cfg.copy()

        # load toy data
        _, trg_vocab, train_data, _, _ = load_data(current_cfg, datasets=["train"])

        # make batches by number of sentences
        train_iter = iter(
            make_data_iter(
                train_data,
                batch_size=10,
                batch_type="sentence",
                shuffle=True,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            ))
        batch = next(train_iter)

        self.assertEqual(batch.src.shape[0], 10)
        self.assertEqual(batch.trg.shape[0], 10)

        # make batches by number of tokens
        train_iter = iter(
            make_data_iter(
                train_data,
                batch_size=100,
                batch_type="token",
                shuffle=True,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            ))
        _ = next(train_iter)  # skip a batch
        _ = next(train_iter)  # skip another batch
        batch = next(train_iter)

        self.assertEqual(batch.src.shape, (9, 10))
        self.assertLessEqual(batch.ntokens, 77)


class TestSpeechData(unittest.TestCase):

    def setUp(self):
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "task": "s2t",
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
                "lang": "en",
                "level": "char",
                "lowercase": True,
                "max_length": 50,
                "voc_file": "test/data/speech/char.txt"
            },
            "dataset_type": "speech",
        }

    def testIteratorBatchShape(self):

        current_cfg = self.data_cfg.copy()

        # load speech data
        _, trg_vocab, train_data, _, test_data = load_data(current_cfg,
                                                           datasets=["train", "test"])

        # no filtering in data loading
        self.assertEqual(len(train_data), 10)
        self.assertEqual(len(test_data), 10)

        # make train batches (filtered by length)
        train_iter = iter(
            make_data_iter(
                train_data,
                batch_size=2,
                batch_type="sentence",
                shuffle=True,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            ))
        _ = next(train_iter)  # skip a batch
        _ = next(train_iter)  # skip another batch
        train_batch = next(train_iter)

        self.assertEqual(train_batch.src.shape, (2, 310, 80))
        self.assertEqual(train_batch.trg.shape, (2, 42))

        # make test batches (not filtered by length)
        test_iter = iter(
            make_data_iter(
                test_data,
                batch_size=2,
                batch_type="sentence",
                shuffle=False,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            ))
        _ = next(test_iter)  # skip a batch
        _ = next(test_iter)  # skip another batch
        test_batch = next(test_iter)

        self.assertEqual(test_batch.src.shape, (2, 500, 80))
        self.assertEqual(test_batch.trg.shape, (2, 169))
