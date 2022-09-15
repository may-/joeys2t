# coding: utf-8
"""
Tokenizer module
"""
import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np
import sentencepiece as sp
from sacrebleu.metrics.bleu import _get_tokenizer
from subword_nmt import apply_bpe

from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from joeynmt.data_augmentation import CMVN, SpecAugment
from joeynmt.helpers import (
    ConfigurationError,
    remove_extra_spaces,
    remove_punctuation,
    unicode_normalize,
)
from joeynmt.helpers_for_audio import get_features

logger = logging.getLogger(__name__)


class BasicTokenizer:
    SPACE = chr(32)  # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default
    SPECIALS = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]

    def __init__(
        self,
        level: str = "word",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        # pylint: disable=unused-argument
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize

        # filter by length
        self.max_length = max_length
        self.min_length = min_length

        # pretokenizer
        self.pretokenizer = kwargs.get("pretokenizer", "none").lower()
        assert self.pretokenizer in ["none", "moses"], \
            "Currently, we support moses tokenizer only."
        # sacremoses
        if self.pretokenizer == "moses":
            try:
                from sacremoses import (  # pylint: disable=import-outside-toplevel
                    MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer,
                )

                # sacremoses package has to be installed.
                # https://github.com/alvations/sacremoses
            except ImportError as e:
                logger.error(e)
                raise ImportError from e

            self.lang = kwargs.get("lang", "en")
            self.moses_tokenizer = MosesTokenizer(lang=self.lang)
            self.moses_detokenizer = MosesDetokenizer(lang=self.lang)
            if self.normalize:
                self.moses_normalizer = MosesPunctNormalizer()

    def pre_process(self, raw_input: str) -> str:
        """
        Pre-process text
            - ex.) Lowercase, Normalize, Remove emojis,
                Pre-tokenize(add extra white space before punc) etc.
            - applied for all inputs both in training and inference.
        """
        assert isinstance(raw_input, str) and raw_input.strip() != "", \
            "The input sentence is empty! Please make sure " \
            "that you are feeding a valid input."

        if self.normalize:
            raw_input = remove_extra_spaces(unicode_normalize(raw_input))

        if self.pretokenizer == "moses":
            if self.normalize:
                raw_input = self.moses_normalizer.normalize(raw_input)
            raw_input = self.moses_tokenizer.tokenize(raw_input, return_str=True)

        if self.lowercase:
            raw_input = raw_input.lower()

        # ensure the string is not empty.
        assert raw_input is not None and len(raw_input) > 0, raw_input
        return raw_input

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize single sentence"""
        if self.level == "word":
            sequence = raw_input.split(self.SPACE)
        elif self.level == "char":
            sequence = list(raw_input.replace(self.SPACE, self.SPACE_ESCAPE))

        if is_train and self._filter_by_length(len(sequence)):
            return None
        return sequence

    def _filter_by_length(self, length: int) -> bool:
        """
        Check if the given seq length is out of the valid range.

        :param length: (int) number of tokens
        :return: True if the length is invalid(= to be filtered out), False if valid.
        """
        return length > self.max_length > 0 or self.min_length > length > 0

    def _remove_special(self, sequence: List[str], generate_unk: bool = False):
        specials = self.SPECIALS[:-1] if generate_unk else self.SPECIALS
        return [token for token in sequence if token not in specials]

    def post_process(self,
                     sequence: Union[List[str], str],
                     generate_unk: bool = True) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            sequence = self._remove_special(sequence, generate_unk=generate_unk)
            if self.level == "word":
                if self.pretokenizer == "moses":
                    sequence = self.moses_detokenizer.detokenize(sequence)
                else:
                    sequence = self.SPACE.join(sequence)
            elif self.level == "char":
                sequence = "".join(sequence).replace(self.SPACE_ESCAPE, self.SPACE)

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, itos: List[str]) -> None:
        """
        Set vocab
        :param itos: (list) indices-to-symbols mapping
        """
        pass  # pylint: disable=unnecessary-pass

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), "
                f"pretokenizer={self.pretokenizer})")


class SentencePieceTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length, **kwargs)
        assert self.level == "bpe"

        self.model_file: Path = Path(kwargs["model_file"])
        assert self.model_file.is_file(), f"model file {self.model_file} not found."

        self.spm = sp.SentencePieceProcessor()
        self.spm.load(kwargs["model_file"])

        self.nbest_size: int = kwargs.get("nbest_size", 5)
        self.alpha: float = kwargs.get("alpha", 0.0)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize"""
        if is_train and self.alpha > 0:
            tokenized = self.spm.sample_encode_as_pieces(
                raw_input,
                nbest_size=self.nbest_size,
                alpha=self.alpha,
            )
        else:
            tokenized = self.spm.encode(raw_input, out_type=str)

        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(self,
                     sequence: Union[List[str], str],
                     generate_unk: bool = True) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            sequence = self._remove_special(sequence, generate_unk=generate_unk)

            # Decode back to str
            sequence = self.spm.decode(sequence)
            sequence = sequence.replace(self.SPACE_ESCAPE, self.SPACE).strip()

        # Apply moses detokenizer
        if self.pretokenizer == "moses":
            sequence = self.moses_detokenizer.detokenize(sequence.split())

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, itos: List[str]) -> None:
        """Set vocab"""
        self.spm.SetVocabulary(itos)

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy confg file to model_dir"""
        if (model_dir / self.model_file.name).is_file():
            logger.warning(
                "%s already exists. Stop copying.",
                (model_dir / self.model_file.name).as_posix(),
            )
        shutil.copy2(self.model_file, (model_dir / self.model_file.name).as_posix())

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), "
                f"pretokenizer={self.pretokenizer}, "
                f"tokenizer={self.spm.__class__.__name__}, "
                f"nbest_size={self.nbest_size}, alpha={self.alpha})")


class SubwordNMTTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length, **kwargs)
        assert self.level == "bpe"

        self.codes: Path = Path(kwargs["codes"])
        assert self.codes.is_file(), f"codes file {self.codes} not found."

        self.separator: str = kwargs.get("separator", "@@")
        bpe_parser = apply_bpe.create_parser()
        bpe_args = bpe_parser.parse_args(
            ["--codes", kwargs["codes"], "--separator", self.separator])
        self.bpe = apply_bpe.BPE(
            bpe_args.codes,
            bpe_args.merges,
            bpe_args.separator,
            None,
            bpe_args.glossaries,
        )
        self.dropout: float = kwargs.get("dropout", 0.0)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize"""
        dropout = self.dropout if is_train else 0.0
        tokenized = self.bpe.process_line(raw_input, dropout).strip().split()
        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(self,
                     sequence: Union[List[str], str],
                     generate_unk: bool = True) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            sequence = self._remove_special(sequence, generate_unk=generate_unk)

            # Remove separators, join with spaces
            sequence = self.SPACE.join(sequence).replace(self.separator + self.SPACE,
                                                         "")
            # Remove final merge marker.
            if sequence.endswith(self.separator):
                sequence = sequence[:-len(self.separator)]

        # Moses detokenizer
        if self.pretokenizer == "moses":
            sequence = self.moses_detokenizer.detokenize(sequence.split())

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, itos: List[str]) -> None:
        """Set vocab"""
        vocab = set(itos) - set(self.SPECIALS)
        self.bpe.vocab = vocab

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy confg file to model_dir"""
        shutil.copy2(self.codes, (model_dir / self.codes.name).as_posix())

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), "
                f"pretokenizer={self.pretokenizer}, "
                f"tokenizer={self.bpe.__class__.__name__}, "
                f"separator={self.separator}, dropout={self.dropout})")


class FastBPETokenizer(SubwordNMTTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        try:
            import fastBPE  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            logger.error(e)
            raise ImportError from e
        super(SubwordNMTTokenizer, self).__init__(level, lowercase, normalize,
                                                  max_length, min_length, **kwargs)
        assert self.level == "bpe"

        # set codes file path
        self.codes: Path = Path(kwargs["codes"])
        assert self.codes.is_file(), f"codes file {self.codes} not found."

        # instantiate fastBPE object
        self.bpe = fastBPE.fastBPE(self.codes.as_posix())
        self.separator = "@@"
        self.dropout = 0.0

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        # fastBPE.apply()
        tokenized = self.bpe.apply([raw_input])
        tokenized = tokenized[0].strip().split()

        # check if the input sequence length stays within the valid length range
        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def set_vocab(self, itos: List[str]) -> None:
        pass


class SpeechProcessor:
    """SpeechProcessor"""

    def __init__(
        self,
        level: str = "frame",
        num_freq: int = 80,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        self.level = level
        self.num_freq = num_freq
        self.normalize = normalize

        # filter by length
        self.max_length = max_length
        self.min_length = min_length

        self.specaugment: Callable = SpecAugment(**kwargs["specaugment"]) \
            if "specaugment" in kwargs else None
        self.cmvn: Callable = CMVN(**kwargs["cmvn"]) if "cmvn" in kwargs else None
        self.root_path = ""  # assigned later by dataset.__init__()

    def __call__(self, line: str, is_train: bool = False) -> np.ndarray:
        """
        get features

        :param line: path to audio file or pre-extracted features
        :param is_train:

        :return: spectrogram in shape (num_frames, num_freq)
        """
        # lookup
        item = get_features(self.root_path, line)  # shape = (num_frames, num_freq)

        num_frames, num_freq = item.shape
        assert num_freq == self.num_freq

        if self._filter_too_short_item(num_frames):
            # A too short sequence cannot be convolved!
            # -> filter out anyway even in test-dev set.
            return None
        if self._filter_too_long_item(num_frames):
            # Don't use too long sequence in training.
            if is_train:  # pylint: disable=no-else-return
                return None
            else:  # in test, truncate the sequence
                item = item[:self.max_length, :]
                num_frames = item.shape[0]
                assert num_frames <= self.max_length

        # cmvn / specaugment
        # pylint: disable=not-callable
        if self.cmvn and self.cmvn.before:
            item = self.cmvn(item)
        if is_train and self.specaugment:
            item = self.specaugment(item)
        if self.cmvn and not self.cmvn.before:
            item = self.cmvn(item)
        return item

    def _filter_too_short_item(self, length: int) -> bool:
        return self.min_length > length > 0

    def _filter_too_long_item(self, length: int) -> bool:
        return length > self.max_length > 0

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"level={self.level}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), "
                f"cmvn={self.cmvn}, specaugment={self.specaugment})")


class EvaluationTokenizer(BasicTokenizer):
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers in
    sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are applied
    after sacreBLEU tokenization.

    :param level: (str) tokenization level. {"word", "bpe", "char"}
    :param lowercase: (bool) lowercase the text.
    :param tokenize: (str) the type of sacreBLEU tokenizer to apply.
    """
    ALL_TOKENIZER_TYPES = ["none", "13a", "intl", "zh", "ja-mecab"]

    def __init__(self, lowercase: bool = False, tokenize: str = "13a", **kwargs):
        super().__init__(level="word",
                         lowercase=lowercase,
                         normalize=False,
                         max_length=-1,
                         min_length=-1)

        assert tokenize in self.ALL_TOKENIZER_TYPES, f"`{tokenize}` not supported."
        self.tokenizer = _get_tokenizer(tokenize)()
        self.no_punc = kwargs.get("no_punc", False)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        tokenized = self.tokenizer(raw_input)

        if self.lowercase:
            tokenized = tokenized.lower()

        # Remove punctuation (apply this after tokenization!)
        if self.no_punc:
            tokenized = remove_punctuation(tokenized, space=self.SPACE)
        return tokenized.split()

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, "
                f"tokenizer={self.tokenizer}, "
                f"no_punc={self.no_punc})")


def _build_tokenizer(cfg: Dict) -> BasicTokenizer:
    """Builds tokenizer."""
    tokenizer = None
    tokenizer_cfg = cfg.get("tokenizer_cfg", {})

    # assign lang for moses tokenizer
    if tokenizer_cfg.get("pretokenizer", "none") == "moses":
        tokenizer_cfg["lang"] = cfg["lang"]

    if cfg["level"] in ["word", "char"]:
        tokenizer = BasicTokenizer(
            level=cfg["level"],
            lowercase=cfg.get("lowercase", False),
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
            **tokenizer_cfg,
        )
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type", cfg.get("bpe_type", "sentencepiece"))
        if tokenizer_type == "sentencepiece":
            assert "model_file" in tokenizer_cfg
            tokenizer = SentencePieceTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        elif tokenizer_type == "subword-nmt":
            assert "codes" in tokenizer_cfg
            tokenizer = SubwordNMTTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        elif tokenizer_type == "fastbpe":
            assert "codes" in tokenizer_cfg
            tokenizer = FastBPETokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        else:
            raise ConfigurationError(f"{tokenizer_type}: Unknown tokenizer type.")
    elif cfg["level"] == "frame":
        tokenizer = SpeechProcessor(
            level=cfg["level"],
            num_freq=cfg["num_freq"],
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
            **tokenizer_cfg,
        )
    else:
        raise ConfigurationError(f"{cfg['level']}: Unknown tokenization level.")
    return tokenizer


def build_tokenizer(data_cfg: Dict) -> Dict[str, BasicTokenizer]:
    task = data_cfg.get("task", "MT").upper()
    src_lang = data_cfg["src"]["lang"] if task == "MT" else "src"
    trg_lang = data_cfg["trg"]["lang"] if task == "MT" else "trg"
    tokenizer = {
        src_lang: _build_tokenizer(data_cfg["src"]),
        trg_lang: _build_tokenizer(data_cfg["trg"]),
    }
    log_str = "Tokenizer" if task == "MT" else "SpeechProcessor"
    logger.info("%s %s: %s", src_lang, log_str, tokenizer[src_lang])
    logger.info("%s Tokenizer: %s", trg_lang, tokenizer[trg_lang])
    return tokenizer
